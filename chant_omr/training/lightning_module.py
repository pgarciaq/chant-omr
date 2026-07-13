"""PyTorch Lightning module for training the ChantOMR model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from chant_omr.data.dataset import build_dataloaders, build_datasets
from chant_omr.inference.checkpoint import load_config, load_model_weights_into_module
from chant_omr.model.chant_omr_model import ChantOMR, ChantOMRConfig, build_model
from chant_omr.model.tokenizer import TOKENIZER_FILENAME, GABCTokenizer
from chant_omr.training.scheduled_sampling import build_scheduled_decoder_input
from chant_omr.training.xpu_strategy import SingleXPUStrategy, xpu_is_available


class ChantOMRLightningModule(LightningModule):
    """Lightning training module with teacher-forcing cross-entropy loss."""

    def __init__(
        self,
        model: ChantOMR,
        *,
        pad_token_id: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_fraction: float = 0.05,
        gradient_clip: float = 1.0,
        scheduled_sampling_prob: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.gradient_clip = gradient_clip
        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    def _compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_scheduled_sampling: bool | None = None,
    ) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        encoder_attention_mask = batch.get("encoder_attention_mask")

        if input_ids.shape[1] < 2:
            raise ValueError("input_ids must contain at least two tokens for teacher forcing")

        gold_decoder_input = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        decoder_mask = attention_mask[:, :-1]

        should_sample = (
            use_scheduled_sampling
            if use_scheduled_sampling is not None
            else self.training and self.scheduled_sampling_prob > 0
        )
        decoder_input = gold_decoder_input
        if should_sample:
            with torch.no_grad():
                teacher_logits = self.model(
                    batch["pixel_values"],
                    gold_decoder_input,
                    attention_mask=decoder_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            decoder_input = build_scheduled_decoder_input(
                gold_decoder_input,
                teacher_logits,
                attention_mask=decoder_mask,
                sampling_prob=self.scheduled_sampling_prob,
            )

        logits = self.model(
            batch["pixel_values"],
            decoder_input,
            attention_mask=decoder_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.trainer is None:
            return optimizer

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * self.warmup_fraction))
        warmup_steps = min(warmup_steps, max(1, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _effective_accelerator(accelerator: str, gpus: int) -> str:
    """Resolve ``auto`` to ``cuda``, ``xpu``, or ``cpu``."""
    if accelerator == "cuda":
        return "cuda"
    if accelerator == "xpu":
        return "xpu"
    if accelerator == "cpu":
        return "cpu"
    if accelerator != "auto":
        raise ValueError(f"unsupported accelerator: {accelerator!r} (use auto, cuda, xpu, cpu)")
    if gpus > 0 and torch.cuda.is_available():
        return "cuda"
    if gpus > 0 and xpu_is_available():
        return "xpu"
    return "cpu"


def resolve_precision(config_precision: str, *, accelerator: str) -> str:
    """Pick a safe precision for the selected accelerator."""
    if accelerator == "cuda":
        return config_precision
    if accelerator == "xpu":
        # Intel Arc A-series lacks FP64 for GradScaler; prefer true bf16 over mixed AMP.
        if config_precision in {"16-mixed", "bf16-mixed"}:
            return "bf16-true"
        if config_precision.endswith("-mixed"):
            return "32-true"
        return config_precision
    if config_precision.endswith("-mixed"):
        return "32-true"
    return config_precision


def resolve_trainer_devices(
    *,
    accelerator: str,
    gpus: int,
    xpu_index: int,
) -> tuple[str | None, int, SingleXPUStrategy | None]:
    """Return ``(accelerator, devices, strategy)`` for Lightning Trainer.

    When *strategy* is set, pass it to ``Trainer(strategy=...)`` and leave
    *accelerator* as ``None`` (Lightning uses the strategy's device).
    """
    kind = _effective_accelerator(accelerator, gpus)
    if kind == "cuda":
        return "gpu", max(1, gpus), None
    if kind == "xpu":
        return None, 1, SingleXPUStrategy(device_index=xpu_index)
    return "cpu", 1, None


def format_training_device_message(
    *,
    effective: str,
    xpu_index: int = 0,
    cuda_devices: int = 1,
) -> str:
    """Human-readable device line printed before Lightning Trainer starts."""
    if effective == "xpu":
        suffix = ""
        if xpu_is_available():
            try:
                suffix = f" ({torch.xpu.get_device_name(xpu_index)})"
            except Exception:
                pass
        return (
            f"Using Intel XPU: xpu:{xpu_index}{suffix} "
            "(Lightning's 'GPU available' line refers to CUDA only)"
        )
    if effective == "cuda":
        suffix = ""
        if torch.cuda.is_available():
            try:
                suffix = f" ({torch.cuda.get_device_name(0)})"
            except Exception:
                pass
        return f"Using CUDA: {cuda_devices} device(s){suffix}"
    return "Using CPU"


def build_training_module(
    config: dict[str, Any],
    *,
    tokenizer: GABCTokenizer,
    encoder_pretrained: bool | None = None,
) -> ChantOMRLightningModule:
    """Build model + Lightning module from a loaded config dict."""
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)
    model = build_model(chant_config, encoder_pretrained=encoder_pretrained)
    return ChantOMRLightningModule(
        model,
        pad_token_id=tokenizer.pad_id,
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.05)),
        warmup_fraction=float(training_cfg.get("warmup_fraction", 0.05)),
        gradient_clip=float(training_cfg.get("gradient_clip", 1.0)),
        scheduled_sampling_prob=float(training_cfg.get("scheduled_sampling_prob", 0.0)),
    )


def build_training_dataloaders(
    config: dict[str, Any],
    *,
    tokenizer: GABCTokenizer,
    overfit_n: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders from config."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    rendered_dir = Path(data_cfg.get("rendered_dir", "data/rendered/"))

    train_ds, val_ds = build_datasets(
        rendered_dir,
        tokenizer,
        train_fraction=float(data_cfg.get("train_split", 0.9)),
        augment=bool(data_cfg.get("augment", False)),
        target_width=int(data_cfg.get("target_width", 1050)),
        max_height=int(data_cfg.get("max_height", 1600)),
        overfit_n=overfit_n,
    )
    return build_dataloaders(
        train_ds,
        val_ds,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
        pad_token_id=tokenizer.pad_id,
    )


def load_tokenizer_from_config(config: dict[str, Any]) -> GABCTokenizer:
    tokenizer_dir = Path(config.get("data", {}).get("tokenizer_dir", "data/tokenizer/"))
    return GABCTokenizer.load(tokenizer_dir / TOKENIZER_FILENAME)


def run_training(
    config_path: Path,
    *,
    resume: Path | None = None,
    finetune: bool = False,
    gpus: int = 1,
    accelerator: str = "auto",
    xpu_index: int = 0,
    precision: str | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    overfit_n: int | None = None,
    encoder_pretrained: bool | None = None,
) -> None:
    """Run a full Lightning training job."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    cfg = load_config(config_path)
    training_cfg = cfg.setdefault("training", {})
    model_cfg = cfg.setdefault("model", {})

    if batch_size is not None:
        training_cfg["batch_size"] = batch_size
    if epochs is not None:
        training_cfg["epochs"] = epochs
    if encoder_pretrained is not None:
        model_cfg["encoder_pretrained"] = encoder_pretrained

    tokenizer = load_tokenizer_from_config(cfg)
    train_loader, val_loader = build_training_dataloaders(
        cfg,
        tokenizer=tokenizer,
        overfit_n=overfit_n,
    )
    module = build_training_module(
        cfg,
        tokenizer=tokenizer,
        encoder_pretrained=encoder_pretrained,
    )

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints/"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="chant-omr-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=int(training_cfg.get("save_top_k", 3)),
        mode="min",
    )

    effective = _effective_accelerator(accelerator, gpus)
    if effective == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if effective == "xpu" and not xpu_is_available():
        raise RuntimeError(
            "XPU requested but torch.xpu.is_available() is False. "
            "Install PyTorch XPU wheels and Intel GPU drivers — see README.md."
        )

    resolved_precision = resolve_precision(
        precision or training_cfg.get("precision", "32-true"),
        accelerator=effective,
    )
    trainer_accelerator, devices, strategy = resolve_trainer_devices(
        accelerator=accelerator,
        gpus=gpus,
        xpu_index=xpu_index,
    )

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": int(training_cfg.get("epochs", 50)),
        "devices": devices,
        "precision": resolved_precision,
        "gradient_clip_val": float(training_cfg.get("gradient_clip", 1.0)),
        "callbacks": [checkpoint_cb],
        "enable_progress_bar": True,
        "log_every_n_steps": 1,
    }
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    else:
        trainer_kwargs["accelerator"] = trainer_accelerator

    device_msg = format_training_device_message(
        effective=effective,
        xpu_index=xpu_index,
        cuda_devices=devices,
    )
    print(device_msg)

    trainer = Trainer(**trainer_kwargs)
    if finetune and resume is None:
        raise ValueError("--finetune requires --resume CHECKPOINT")
    ckpt_path: str | None = str(resume) if resume else None
    if finetune and resume is not None:
        load_model_weights_into_module(module.model, resume)
        ckpt_path = None
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
