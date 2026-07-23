"""Load trained ChantOMR weights from Lightning checkpoints and safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from chant_omr.model.chant_omr_model import ChantOMR, ChantOMRConfig, build_model
from chant_omr.model.tokenizer import TOKENIZER_FILENAME, GABCTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _strip_lightning_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only ``model.*`` weights and remove the prefix."""
    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value
    if not model_state:
        raise ValueError("checkpoint state_dict contains no model.* keys")
    return model_state


def load_model_weights_into_module(module: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load only ``model.*`` weights from a Lightning checkpoint into *module*."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"not a Lightning checkpoint: {checkpoint_path}")
    module.load_state_dict(_strip_lightning_prefix(ckpt["state_dict"]), strict=True)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
    tokenizer_dir: Path | None = None,
    device: torch.device | str | None = None,
) -> tuple[ChantOMR, GABCTokenizer, dict[str, Any]]:
    """Rebuild ``ChantOMR`` + tokenizer from a Lightning ``.ckpt`` file."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"not a Lightning checkpoint: {checkpoint_path}")

    cfg_path = Path(config_path or "configs/default.yaml")
    cfg = load_config(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = dict(cfg.get("model", {}))

    hparams = ckpt.get("hyper_parameters") or {}
    if isinstance(hparams, dict):
        model_cfg.setdefault("encoder_pretrained", False)

    chant_config = ChantOMRConfig.from_mapping(model_cfg)
    model = build_model(chant_config, encoder_pretrained=False)
    model.load_state_dict(_strip_lightning_prefix(ckpt["state_dict"]), strict=True)

    tok_dir = Path(tokenizer_dir or data_cfg.get("tokenizer_dir", "data/tokenizer/"))
    tokenizer = GABCTokenizer.load(tok_dir / TOKENIZER_FILENAME)

    if device is not None:
        model = model.to(device)
    model.eval()

    meta = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config_path": str(cfg_path.resolve()),
        "tokenizer_dir": str(tok_dir.resolve()),
    }
    return model, tokenizer, meta


def load_model_from_safetensors(
    model_dir: Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[ChantOMR, GABCTokenizer, dict[str, Any]]:
    """Rebuild ``ChantOMR`` + tokenizer from a HuggingFace Hub download.

    Expects *model_dir* to contain ``model.safetensors``, ``config.yaml``,
    and ``tokenizer.json`` (the layout produced by :func:`chant_omr.hub.upload_to_hub`).
    """
    from safetensors.torch import load_file

    model_dir = Path(model_dir)

    cfg_path = model_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")
    cfg = load_config(cfg_path)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("encoder_pretrained", False)

    chant_config = ChantOMRConfig.from_mapping(model_cfg)
    model = build_model(chant_config, encoder_pretrained=False)

    st_path = model_dir / "model.safetensors"
    if not st_path.is_file():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    state_dict = load_file(str(st_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)

    tok_path = model_dir / TOKENIZER_FILENAME
    if not tok_path.is_file():
        raise FileNotFoundError(f"{TOKENIZER_FILENAME} not found in {model_dir}")
    tokenizer = GABCTokenizer.load(tok_path)

    if device is not None:
        model = model.to(device)
    model.eval()

    meta = {
        "model_dir": str(model_dir.resolve()),
        "config_path": str(cfg_path.resolve()),
    }
    return model, tokenizer, meta
