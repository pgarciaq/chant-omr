"""Load trained ChantOMR weights from Lightning checkpoints."""

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
