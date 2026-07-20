"""Run OMR inference on score images (13a — PyTorch only)."""

from __future__ import annotations

from pathlib import Path

import torch

from chant_omr.inference.beam_search import DecodeConfig, decode_token_ids
from chant_omr.inference.checkpoint import load_model_from_checkpoint
from chant_omr.inference.gabc_output import assemble_gabc
from chant_omr.inference.metrics import (
    compute_predict_metrics,
    format_predict_metrics,
    resolve_reference_gabc,
)
from chant_omr.inference.preprocess import prepare_inference_tensor
from chant_omr.training.lightning_module import format_training_device_message
from chant_omr.training.xpu_strategy import xpu_is_available


def resolve_inference_device(device: str, *, xpu_index: int = 0) -> torch.device:
    """Map CLI device string to ``torch.device``."""
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device == "xpu":
        if not xpu_is_available():
            raise RuntimeError("XPU requested but torch.xpu.is_available() is False")
        return torch.device("xpu", xpu_index)
    if device != "auto":
        raise ValueError(f"unsupported device: {device!r} (use auto, cuda, xpu, cpu)")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if xpu_is_available():
        return torch.device("xpu", xpu_index)
    return torch.device("cpu")


def _effective_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "xpu":
        return "xpu"
    return "cpu"


def predict_gabc(
    image_path: Path,
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
    device: str = "auto",
    xpu_index: int = 0,
    beam_width: int = 3,
    max_length: int | None = None,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    grammar_penalty: float = float("-inf"),
    name: str | None = None,
    dump_metrics: bool = False,
    reference_gabc_path: Path | None = None,
) -> str:
    """Run OMR on a single image and return a full GABC file string."""
    torch_device = resolve_inference_device(device, xpu_index=xpu_index)
    effective = _effective_device_name(torch_device)
    print(
        format_training_device_message(
            effective=effective,
            xpu_index=xpu_index,
            cuda_devices=1,
        )
    )

    model, tokenizer, _meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device=torch_device,
    )
    resolved_max_length = max_length if max_length is not None else model.config.max_seq_len
    pixel_values = prepare_inference_tensor(image_path, device=torch_device)
    decode_config = DecodeConfig(
        beam_width=beam_width,
        max_length=resolved_max_length,
        repetition_penalty=repetition_penalty,
        grammar_constrained=grammar_constrained,
        grammar_penalty=grammar_penalty,
    )
    ref_gabc = reference_gabc_path
    if dump_metrics and ref_gabc is None:
        ref_gabc = resolve_reference_gabc(image_path)
    if dump_metrics:
        _, metrics = compute_predict_metrics(
            model,
            pixel_values,
            tokenizer,
            decode_config=decode_config,
            reference_gabc_path=ref_gabc,
        )
        print(format_predict_metrics(metrics))
    token_ids = decode_token_ids(
        model,
        pixel_values,
        tokenizer,
        decode_config,
    )
    body = tokenizer.decode(token_ids, skip_special_tokens=True)
    return assemble_gabc(body, name=name or "OMR output")
