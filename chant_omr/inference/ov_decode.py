"""OpenVINO Runtime decode loop — no PyTorch at inference (#41).

Loads encoder + decoder IR files and runs the full beam search using
only OpenVINO Runtime.  This is the inference path for ghh (#15).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chant_omr.inference.beam_search import (
    DecodeConfig,
    LogitsFunc,
    beam_search_decode_generic,
    greedy_decode_generic,
)


def load_openvino_models(
    model_dir: Path,
    *,
    device: str = "CPU",
) -> tuple:
    """Load compiled encoder and decoder from an export directory.

    Returns ``(encoder_compiled, decoder_compiled, manifest)``.
    """
    import openvino as ov

    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    core = ov.Core()
    encoder_xml = model_dir / "encoder.xml"
    decoder_xml = model_dir / "decoder.xml"
    if not encoder_xml.is_file():
        raise FileNotFoundError(f"encoder.xml not found in {model_dir}")
    if not decoder_xml.is_file():
        raise FileNotFoundError(f"decoder.xml not found in {model_dir}")

    encoder_compiled = core.compile_model(str(encoder_xml), device)
    decoder_compiled = core.compile_model(str(decoder_xml), device)

    return encoder_compiled, decoder_compiled, manifest_data


def ov_encoder_infer(
    encoder_compiled,
    pixel_values: np.ndarray,
) -> np.ndarray:
    """Run the encoder IR and return ``encoder_memory (1, N, d_model)``."""
    result = encoder_compiled({"pixel_values": pixel_values})
    return np.array(result[0])


def ov_decoder_logits_func(decoder_compiled) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by the OpenVINO decoder IR.

    The returned callable converts numpy ↔ torch at the boundary so the
    generic decode loop works unchanged.
    """

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        result = decoder_compiled({
            "input_ids": input_ids.cpu().numpy(),
            "encoder_memory": memory.cpu().numpy(),
        })
        next_logits = torch.from_numpy(np.array(result[0]))
        return F.log_softmax(next_logits[0, 0], dim=-1)

    return _step


def ov_decode_token_ids(
    encoder_compiled,
    decoder_compiled,
    pixel_values: np.ndarray,
    *,
    bos_token_id: int,
    eos_token_id: int,
    config: DecodeConfig,
) -> list[int]:
    """Full OpenVINO pipeline: encode image → decode tokens (no PyTorch model).

    Args:
        encoder_compiled: Compiled OpenVINO encoder model.
        decoder_compiled: Compiled OpenVINO decoder model.
        pixel_values: Preprocessed image as ``(1, 3, H, W)`` float32 numpy array.
        bos_token_id: Beginning-of-sequence token ID.
        eos_token_id: End-of-sequence token ID.
        config: Decode settings (beam width, max length, repetition penalty).

    Returns:
        List of token IDs including BOS and (if generated) EOS.
    """
    encoder_memory = ov_encoder_infer(encoder_compiled, pixel_values)
    memory_tensor = torch.from_numpy(encoder_memory)
    logits_fn = ov_decoder_logits_func(decoder_compiled)

    if config.beam_width <= 1:
        return greedy_decode_generic(
            logits_fn,
            memory_tensor,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=config.max_length,
            repetition_penalty=config.repetition_penalty,
        )
    return beam_search_decode_generic(
        logits_fn,
        memory_tensor,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_length=config.max_length,
        beam_width=config.beam_width,
        repetition_penalty=config.repetition_penalty,
    )
