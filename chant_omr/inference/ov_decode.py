"""OpenVINO Runtime decode loop with dual-path decoding (#36).

Loads encoder + decoder (non-cached) + decoder_init/step (KV-cached) IR
files and runs inference.  Greedy uses the O(n) cached path; beam search
falls back to the O(n^2) non-cached decoder for correctness.

This is the inference path for ``chant-omr predict --device openvino``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from chant_omr.inference.beam_search import (
    GrammarMask,
    LogitsFunc,
    beam_search_decode_generic,
    build_paren_table,
    greedy_decode_generic,
)
from chant_omr.inference.gabc_output import assemble_gabc
from chant_omr.inference.preprocess import prepare_inference_numpy
from chant_omr.model.tokenizer import GABCTokenizer


@dataclass
class OvModelBundle:
    """Pre-loaded OpenVINO models, manifest, and tokenizer."""

    encoder: Any
    decoder: Any
    decoder_init: Any
    decoder_step: Any
    manifest: dict[str, Any]
    tokenizer: GABCTokenizer


def load_openvino_models(
    model_dir: Path,
    *,
    device: str = "AUTO",
) -> OvModelBundle:
    """Load compiled OpenVINO IRs, manifest, and tokenizer from *model_dir*.

    Returns an :class:`OvModelBundle` with ``encoder``, ``decoder``,
    ``decoder_init``, ``decoder_step``, ``manifest``, and ``tokenizer``.

    The *decoder* is the non-cached IR for beam search; *decoder_init* and
    *decoder_step* are the KV-cached IRs for greedy decoding.
    """
    import openvino as ov

    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    core = ov.Core()

    encoder_xml = model_dir / "encoder.xml"
    decoder_xml = model_dir / "decoder.xml"
    init_xml = model_dir / "decoder_init.xml"
    step_xml = model_dir / "decoder_step.xml"

    for p in (encoder_xml, decoder_xml, init_xml, step_xml):
        if not p.is_file():
            raise FileNotFoundError(f"{p.name} not found in {model_dir}")

    encoder_compiled = core.compile_model(str(encoder_xml), device)
    decoder_compiled = core.compile_model(str(decoder_xml), device)
    init_compiled = core.compile_model(str(init_xml), device)
    step_compiled = core.compile_model(str(step_xml), device)

    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    tokenizer = GABCTokenizer.load(tok_path)

    return OvModelBundle(
        encoder=encoder_compiled,
        decoder=decoder_compiled,
        decoder_init=init_compiled,
        decoder_step=step_compiled,
        manifest=manifest,
        tokenizer=tokenizer,
    )


def ov_encoder_infer(
    encoder_compiled: Any,
    pixel_values: np.ndarray,
) -> np.ndarray:
    """Run the encoder IR and return ``encoder_memory (1, N, d_model)``."""
    result = encoder_compiled({"pixel_values": pixel_values})
    return np.array(result[0])


def ov_decoder_logits_func(
    decoder_compiled: Any,
    encoder_mask: np.ndarray,
) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by the non-cached OpenVINO decoder IR.

    Used for beam search where the full ``input_ids`` sequence is passed
    each step (O(n^2) but correct for variable-prefix beam candidates).
    """

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        result = decoder_compiled({
            "input_ids": input_ids.cpu().numpy(),
            "encoder_memory": memory.cpu().numpy(),
            "encoder_mask": encoder_mask,
        })
        next_logits = torch.from_numpy(np.array(result[0]))
        return F.log_softmax(next_logits[0, 0], dim=-1)

    return _step


def ov_logits_func_cached(
    init_compiled: Any,
    step_compiled: Any,
    encoder_mask_np: np.ndarray,
) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by KV-cached OpenVINO decoder IRs.

    Uses ``decoder_init`` for the first step (computes cross-attention
    K/V from encoder memory) and ``decoder_step`` for subsequent steps
    (reuses cached cross-attention K/V).  O(n) total computation.
    """
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = [None]

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        ids_np = input_ids[:, -1:].cpu().numpy()

        if cache[0] is None:
            result = init_compiled({
                "input_ids": ids_np,
                "encoder_memory": memory.cpu().numpy(),
                "encoder_mask": encoder_mask_np,
            })
            logits_np = np.array(result[0])
            self_k = np.array(result[1])
            self_v = np.array(result[2])
            cross_k = np.array(result[3])
            cross_v = np.array(result[4])
            cache[0] = (self_k, self_v, cross_k, cross_v)
        else:
            self_k, self_v, cross_k, cross_v = cache[0]
            result = step_compiled({
                "input_ids": ids_np,
                "past_self_k": self_k,
                "past_self_v": self_v,
                "past_cross_k": cross_k,
                "past_cross_v": cross_v,
                "encoder_mask": encoder_mask_np,
            })
            logits_np = np.array(result[0])
            self_k = np.array(result[1])
            self_v = np.array(result[2])
            cross_k = np.array(result[3])
            cross_v = np.array(result[4])
            cache[0] = (self_k, self_v, cross_k, cross_v)

        return F.log_softmax(torch.from_numpy(logits_np[0, 0].copy()), dim=-1)

    return _step


def _decode_from_pixels(
    pixel_values_np: np.ndarray,
    models: OvModelBundle,
    *,
    beam_width: int = 1,
    max_length: int | None = None,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    grammar_penalty: float = float("-inf"),
    name: str | None = None,
) -> str:
    """Shared decode logic for both file-path and array entry points."""
    resolved_max_length = (
        max_length
        if max_length is not None
        else models.manifest.get("config", {}).get("max_seq_len", 8192)
    )

    encoder_memory = ov_encoder_infer(models.encoder, pixel_values_np)
    memory_tensor = torch.from_numpy(encoder_memory)
    encoder_mask_np = np.ones((1, encoder_memory.shape[1]), dtype=np.float32)

    tokenizer = models.tokenizer
    gm: GrammarMask | None = None
    if grammar_constrained:
        paren_table = build_paren_table(tokenizer)
        gm = GrammarMask(
            paren_table, tokenizer.eos_id, tokenizer.vocab_size,
            penalty=grammar_penalty,
        )

    if beam_width <= 1:
        logits_fn = ov_logits_func_cached(
            models.decoder_init, models.decoder_step, encoder_mask_np,
        )
        token_ids = greedy_decode_generic(
            logits_fn,
            memory_tensor,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=resolved_max_length,
            repetition_penalty=repetition_penalty,
            grammar_mask=gm,
        )
    else:
        logits_fn = ov_decoder_logits_func(models.decoder, encoder_mask_np)
        token_ids = beam_search_decode_generic(
            logits_fn,
            memory_tensor,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=resolved_max_length,
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            grammar_mask=gm,
        )

    body = tokenizer.decode(token_ids, skip_special_tokens=True)
    return assemble_gabc(body, name=name or "OMR output")


def ov_predict_gabc(
    image_path: Path,
    model_dir: Path,
    *,
    ov_device: str = "AUTO",
    beam_width: int = 1,
    max_length: int | None = None,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    grammar_penalty: float = float("-inf"),
    name: str | None = None,
) -> str:
    """Run OpenVINO OMR inference on a single image and return GABC.

    Dual-path decoding:
        - ``beam_width <= 1``: KV-cached greedy (O(n), fast)
        - ``beam_width > 1``: non-cached beam search (O(n^2), correct)

    Grammar-constrained decoding is supported on both paths.
    """
    models = load_openvino_models(model_dir, device=ov_device)
    pixel_values_np = prepare_inference_numpy(Path(image_path))
    return _decode_from_pixels(
        pixel_values_np,
        models,
        beam_width=beam_width,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_constrained=grammar_constrained,
        grammar_penalty=grammar_penalty,
        name=name,
    )


def ov_predict_gabc_from_array(
    img_array: np.ndarray,
    models: OvModelBundle,
    *,
    beam_width: int = 1,
    max_length: int | None = None,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    grammar_penalty: float = float("-inf"),
    name: str | None = None,
) -> str:
    """Run OpenVINO OMR inference on an in-memory image array and return GABC.

    Accepts a pre-loaded :class:`OvModelBundle` so the caller can reuse
    compiled models across many images (avoiding repeated compilation).

    *img_array* must be a ``(1, 3, H, W)`` float32 numpy array as returned
    by :func:`~chant_omr.inference.preprocess.prepare_inference_numpy_from_array`.
    """
    return _decode_from_pixels(
        img_array,
        models,
        beam_width=beam_width,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_constrained=grammar_constrained,
        grammar_penalty=grammar_penalty,
        name=name,
    )
