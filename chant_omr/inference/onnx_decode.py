"""ONNX Runtime decode loop with KV-cached decoding (#51).

Loads encoder + decoder_init + decoder_step ONNX files and runs
greedy decoding with auto execution-provider selection.  This is the
inference path for ``chant-omr predict --device onnx``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from chant_omr.inference.beam_search import (
    GrammarMask,
    LogitsFunc,
    build_paren_table,
    greedy_decode_generic,
)
from chant_omr.inference.gabc_output import assemble_gabc
from chant_omr.inference.preprocess import prepare_inference_numpy
from chant_omr.model.tokenizer import GABCTokenizer


def load_onnx_models(
    model_dir: Path,
    *,
    providers: list[str] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], GABCTokenizer]:
    """Load three ONNX sessions, manifest, and tokenizer from *model_dir*.

    Args:
        model_dir: Directory produced by ``chant-omr export --format onnx``.
        providers: ONNX Runtime execution providers.  When ``None``, uses
            ``ort.get_available_providers()`` for automatic selection.

    Returns:
        ``(encoder_session, init_session, step_session, manifest, tokenizer)``
    """
    import onnxruntime as ort

    model_dir = Path(model_dir)
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if providers is None:
        providers = ort.get_available_providers()

    encoder_path = model_dir / "encoder.onnx"
    init_path = model_dir / "decoder_init.onnx"
    step_path = model_dir / "decoder_step.onnx"

    for p in (encoder_path, init_path, step_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p.name} not found in {model_dir}")

    encoder_session = ort.InferenceSession(str(encoder_path), providers=providers)
    init_session = ort.InferenceSession(str(init_path), providers=providers)
    step_session = ort.InferenceSession(str(step_path), providers=providers)

    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    tokenizer = GABCTokenizer.load(tok_path)

    return encoder_session, init_session, step_session, manifest, tokenizer


def onnx_encoder_infer(
    encoder_session: Any,
    pixel_values_np: np.ndarray,
) -> np.ndarray:
    """Run the ONNX encoder and return ``encoder_memory (1, N, d_model)``."""
    result = encoder_session.run(None, {"pixel_values": pixel_values_np})
    return result[0]


def onnx_logits_func_cached(
    init_session: Any,
    step_session: Any,
    encoder_mask_np: np.ndarray,
) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by ONNX Runtime with KV cache.

    Uses ``decoder_init.onnx`` for the first step (computes cross-attention
    K/V from encoder memory) and ``decoder_step.onnx`` for subsequent steps
    (reuses cached cross-attention K/V).  The numpy KV cache is managed as
    closed-over mutable state.
    """
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = [None]

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        ids_np = input_ids[:, -1:].cpu().numpy()

        if cache[0] is None:
            result = init_session.run(None, {
                "input_ids": ids_np,
                "encoder_memory": memory.cpu().numpy(),
                "encoder_mask": encoder_mask_np,
            })
            logits_np, self_k, self_v, cross_k, cross_v = result
            cache[0] = (self_k, self_v, cross_k, cross_v)
        else:
            self_k, self_v, cross_k, cross_v = cache[0]
            result = step_session.run(None, {
                "input_ids": ids_np,
                "past_self_k": self_k,
                "past_self_v": self_v,
                "past_cross_k": cross_k,
                "past_cross_v": cross_v,
                "encoder_mask": encoder_mask_np,
            })
            logits_np, self_k, self_v, cross_k, cross_v = result
            cache[0] = (self_k, self_v, cross_k, cross_v)

        return F.log_softmax(torch.from_numpy(logits_np[0, 0].copy()), dim=-1)

    return _step


def onnx_predict_gabc(
    image_path: Path,
    model_dir: Path,
    *,
    providers: list[str] | None = None,
    max_length: int = 2048,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    grammar_penalty: float = float("-inf"),
    name: str | None = None,
) -> str:
    """Run ONNX Runtime OMR inference on a single image and return GABC.

    This is the top-level entry point for ``--device onnx``.  It loads the
    ONNX models, preprocesses the image as a numpy array, runs KV-cached
    greedy decoding, and assembles the final GABC output.

    Beam search is not supported with ONNX — use the PyTorch backend for
    beam search (deferred to a follow-up issue).
    """
    enc_sess, init_sess, step_sess, manifest, tokenizer = load_onnx_models(
        model_dir, providers=providers,
    )

    pixel_values_np = prepare_inference_numpy(Path(image_path))

    encoder_memory = onnx_encoder_infer(enc_sess, pixel_values_np)
    memory_tensor = torch.from_numpy(encoder_memory)

    encoder_mask_np = np.ones((1, encoder_memory.shape[1]), dtype=np.float32)
    logits_fn = onnx_logits_func_cached(init_sess, step_sess, encoder_mask_np)

    gm: GrammarMask | None = None
    if grammar_constrained:
        paren_table = build_paren_table(tokenizer)
        gm = GrammarMask(
            paren_table, tokenizer.eos_id, tokenizer.vocab_size,
            penalty=grammar_penalty,
        )

    token_ids = greedy_decode_generic(
        logits_fn,
        memory_tensor,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_mask=gm,
    )
    body = tokenizer.decode(token_ids, skip_special_tokens=True)
    return assemble_gabc(body, name=name or "OMR output")
