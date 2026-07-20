"""Autoregressive decoding with greedy and beam search.

Supports both PyTorch and OpenVINO backends via the ``LogitsFunc`` callable
protocol: any function ``(token_ids, encoder_memory) → log_probs (vocab,)``
can drive the decode loop.

KV cache (#44): ``pytorch_logits_func_cached`` returns a stateful
``LogitsFunc`` that uses the decoder's KV cache for O(n) greedy generation.

Grammar-constrained decoding (#37): when ``grammar_constrained=True``, a
parenthesis-balancing mask is applied at each step to prevent structurally
invalid GABC output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chant_omr.inference.grammar import GrammarMask, build_paren_table
from chant_omr.model.chant_omr_model import ChantOMR
from chant_omr.model.decoder import KVCache
from chant_omr.model.tokenizer import GABCTokenizer

LogitsFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(input_ids (1, T), encoder_memory (1, N, D)) → log_probs (vocab,)``"""


@dataclass(frozen=True)
class DecodeConfig:
    """Token generation settings."""

    beam_width: int = 1
    max_length: int = 8192
    repetition_penalty: float = 1.0
    grammar_constrained: bool = False
    grammar_penalty: float = float("-inf")


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Down-weight logits for tokens already present in *token_ids*."""
    if penalty == 1.0 or not token_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(token_ids):
        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty
    return adjusted


def pytorch_logits_func(
    model: ChantOMR,
    encoder_attention_mask: torch.Tensor | None = None,
) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by the PyTorch decoder (no KV cache).

    Used by beam search where per-beam cache management is complex.
    The *encoder_attention_mask* is closed over.
    """

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        logits, _ = model.decoder(
            input_ids,
            memory,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return F.log_softmax(logits[0, -1], dim=-1)

    return _step


def pytorch_logits_func_cached(
    model: ChantOMR,
    encoder_attention_mask: torch.Tensor | None = None,
) -> LogitsFunc:
    """Return a stateful ``LogitsFunc`` with KV cache for O(n) greedy decode.

    On the first call (``input_ids`` has >1 token), runs the full prefix and
    populates the cache.  On subsequent calls (single new token), only the
    new token is processed.  The *encoder_attention_mask* and the cache are
    closed over.
    """
    cache_state: list[KVCache | None] = [None]

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        cached = cache_state[0]
        if cached is not None:
            new_ids = input_ids[:, -1:]
        else:
            new_ids = input_ids

        logits, new_cache = model.decoder(
            new_ids,
            memory,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cached,
            use_cache=True,
        )
        cache_state[0] = new_cache
        return F.log_softmax(logits[0, -1], dim=-1)

    return _step


def _decoder_step_logits(
    model: ChantOMR,
    memory: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return next-token log-probs ``(vocab,)`` for the final position.

    Kept for backward compatibility — delegates to ``pytorch_logits_func``.
    """
    return pytorch_logits_func(model)(input_ids, memory)


def greedy_decode(
    model: ChantOMR,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    repetition_penalty: float = 1.0,
    encoder_attention_mask: torch.Tensor | None = None,
    grammar_mask: GrammarMask | None = None,
) -> list[int]:
    """Greedy left-to-right decode starting from BOS (uses KV cache)."""
    return greedy_decode_generic(
        pytorch_logits_func_cached(model, encoder_attention_mask),
        memory,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_mask=grammar_mask,
    )


def beam_search_decode(
    model: ChantOMR,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    beam_width: int,
    repetition_penalty: float = 1.0,
    encoder_attention_mask: torch.Tensor | None = None,
    grammar_mask: GrammarMask | None = None,
) -> list[int]:
    """Beam search decode for batch size 1 encoder memory."""
    return beam_search_decode_generic(
        pytorch_logits_func(model, encoder_attention_mask),
        memory,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_length=max_length,
        beam_width=beam_width,
        repetition_penalty=repetition_penalty,
        grammar_mask=grammar_mask,
    )


# ---------------------------------------------------------------------------
# Generic decode loops — backend-agnostic via LogitsFunc
# ---------------------------------------------------------------------------


def greedy_decode_generic(
    logits_fn: LogitsFunc,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    repetition_penalty: float = 1.0,
    grammar_mask: GrammarMask | None = None,
) -> list[int]:
    """Greedy decode using any ``LogitsFunc`` backend."""
    device = memory.device
    tokens = [bos_token_id]
    if grammar_mask is not None:
        grammar_mask.update(bos_token_id)
    for _ in range(max_length - 1):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        log_probs = logits_fn(input_ids, memory)
        logits = log_probs.clone()
        logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
        if grammar_mask is not None:
            logits = logits + grammar_mask.get_mask(device=device)
        next_id = int(torch.argmax(logits).item())
        tokens.append(next_id)
        if grammar_mask is not None:
            grammar_mask.update(next_id)
        if next_id == eos_token_id:
            break
    return tokens


def beam_search_decode_generic(
    logits_fn: LogitsFunc,
    memory: torch.Tensor,
    *,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    beam_width: int,
    repetition_penalty: float = 1.0,
    grammar_mask: GrammarMask | None = None,
) -> list[int]:
    """Beam search decode using any ``LogitsFunc`` backend."""
    device = memory.device

    # Each beam carries its own grammar state so branching is independent
    init_gm = grammar_mask.clone() if grammar_mask is not None else None
    if init_gm is not None:
        init_gm.update(bos_token_id)

    beams: list[tuple[list[int], float, GrammarMask | None]] = [
        ([bos_token_id], 0.0, init_gm),
    ]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_length - 1):
        candidates: list[tuple[list[int], float, GrammarMask | None]] = []
        for tokens, score, gm in beams:
            if tokens[-1] == eos_token_id:
                finished.append((tokens, score))
                continue
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            log_probs = logits_fn(input_ids, memory)
            logits = log_probs.clone()
            logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
            if gm is not None:
                logits = logits + gm.get_mask(device=device)
            topk = torch.topk(logits, k=min(beam_width, logits.numel()))
            for log_prob, token_id in zip(
                topk.values.tolist(), topk.indices.tolist(), strict=True
            ):
                new_gm = gm.clone() if gm is not None else None
                if new_gm is not None:
                    new_gm.update(token_id)
                candidates.append((tokens + [token_id], score + log_prob, new_gm))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == eos_token_id for seq, _, _gm in beams):
            finished.extend((seq, sc) for seq, sc, _gm in beams)
            break

    if finished:
        finished.sort(key=lambda item: item[1], reverse=True)
        return finished[0][0]

    beams.sort(key=lambda item: item[1], reverse=True)
    return beams[0][0]


def build_encoder_attention_mask(
    pixel_values: torch.Tensor,
    patch_size: int = 32,
) -> torch.Tensor:
    """Build a ``(1, N)`` mask marking real (non-padded) encoder patches.

    The encoder divides ``(1, 3, H, W)`` into ``(H // patch_size)`` vertical
    and ``(W // patch_size)`` horizontal patches.  If the original image was
    shorter than the padded tensor's ``H``, trailing vertical rows of patches
    are all-padding.  This function marks those as 0 and real rows as 1.

    For now, we only mask along the height axis (width is always fully used
    after the fixed-width resize).
    """
    _, _, h, w = pixel_values.shape
    rows = h // patch_size
    cols = w // patch_size
    mask = torch.ones(1, rows * cols, device=pixel_values.device)
    return mask


def decode_token_ids(
    model: ChantOMR,
    pixel_values: torch.Tensor,
    tokenizer: GABCTokenizer,
    config: DecodeConfig,
) -> list[int]:
    """Encode image once, then greedy or beam decode to token IDs."""
    if pixel_values.shape[0] != 1:
        raise ValueError("decode_token_ids expects batch size 1")

    gm: GrammarMask | None = None
    if config.grammar_constrained:
        paren_table = build_paren_table(tokenizer)
        gm = GrammarMask(
            paren_table, tokenizer.eos_id, tokenizer.vocab_size,
            penalty=config.grammar_penalty,
        )

    with torch.inference_mode():
        memory = model.encode(pixel_values)
        enc_mask = build_encoder_attention_mask(pixel_values)
        if config.beam_width <= 1:
            return greedy_decode(
                model,
                memory,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                max_length=config.max_length,
                repetition_penalty=config.repetition_penalty,
                encoder_attention_mask=enc_mask,
                grammar_mask=gm,
            )
        return beam_search_decode(
            model,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=config.max_length,
            beam_width=config.beam_width,
            repetition_penalty=config.repetition_penalty,
            encoder_attention_mask=enc_mask,
            grammar_mask=gm,
        )
