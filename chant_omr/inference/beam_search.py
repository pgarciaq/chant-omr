"""Autoregressive decoding with greedy and beam search.

Supports both PyTorch and OpenVINO backends via the ``LogitsFunc`` callable
protocol: any function ``(token_ids, encoder_memory) → log_probs (vocab,)``
can drive the decode loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chant_omr.model.chant_omr_model import ChantOMR
from chant_omr.model.tokenizer import GABCTokenizer

LogitsFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(input_ids (1, T), encoder_memory (1, N, D)) → log_probs (vocab,)``"""


@dataclass(frozen=True)
class DecodeConfig:
    """Token generation settings."""

    beam_width: int = 1
    max_length: int = 2048
    repetition_penalty: float = 1.0


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


def pytorch_logits_func(model: ChantOMR) -> LogitsFunc:
    """Return a ``LogitsFunc`` backed by the PyTorch decoder."""

    def _step(input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        logits = model.decoder(
            input_ids,
            memory,
            attention_mask=attention_mask,
            encoder_attention_mask=None,
        )
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
) -> list[int]:
    """Greedy left-to-right decode starting from BOS."""
    return greedy_decode_generic(
        pytorch_logits_func(model),
        memory,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
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
) -> list[int]:
    """Beam search decode for batch size 1 encoder memory."""
    return beam_search_decode_generic(
        pytorch_logits_func(model),
        memory,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        max_length=max_length,
        beam_width=beam_width,
        repetition_penalty=repetition_penalty,
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
) -> list[int]:
    """Greedy decode using any ``LogitsFunc`` backend."""
    device = memory.device
    tokens = [bos_token_id]
    for _ in range(max_length - 1):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        log_probs = logits_fn(input_ids, memory)
        logits = log_probs.clone()
        logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
        next_id = int(torch.argmax(logits).item())
        tokens.append(next_id)
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
) -> list[int]:
    """Beam search decode using any ``LogitsFunc`` backend."""
    device = memory.device
    beams: list[tuple[list[int], float]] = [([bos_token_id], 0.0)]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_length - 1):
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            if tokens[-1] == eos_token_id:
                finished.append((tokens, score))
                continue
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            log_probs = logits_fn(input_ids, memory)
            logits = log_probs.clone()
            logits = apply_repetition_penalty(logits, tokens, repetition_penalty)
            topk = torch.topk(logits, k=min(beam_width, logits.numel()))
            for log_prob, token_id in zip(
                topk.values.tolist(), topk.indices.tolist(), strict=True
            ):
                candidates.append((tokens + [token_id], score + log_prob))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == eos_token_id for seq, _ in beams):
            finished.extend(beams)
            break

    if finished:
        finished.sort(key=lambda item: item[1], reverse=True)
        return finished[0][0]

    beams.sort(key=lambda item: item[1], reverse=True)
    return beams[0][0]


def decode_token_ids(
    model: ChantOMR,
    pixel_values: torch.Tensor,
    tokenizer: GABCTokenizer,
    config: DecodeConfig,
) -> list[int]:
    """Encode image once, then greedy or beam decode to token IDs."""
    if pixel_values.shape[0] != 1:
        raise ValueError("decode_token_ids expects batch size 1")
    with torch.inference_mode():
        memory = model.encode(pixel_values)
        if config.beam_width <= 1:
            return greedy_decode(
                model,
                memory,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                max_length=config.max_length,
                repetition_penalty=config.repetition_penalty,
            )
        return beam_search_decode(
            model,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=config.max_length,
            beam_width=config.beam_width,
            repetition_penalty=config.repetition_penalty,
        )
