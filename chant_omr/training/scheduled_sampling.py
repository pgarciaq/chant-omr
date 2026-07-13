"""Scheduled sampling helpers for decoder training."""

from __future__ import annotations

import torch


def build_scheduled_decoder_input(
    gold_decoder_input: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    sampling_prob: float,
) -> torch.Tensor:
    """Mix gold decoder inputs with argmax predictions from a teacher-forcing pass.

    Position 0 (typically ``<bos>``) is always kept. For each later valid position
    ``t``, replace the gold token with the model's prediction at ``t - 1`` with
    probability *sampling_prob*.
    """
    if sampling_prob <= 0:
        return gold_decoder_input

    batch, seq_len = gold_decoder_input.shape
    if seq_len <= 1:
        return gold_decoder_input

    mixed = gold_decoder_input.clone()
    preds = teacher_logits.argmax(dim=-1)
    rand = torch.rand(batch, seq_len - 1, device=gold_decoder_input.device)
    replace = rand < sampling_prob
    valid = attention_mask[:, 1:].bool()
    replace = replace & valid
    mixed[:, 1:] = torch.where(replace, preds[:, :-1], mixed[:, 1:])
    return mixed
