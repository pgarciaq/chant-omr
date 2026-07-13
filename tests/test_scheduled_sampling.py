"""Tests for scheduled sampling decoder inputs."""

from __future__ import annotations

import torch

from chant_omr.training.scheduled_sampling import build_scheduled_decoder_input


def test_build_scheduled_decoder_input_keeps_bos():
    gold = torch.tensor([[1, 10, 11, 12]])
    logits = torch.zeros(1, 4, 5)
    logits[0, 0, 2] = 100.0
    mask = torch.ones(1, 4, dtype=torch.long)
    mixed = build_scheduled_decoder_input(
        gold,
        logits,
        attention_mask=mask,
        sampling_prob=1.0,
    )
    assert mixed[0, 0].item() == 1
    assert mixed[0, 1].item() == 2


def test_build_scheduled_decoder_input_zero_prob_is_unchanged():
    gold = torch.tensor([[1, 10, 11]])
    logits = torch.zeros(1, 3, 5)
    logits[0, :, 2] = 100.0
    mask = torch.ones(1, 3, dtype=torch.long)
    mixed = build_scheduled_decoder_input(
        gold,
        logits,
        attention_mask=mask,
        sampling_prob=0.0,
    )
    assert torch.equal(mixed, gold)


def test_build_scheduled_decoder_input_respects_padding_mask():
    gold = torch.tensor([[1, 10, 0]])
    logits = torch.zeros(1, 3, 5)
    logits[0, :, 2] = 100.0
    mask = torch.tensor([[1, 1, 0]])
    mixed = build_scheduled_decoder_input(
        gold,
        logits,
        attention_mask=mask,
        sampling_prob=1.0,
    )
    assert mixed[0, 1].item() == 2
    assert mixed[0, 2].item() == 0
