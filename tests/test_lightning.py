"""Tests for Lightning training module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chant_omr.model.chant_omr_model import build_model
from chant_omr.model.tokenizer import train_tokenizer
from chant_omr.training.lightning_module import ChantOMRLightningModule

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"


@pytest.fixture
def tokenizer(tmp_path: Path):
    return train_tokenizer(
        GABC_FIXTURES,
        vocab_size=256,
        output_dir=tmp_path / "tokenizer",
        min_body_len=10,
        use_manifest=False,
    )


@pytest.fixture
def module(tokenizer) -> ChantOMRLightningModule:
    model = build_model(encoder_pretrained=False)
    return ChantOMRLightningModule(model, pad_token_id=tokenizer.pad_id)


class TestTrainingStep:
    def test_training_step(self, module: ChantOMRLightningModule):
        batch = {
            "pixel_values": torch.randn(2, 3, 512, 1050),
            "input_ids": torch.tensor([[1, 10, 11, 2, 0], [1, 20, 21, 22, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
            "encoder_attention_mask": torch.ones(2, 16 * 32, dtype=torch.long),
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_validation_step(self, module: ChantOMRLightningModule):
        batch = {
            "pixel_values": torch.randn(1, 3, 400, 1050),
            "input_ids": torch.tensor([[1, 5, 6, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "encoder_attention_mask": torch.ones(1, 12 * 32, dtype=torch.long),
        }
        loss = module.validation_step(batch, 0)
        assert torch.isfinite(loss)

    def test_compute_loss_with_variable_encoder_mask(self, module: ChantOMRLightningModule):
        batch = {
            "pixel_values": torch.randn(1, 3, 400, 1050),
            "input_ids": torch.tensor([[1, 5, 6, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "encoder_attention_mask": torch.ones(1, 12 * 32, dtype=torch.long),
        }
        loss = module._compute_loss(batch)
        assert torch.isfinite(loss)
