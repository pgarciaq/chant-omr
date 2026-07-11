"""Tests for Transformer decoder."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chant_omr.model.decoder import (
    ChantDecoder,
    DecoderConfig,
    build_decoder,
    count_parameters,
)


@pytest.fixture
def config() -> DecoderConfig:
    return DecoderConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=1024,
        dropout=0.0,
        max_seq_len=2048,
        vocab_size=2048,
    )


@pytest.fixture
def decoder(config: DecoderConfig) -> ChantDecoder:
    return build_decoder(config)


class TestDecoderConfig:
    def test_from_mapping_defaults(self):
        cfg = DecoderConfig.from_mapping({})
        assert cfg.d_model == 512
        assert cfg.n_layers == 8
        assert cfg.vocab_size == 2048

    def test_from_yaml_model_section(self, tmp_path: Path):
        config_path = tmp_path / "default.yaml"
        config_path.write_text(
            yaml.dump({"model": {"d_model": 256, "n_layers": 2, "n_heads": 4, "d_ff": 512}}),
            encoding="utf-8",
        )
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = DecoderConfig.from_mapping(data["model"])
        assert cfg.d_model == 256
        assert cfg.n_layers == 2

    def test_rejects_invalid_head_split(self):
        with pytest.raises(ValueError, match="divisible"):
            DecoderConfig(d_model=512, n_heads=7)


class TestBuildDecoder:
    def test_returns_chant_decoder(self, decoder: ChantDecoder):
        assert isinstance(decoder, ChantDecoder)
        assert len(decoder.layers) == 8


class TestDecoderForward:
    def test_decoder_forward_shape(self, decoder: ChantDecoder):
        batch, seq_len, enc_len = 2, 16, 64
        input_ids = torch.randint(0, 2048, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)
        logits = decoder(input_ids, encoder_memory)
        assert logits.shape == (batch, seq_len, 2048)

    def test_rejects_encoder_dim_mismatch(self, decoder: ChantDecoder):
        input_ids = torch.randint(0, 2048, (1, 8))
        encoder_memory = torch.randn(1, 32, 768)
        with pytest.raises(ValueError, match="d_model"):
            decoder(input_ids, encoder_memory)

    def test_rejects_batch_mismatch(self, decoder: ChantDecoder):
        input_ids = torch.randint(0, 2048, (2, 8))
        encoder_memory = torch.randn(1, 32, 512)
        with pytest.raises(ValueError, match="batch size"):
            decoder(input_ids, encoder_memory)

    def test_attention_mask_allows_padding(self, decoder: ChantDecoder):
        batch, seq_len, enc_len = 1, 6, 32
        input_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        encoder_memory = torch.randn(batch, enc_len, 512)
        logits = decoder(input_ids, encoder_memory, attention_mask=attention_mask)
        assert logits.shape == (batch, seq_len, 2048)
        assert torch.isfinite(logits).all()


class TestCausalMask:
    def test_causal_mask(self, decoder: ChantDecoder):
        torch.manual_seed(0)
        batch, seq_len, enc_len = 1, 10, 32
        base_ids = torch.randint(1, 100, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)

        decoder.eval()
        with torch.no_grad():
            base_logits = decoder(base_ids, encoder_memory)

        position = 4
        modified_ids = base_ids.clone()
        modified_ids[:, position + 1 :] = torch.randint(100, 200, (batch, seq_len - position - 1))
        with torch.no_grad():
            modified_logits = decoder(modified_ids, encoder_memory)

        assert torch.allclose(
            base_logits[:, : position + 1],
            modified_logits[:, : position + 1],
            atol=1e-5,
            rtol=1e-5,
        )
        assert not torch.allclose(
            base_logits[:, position + 1 :],
            modified_logits[:, position + 1 :],
        )


class TestEncoderMask:
    def test_encoder_mask_blocks_padded_patches(self, decoder: ChantDecoder):
        torch.manual_seed(1)
        batch, seq_len, enc_len = 1, 6, 8
        input_ids = torch.randint(1, 50, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)

        all_valid = torch.ones(batch, enc_len, dtype=torch.long)
        mask_head = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

        modified_memory = encoder_memory.clone()
        modified_memory[:, 4:, :] = 999.0

        decoder.eval()
        with torch.no_grad():
            logits_clean = decoder(input_ids, encoder_memory, encoder_attention_mask=mask_head)
            logits_corrupt = decoder(
                input_ids,
                modified_memory,
                encoder_attention_mask=mask_head,
            )
            logits_all = decoder(input_ids, modified_memory, encoder_attention_mask=all_valid)

        assert torch.allclose(logits_clean, logits_corrupt, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(logits_clean, logits_all)


class TestParameterCount:
    def test_decoder_parameter_budget(self, decoder: ChantDecoder):
        params = count_parameters(decoder)
        assert 20_000_000 <= params <= 35_000_000
