"""Tests for full ChantOMR model assembly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chant_omr.model.chant_omr_model import (
    ChantOMR,
    ChantOMRConfig,
    ParameterBreakdown,
    Sinusoidal2DPositionalEncoding,
    build_model,
    count_model_parameters,
)

TARGET_PARAM_COUNT = 59_000_000
PARAM_TOLERANCE = 0.10


@pytest.fixture
def config() -> ChantOMRConfig:
    return ChantOMRConfig()


@pytest.fixture
def model(config: ChantOMRConfig) -> ChantOMR:
    return build_model(config, encoder_pretrained=False)


class TestBuildModel:
    def test_returns_chant_omr(self, model: ChantOMR):
        assert isinstance(model, ChantOMR)
        assert model.encoder is not None
        assert model.positional_encoding is not None
        assert model.projector is not None
        assert model.decoder is not None

    def test_config_from_yaml(self, tmp_path: Path):
        config_path = tmp_path / "default.yaml"
        config_path.write_text(
            yaml.dump({"model": {"encoder_variant": "convnextv2_tiny", "d_model": 512}}),
            encoding="utf-8",
        )
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = ChantOMRConfig.from_mapping(data["model"])
        assert cfg.encoder_variant == "convnextv2_tiny"
        assert cfg.d_model == 512


class TestSinusoidal2D:
    def test_adds_no_learned_params(self):
        pe = Sinusoidal2DPositionalEncoding(768)
        assert sum(p.numel() for p in pe.parameters()) == 0

    def test_output_shape_matches_input(self):
        pe = Sinusoidal2DPositionalEncoding(768)
        feature_map = torch.randn(2, 768, 12, 32)
        output = pe(feature_map)
        assert output.shape == feature_map.shape
        assert not torch.allclose(output, feature_map)


class TestE2EForward:
    def test_e2e_forward_shape(self, model: ChantOMR):
        batch, seq_len = 2, 16
        pixel_values = torch.randn(batch, 3, 980, 1050)
        input_ids = torch.randint(0, 2048, (batch, seq_len))
        logits = model(pixel_values, input_ids)
        assert logits.shape == (batch, seq_len, 2048)

    def test_variable_height_images(self, model: ChantOMR):
        input_ids = torch.randint(0, 2048, (1, 8))
        short_logits = model(torch.randn(1, 3, 400, 1050), input_ids)
        tall_logits = model(torch.randn(1, 3, 1200, 1050), input_ids)
        assert short_logits.shape == (1, 8, 2048)
        assert tall_logits.shape == (1, 8, 2048)
        assert short_logits.shape[-1] == tall_logits.shape[-1]

    def test_encoder_attention_mask_forwarded(self, model: ChantOMR):
        pixel_values = torch.randn(1, 3, 512, 1050)
        input_ids = torch.randint(1, 100, (1, 6))
        encoder_output = model.encoder(pixel_values)
        num_patches = encoder_output.num_patches
        mask = torch.ones(1, num_patches, dtype=torch.long)
        mask[:, num_patches // 2 :] = 0
        logits = model(
            pixel_values,
            input_ids,
            encoder_attention_mask=mask,
        )
        assert logits.shape == (1, 6, 2048)


class TestParameterCount:
    def test_param_count_within_target(self, model: ChantOMR):
        breakdown = count_model_parameters(model)
        assert isinstance(breakdown, ParameterBreakdown)
        assert breakdown.positional_encoding == 0
        assert breakdown.encoder > 0
        assert breakdown.projector > 0
        assert breakdown.decoder > 0

        total = breakdown.total
        lower = TARGET_PARAM_COUNT * (1.0 - PARAM_TOLERANCE)
        upper = TARGET_PARAM_COUNT * (1.0 + PARAM_TOLERANCE)
        assert lower <= total <= upper, (
            f"expected ~{TARGET_PARAM_COUNT} params (+/- {PARAM_TOLERANCE:.0%}), "
            f"got {total} (encoder={breakdown.encoder}, projector={breakdown.projector}, "
            f"decoder={breakdown.decoder})"
        )
