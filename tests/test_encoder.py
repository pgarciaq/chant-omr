"""Tests for ConvNeXt-V2 visual encoder."""

from __future__ import annotations

import pytest
import torch

from chant_omr.model.encoder import (
    DEFAULT_OUTPUT_STRIDE,
    ChantEncoder,
    build_encoder,
    resolve_encoder_name,
)


@pytest.fixture
def encoder() -> ChantEncoder:
    model, _embed_dim, _stride = build_encoder(pretrained=False)
    return model


class TestResolveEncoderName:
    def test_alias_maps_to_timm_name(self):
        assert resolve_encoder_name("convnextv2_tiny") == (
            "convnextv2_tiny.fcmae_ft_in22k_in1k"
        )

    def test_passthrough_full_name(self):
        name = "convnextv2_nano.fcmae_ft_in22k_in1k"
        assert resolve_encoder_name(name) == name


class TestBuildEncoder:
    def test_returns_tiny_with_768_dim(self):
        encoder, embed_dim, stride = build_encoder(
            variant="convnextv2_tiny",
            pretrained=False,
        )
        assert isinstance(encoder, ChantEncoder)
        assert embed_dim == 768
        assert stride == DEFAULT_OUTPUT_STRIDE
        assert encoder.embed_dim == 768


class TestEncoderForward:
    def test_output_shape_median_height(self, encoder: ChantEncoder):
        pixel_values = torch.randn(2, 3, 980, 1050)
        output = encoder(pixel_values)
        assert output.feature_map.shape == (2, 768, 30, 32)
        assert output.memory.shape == (2, 30 * 32, 768)
        assert output.grid_size == (30, 32)
        assert output.num_patches == 960

    def test_output_shape_short_chant(self, encoder: ChantEncoder):
        pixel_values = torch.randn(1, 3, 400, 1050)
        output = encoder(pixel_values)
        assert output.feature_map.shape == (1, 768, 12, 32)
        assert output.memory.shape == (1, 12 * 32, 768)
        assert output.grid_size == (12, 32)

    def test_output_shape_tall_chant(self, encoder: ChantEncoder):
        pixel_values = torch.randn(1, 3, 1600, 591)
        output = encoder(pixel_values)
        assert output.feature_map.shape == (1, 768, 50, 18)
        assert output.grid_size == (50, 18)

    def test_feature_dim_matches_config(self, encoder: ChantEncoder):
        pixel_values = torch.randn(1, 3, 512, 1050)
        output = encoder(pixel_values)
        assert output.embed_dim == 768

    def test_memory_matches_flattened_feature_map(self, encoder: ChantEncoder):
        pixel_values = torch.randn(1, 3, 600, 1050)
        output = encoder(pixel_values)
        expected = (
            output.feature_map.flatten(2).transpose(1, 2).contiguous()
        )
        assert torch.equal(output.memory, expected)

    def test_rejects_invalid_input_channels(self, encoder: ChantEncoder):
        with pytest.raises(ValueError, match="expected input"):
            encoder(torch.randn(1, 1, 400, 1050))
