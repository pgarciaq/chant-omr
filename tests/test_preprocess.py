"""Tests for inference preprocessing (numpy-array parity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chant_omr.inference.preprocess import (
    prepare_inference_numpy,
    prepare_inference_numpy_from_array,
)


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a small random RGB PNG on disk."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (200, 600, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "score.png"
    img.save(path)
    return path


class TestPrepareInferenceNumpyFromArray:
    def test_parity_with_disk_path(self, sample_png: Path):
        """from_array produces identical output to the file-path version."""
        from_disk = prepare_inference_numpy(sample_png)

        rgb_array = np.asarray(Image.open(sample_png).convert("RGB"))
        from_array = prepare_inference_numpy_from_array(rgb_array)

        assert from_disk.shape == from_array.shape
        assert from_disk.dtype == from_array.dtype
        np.testing.assert_allclose(from_array, from_disk, atol=1e-6)

    def test_bgr_flag_swaps_channels(self, sample_png: Path):
        """BGR input with bgr=True produces same result as RGB input."""
        rgb_array = np.asarray(Image.open(sample_png).convert("RGB"))
        bgr_array = rgb_array[:, :, ::-1].copy()

        from_rgb = prepare_inference_numpy_from_array(rgb_array)
        from_bgr = prepare_inference_numpy_from_array(bgr_array, bgr=True)

        np.testing.assert_allclose(from_bgr, from_rgb, atol=1e-6)

    def test_output_shape_and_dtype(self, sample_png: Path):
        rgb_array = np.asarray(Image.open(sample_png).convert("RGB"))
        result = prepare_inference_numpy_from_array(rgb_array)

        assert result.ndim == 4
        assert result.shape[0] == 1
        assert result.shape[1] == 3
        assert result.dtype == np.float32

    def test_custom_target_width(self, sample_png: Path):
        rgb_array = np.asarray(Image.open(sample_png).convert("RGB"))
        result = prepare_inference_numpy_from_array(rgb_array, target_width=512)
        assert result.shape[3] == 512
