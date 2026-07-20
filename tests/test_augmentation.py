"""Tests for domain augmentation transforms."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chant_omr.data.augmentation import (
    AugmentationConfig,
    _apply_aging,
    _apply_barrel_distortion,
    _apply_foxing,
    _apply_ink_bleeding,
    _apply_ink_fade,
    _apply_ink_thickness,
    _apply_iron_gall,
    _apply_jpeg_compression,
    _apply_parchment_texture,
    _apply_perspective,
    _apply_salt_deposits,
    _apply_shadow,
    _apply_staff_hue_shift,
    _apply_uneven_lighting,
    _apply_water_stain,
    augment,
)


def _make_score_image(width: int = 400, height: int = 120) -> np.ndarray:
    """Create a synthetic score image with white background and dark marks."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    # Red staff lines
    for y in [20, 30, 40, 50, 60]:
        img[y, 10 : width - 10] = [180, 40, 40]
    # Black neume blobs
    for x in range(50, width - 50, 40):
        img[25:55, x : x + 10] = [20, 20, 20]
    return img


@pytest.fixture
def score_image() -> np.ndarray:
    return _make_score_image()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def texture_dir(tmp_path: Path) -> Path:
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        patch = rng.integers(160, 220, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(patch).save(tex_dir / f"parchment_{i:03d}.jpg")
    return tex_dir


class TestIndividualTransforms:
    """Each transform returns a valid uint8 image of the same shape."""

    def test_aging(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_aging(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, score_image)

    def test_foxing(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_foxing(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_water_stain(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_water_stain(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_ink_fade(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_ink_fade(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_staff_hue_shift(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_staff_hue_shift(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_ink_bleeding(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_ink_bleeding(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, score_image)

    def test_ink_thickness(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_ink_thickness(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, score_image)

    def test_iron_gall(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_iron_gall(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_salt_deposits(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_salt_deposits(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_perspective(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_perspective(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_uneven_lighting(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_uneven_lighting(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_shadow(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_shadow(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_barrel_distortion(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_barrel_distortion(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_jpeg_compression(self, score_image, rng):
        cfg = AugmentationConfig()
        out = _apply_jpeg_compression(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_parchment_texture(self, score_image, rng, texture_dir):
        cfg = AugmentationConfig(texture_dir=str(texture_dir))
        out = _apply_parchment_texture(score_image, cfg, rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, score_image)

    def test_parchment_no_textures_is_noop(self, score_image, rng, tmp_path):
        cfg = AugmentationConfig(texture_dir=str(tmp_path / "empty"))
        out = _apply_parchment_texture(score_image, cfg, rng)
        assert np.array_equal(out, score_image)


class TestAugmentPipeline:
    def test_output_shape_and_type(self, score_image, rng):
        cfg = AugmentationConfig(parchment_texture_prob=0.0)
        out = augment(score_image, config=cfg, rng=rng)
        assert out.shape == score_image.shape
        assert out.dtype == np.uint8

    def test_reproducible_with_same_seed(self, score_image):
        cfg = AugmentationConfig(parchment_texture_prob=0.0)
        a = augment(score_image, config=cfg, rng=np.random.default_rng(99))
        b = augment(score_image, config=cfg, rng=np.random.default_rng(99))
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self, score_image):
        cfg = AugmentationConfig(parchment_texture_prob=0.0)
        a = augment(score_image, config=cfg, rng=np.random.default_rng(1))
        b = augment(score_image, config=cfg, rng=np.random.default_rng(2))
        assert not np.array_equal(a, b)

    def test_all_transforms_fire_when_prob_1(self, score_image, texture_dir):
        cfg = AugmentationConfig(
            parchment_texture_prob=1.0,
            aging_prob=1.0,
            foxing_prob=1.0,
            water_stain_prob=1.0,
            ink_fade_prob=1.0,
            staff_hue_prob=1.0,
            ink_bleeding_prob=1.0,
            ink_thickness_prob=1.0,
            iron_gall_prob=1.0,
            salt_deposit_prob=1.0,
            perspective_prob=1.0,
            uneven_lighting_prob=1.0,
            shadow_prob=1.0,
            barrel_distortion_prob=1.0,
            jpeg_prob=1.0,
            texture_dir=str(texture_dir),
        )
        rng = np.random.default_rng(42)
        out = augment(score_image, config=cfg, rng=rng)
        assert out.shape == score_image.shape
        assert not np.array_equal(out, score_image)

    def test_no_transforms_fire_when_prob_0(self, score_image):
        cfg = AugmentationConfig(
            parchment_texture_prob=0.0,
            aging_prob=0.0,
            foxing_prob=0.0,
            water_stain_prob=0.0,
            ink_fade_prob=0.0,
            staff_hue_prob=0.0,
            ink_bleeding_prob=0.0,
            ink_thickness_prob=0.0,
            iron_gall_prob=0.0,
            salt_deposit_prob=0.0,
            perspective_prob=0.0,
            uneven_lighting_prob=0.0,
            shadow_prob=0.0,
            barrel_distortion_prob=0.0,
            jpeg_prob=0.0,
        )
        rng = np.random.default_rng(42)
        out = augment(score_image, config=cfg, rng=rng)
        assert np.array_equal(out, score_image)

    def test_rejects_non_uint8(self):
        bad = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="uint8"):
            augment(bad)

    def test_default_config_works(self, score_image):
        out = augment(score_image, rng=np.random.default_rng(0))
        assert out.shape == score_image.shape


class TestAugmentationOnlyTrainSplit:
    """Verify build_datasets applies augmentation to train only."""

    def test_val_dataset_has_augment_false(self, tmp_path):
        from chant_omr.data.dataset import build_datasets
        from chant_omr.model.tokenizer import train_tokenizer

        fixtures = Path(__file__).parent / "fixtures" / "gregobase"
        rendered = tmp_path / "rendered"
        rendered.mkdir()
        gabc_src = fixtures / "respice_domine.gabc"
        for stem_id in [101, 102, 103, 104, 105]:
            shutil.copy(gabc_src, rendered / f"{stem_id}.gabc")
            img = Image.new("RGB", (236, 80), color=(255, 255, 255))
            img.save(rendered / f"{stem_id}.png")

        tok = train_tokenizer(
            fixtures,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )
        train_ds, val_ds = build_datasets(
            rendered,
            tok,
            train_fraction=0.7,
            augment=True,
            exclude_test_split=False,
        )
        assert train_ds.augment is True
        assert val_ds.augment is False
