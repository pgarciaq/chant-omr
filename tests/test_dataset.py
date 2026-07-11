"""Tests for ChantOMRDataset and data loading."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from chant_omr.data.dataset import (
    ChantOMRDataset,
    build_dataloaders,
    build_datasets,
    build_encoder_attention_mask,
    catalog_id_from_render_stem,
    collate_chant_omr_batch,
    discover_rendered_pairs,
    resize_score_image,
    split_samples_by_catalog_id,
)
from chant_omr.model.encoder import patch_grid_size
from chant_omr.model.tokenizer import train_tokenizer

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"
RESPICE_GABC = GABC_FIXTURES / "respice_domine.gabc"
DOUBLE_HEADER_GABC = GABC_FIXTURES / "double_header.gabc"


def _write_fake_png(path: Path, size: tuple[int, int] = (236, 80)) -> None:
    image = Image.new("RGB", size, color=(255, 255, 255))
    pixels = image.load()
    for x in range(20, min(200, size[0])):
        pixels[x, size[1] // 2] = (180, 40, 40)
    image.save(path)


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
def rendered_dir(tmp_path: Path) -> Path:
    rendered = tmp_path / "rendered"
    rendered.mkdir()

    pairs = [
        ("9000", RESPICE_GABC, (236, 80)),
        ("9000_elem1", DOUBLE_HEADER_GABC, (236, 80)),
        ("9100", RESPICE_GABC, (400, 120)),
        ("9200", DOUBLE_HEADER_GABC, (236, 80)),
        ("9300", RESPICE_GABC, (600, 120)),
    ]
    for stem, gabc_src, size in pairs:
        shutil.copy(gabc_src, rendered / f"{stem}.gabc")
        _write_fake_png(rendered / f"{stem}.png", size=size)
    return rendered


class TestCatalogIdParsing:
    def test_bare_id(self):
        assert catalog_id_from_render_stem("9000") == 9000

    def test_elem_variant(self):
        assert catalog_id_from_render_stem("9000_elem1") == 9000


class TestDiscoverRenderedPairs:
    def test_finds_id_based_pairs(self, rendered_dir: Path):
        pairs = discover_rendered_pairs(rendered_dir, min_body_len=10)
        stems = {p.stem for p in pairs}
        assert stems == {"9000", "9000_elem1", "9100", "9200", "9300"}

    def test_ignores_unpaired_gabc(self, rendered_dir: Path):
        (rendered_dir / "--orphan-slug--.gabc").write_bytes(RESPICE_GABC.read_bytes())
        pairs = discover_rendered_pairs(rendered_dir, min_body_len=10)
        assert all(p.stem != "--orphan-slug--" for p in pairs)

    def test_ignores_unpaired_png(self, rendered_dir: Path):
        _write_fake_png(rendered_dir / "9999.png")
        pairs = discover_rendered_pairs(rendered_dir, min_body_len=10)
        assert all(p.stem != "9999" for p in pairs)


class TestResizeScoreImage:
    def test_scales_to_target_width(self):
        image = Image.new("RGB", (1182, 400), color=(255, 255, 255))
        resized = resize_score_image(image, target_width=1050, max_height=1600)
        assert resized.size[0] == 1050
        assert resized.size[1] == pytest.approx(356, abs=2)

    def test_caps_height(self):
        image = Image.new("RGB", (1182, 3200), color=(255, 255, 255))
        resized = resize_score_image(image, target_width=1050, max_height=1600)
        assert resized.size == (591, 1600)


class TestSplitByCatalogId:
    def test_variants_share_partition(self, rendered_dir: Path):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        train, val = split_samples_by_catalog_id(samples, train_fraction=0.6, seed=7)
        train_ids = {s.catalog_id for s in train}
        val_ids = {s.catalog_id for s in val}
        assert 9000 in train_ids or 9000 in val_ids
        assert not (9000 in train_ids and 9000 in val_ids)
        train_stems = {s.stem for s in train}
        val_stems = {s.stem for s in val}
        if 9000 in train_ids:
            assert {"9000", "9000_elem1"}.issubset(train_stems)
        else:
            assert {"9000", "9000_elem1"}.issubset(val_stems)

    def test_reproducible(self, rendered_dir: Path):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        train_a, val_a = split_samples_by_catalog_id(samples, seed=123)
        train_b, val_b = split_samples_by_catalog_id(samples, seed=123)
        assert [s.stem for s in train_a] == [s.stem for s in train_b]
        assert [s.stem for s in val_a] == [s.stem for s in val_b]


class TestChantOMRDataset:
    def test_len_and_getitem_shapes(self, rendered_dir: Path, tokenizer):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        dataset = ChantOMRDataset(samples[:2], tokenizer, augment=False)
        assert len(dataset) == 2
        item = dataset[0]
        assert item["image"].shape[2] == 3
        assert item["image"].dtype == np.uint8
        assert len(item["input_ids"]) > 2
        assert item["attention_mask"] == [1] * len(item["input_ids"])
        assert "name:" not in tokenizer.decode(item["input_ids"])

    def test_augmentation_toggle(self, rendered_dir: Path, tokenizer):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        rng = np.random.default_rng(0)
        off = ChantOMRDataset(samples[:1], tokenizer, augment=False)
        on = ChantOMRDataset(samples[:1], tokenizer, augment=True, rng=rng)
        image_off = off[0]["image"]
        image_on = on[0]["image"]
        assert not np.array_equal(image_off, image_on)
        assert off[0]["input_ids"] == on[0]["input_ids"]


class TestCollateBatch:
    def test_collate_shapes(self, rendered_dir: Path, tokenizer):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        dataset = ChantOMRDataset(samples[:3], tokenizer, augment=False)
        batch = [dataset[i] for i in range(3)]
        collated = collate_chant_omr_batch(
            batch,
            pad_token_id=tokenizer.pad_id,
            max_seq_len=512,
        )
        assert collated["pixel_values"].shape[0] == 3
        assert collated["pixel_values"].shape[1] == 3
        assert collated["input_ids"].shape[0] == 3
        assert collated["attention_mask"].sum(dim=1).tolist() == [
            len(batch[i]["input_ids"]) for i in range(3)
        ]

    def test_encoder_attention_mask_shape(self, rendered_dir: Path, tokenizer):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        dataset = ChantOMRDataset(samples[:3], tokenizer, augment=False)
        batch = [dataset[i] for i in range(3)]
        collated = collate_chant_omr_batch(
            batch,
            pad_token_id=tokenizer.pad_id,
            max_seq_len=512,
        )
        max_h = max(item["image"].shape[0] for item in batch)
        max_w = max(item["image"].shape[1] for item in batch)
        grid_h, grid_w = patch_grid_size(max_h, max_w)
        assert collated["encoder_attention_mask"].shape == (3, grid_h * grid_w)

    def test_encoder_attention_mask_zeros_padded_rows(self, rendered_dir: Path, tokenizer):
        samples = discover_rendered_pairs(rendered_dir, min_body_len=10)
        dataset = ChantOMRDataset(samples, tokenizer, augment=False)
        batch = [dataset[i] for i in range(3)]
        collated = collate_chant_omr_batch(
            batch,
            pad_token_id=tokenizer.pad_id,
            max_seq_len=512,
        )
        max_h = max(item["image"].shape[0] for item in batch)
        max_w = max(item["image"].shape[1] for item in batch)
        grid_h, grid_w = patch_grid_size(max_h, max_w)
        for row, item in enumerate(batch):
            valid_h = item["image"].shape[0] // 32
            mask = collated["encoder_attention_mask"][row].view(grid_h, grid_w)
            if valid_h < grid_h:
                assert mask[valid_h:, :].sum() == 0
            assert mask[:valid_h, :grid_w].sum() == valid_h * grid_w


class TestEncoderAttentionMaskHelper:
    def test_build_encoder_attention_mask(self):
        mask = build_encoder_attention_mask(
            [(400, 1050), (800, 1050)],
            padded_height=800,
            padded_width=1050,
        )
        grid_h, grid_w = patch_grid_size(800, 1050)
        assert mask.shape == (2, grid_h * grid_w)
        assert mask[0].sum() == (400 // 32) * grid_w
        assert mask[1].sum() == grid_h * grid_w


class TestDataLoaderIntegration:
    def test_dataloader_batch(self, rendered_dir: Path, tokenizer):
        train_ds, val_ds = build_datasets(
            rendered_dir,
            tokenizer,
            train_fraction=0.6,
            split_seed=99,
            augment=False,
        )
        train_loader, val_loader = build_dataloaders(
            train_ds,
            val_ds,
            batch_size=2,
            num_workers=0,
            max_seq_len=512,
        )
        batch = next(iter(train_loader))
        assert batch["pixel_values"].dtype == torch.float32
        assert batch["pixel_values"].shape[0] <= 2
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["encoder_attention_mask"].shape[0] == batch["pixel_values"].shape[0]
        assert len(val_loader) >= 1
