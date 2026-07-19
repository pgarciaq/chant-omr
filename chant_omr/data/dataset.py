"""PyTorch dataset for (image, GABC) training pairs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chant_omr.data.augmentation import AugmentationConfig, augment
from chant_omr.data.gabc_parser import (
    DEFAULT_MIN_BODY_LEN,
    extract_gabc_body,
    plain_gabc_reject_reason,
)
from chant_omr.model.encoder import DEFAULT_OUTPUT_STRIDE, patch_grid_size
from chant_omr.model.tokenizer import TOKENIZER_FILENAME, GABCTokenizer

DEFAULT_PATCH_STRIDE = DEFAULT_OUTPUT_STRIDE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_TARGET_WIDTH = 1050
DEFAULT_MAX_HEIGHT = 1600
DEFAULT_TRAIN_SPLIT = 0.9
DEFAULT_SPLIT_SEED = 42
TEST_SPLIT_MODULUS = 20


@dataclass(frozen=True)
class RenderedSample:
    """One PNG + GABC training pair under ``data/rendered/``."""

    png_path: Path
    gabc_path: Path
    catalog_id: int
    stem: str


def catalog_id_from_render_stem(stem: str) -> int:
    """Parse GregoBase catalog id from ``{id}`` or ``{id}_elem{N}`` rendered stem."""
    base = stem.split("_elem", 1)[0]
    return int(base)


def is_test_split(catalog_id: int, *, modulus: int = TEST_SPLIT_MODULUS) -> bool:
    """Return True if *catalog_id* belongs to the held-out test split.

    Uses a simple ``catalog_id % modulus == 0`` predicate — stable regardless
    of which files exist on disk (unlike RNG-shuffle splits).  With the
    default modulus of 20 this holds out ~5% of catalog IDs.
    """
    return catalog_id % modulus == 0


def discover_rendered_pairs(
    rendered_dir: Path,
    *,
    min_body_len: int = DEFAULT_MIN_BODY_LEN,
) -> list[RenderedSample]:
    """Return plain trainable PNG+GABC pairs (PNG-first indexing)."""
    rendered_dir = Path(rendered_dir)
    samples: list[RenderedSample] = []

    for png_path in sorted(rendered_dir.glob("*.png")):
        gabc_path = rendered_dir / f"{png_path.stem}.gabc"
        if not gabc_path.is_file():
            continue
        raw = gabc_path.read_bytes()
        if plain_gabc_reject_reason(raw, min_body_len=min_body_len):
            continue
        try:
            catalog_id = catalog_id_from_render_stem(png_path.stem)
        except ValueError:
            continue
        samples.append(
            RenderedSample(
                png_path=png_path,
                gabc_path=gabc_path,
                catalog_id=catalog_id,
                stem=png_path.stem,
            )
        )
    return samples


def split_samples_by_catalog_id(
    samples: Sequence[RenderedSample],
    *,
    train_fraction: float = DEFAULT_TRAIN_SPLIT,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[RenderedSample], list[RenderedSample]]:
    """Split samples by catalog id so elem variants stay in one partition."""
    ids = sorted({sample.catalog_id for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_fraction)) if len(ids) > 1 else 1
    if n_train >= len(ids) and len(ids) > 1:
        n_train = len(ids) - 1
    train_ids = set(ids[:n_train])
    train = [s for s in samples if s.catalog_id in train_ids]
    val = [s for s in samples if s.catalog_id not in train_ids]
    return train, val


def resize_score_image(
    image: Image.Image,
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> Image.Image:
    """Scale to *target_width* preserving aspect ratio; cap height at *max_height*."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image.size}")

    scale = target_width / width
    new_width = target_width
    new_height = max(1, round(height * scale))

    if new_height > max_height:
        scale = max_height / height
        new_width = max(1, round(width * scale))
        new_height = max_height

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_score_image_array(
    png_path: Path,
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> np.ndarray:
    """Load PNG as uint8 ``(H, W, 3)`` after model resize policy."""
    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        resized = resize_score_image(rgb, target_width=target_width, max_height=max_height)
        return np.asarray(resized, dtype=np.uint8)


def normalize_pixel_batch(pixels: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a ``(B, 3, H, W)`` float tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    return (pixels - mean) / std


def build_encoder_attention_mask(
    image_sizes: Sequence[tuple[int, int]],
    *,
    padded_height: int,
    padded_width: int,
    stride: int = DEFAULT_PATCH_STRIDE,
) -> torch.Tensor:
    """Build ``(B, H'W')`` mask marking valid (non-padded) encoder patches."""
    grid_h, grid_w = patch_grid_size(padded_height, padded_width, stride=stride)
    masks: list[torch.Tensor] = []
    for height, width in image_sizes:
        valid_h = min(grid_h, height // stride)
        valid_w = min(grid_w, width // stride)
        mask_2d = torch.zeros(grid_h, grid_w, dtype=torch.long)
        mask_2d[:valid_h, :valid_w] = 1
        masks.append(mask_2d.flatten())
    return torch.stack(masks)


def collate_chant_omr_batch(
    batch: list[dict],
    *,
    pad_token_id: int,
    max_seq_len: int,
    normalize: bool = True,
    patch_stride: int = DEFAULT_PATCH_STRIDE,
) -> dict[str, torch.Tensor]:
    """Pad variable-height images and token sequences into one batch."""
    if not batch:
        raise ValueError("empty batch")

    max_h = max(item["image"].shape[0] for item in batch)
    max_w = max(item["image"].shape[1] for item in batch)
    image_sizes: list[tuple[int, int]] = []

    images = []
    for item in batch:
        arr = item["image"]
        h, w, _ = arr.shape
        image_sizes.append((h, w))
        padded = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        padded[:h, :w] = arr
        images.append(torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0)

    pixel_values = torch.stack(images)
    if normalize:
        pixel_values = normalize_pixel_batch(pixel_values)

    encoder_attention_mask = build_encoder_attention_mask(
        image_sizes,
        padded_height=max_h,
        padded_width=max_w,
        stride=patch_stride,
    )

    max_tokens = min(max(len(item["input_ids"]) for item in batch), max_seq_len)
    input_ids = torch.full((len(batch), max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_tokens), dtype=torch.long)
    for row, item in enumerate(batch):
        ids = item["input_ids"][:max_seq_len]
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_attention_mask": encoder_attention_mask,
    }


class ChantOMRDataset(Dataset):
    """Dataset of (rendered score image, tokenized GABC body) pairs."""

    def __init__(
        self,
        samples: Sequence[RenderedSample],
        tokenizer: GABCTokenizer,
        *,
        augment: bool = False,
        target_width: int = DEFAULT_TARGET_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
        augmentation_config: AugmentationConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not samples:
            raise ValueError("dataset requires at least one rendered pair")
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.augment = augment
        self.target_width = target_width
        self.max_height = max_height
        self.augmentation_config = augmentation_config
        self._rng = rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = load_score_image_array(
            sample.png_path,
            target_width=self.target_width,
            max_height=self.max_height,
        )
        if self.augment:
            rng = self._rng if self._rng is not None else np.random.default_rng()
            image = augment(image, config=self.augmentation_config, rng=rng)

        raw = sample.gabc_path.read_bytes()
        body = extract_gabc_body(raw.decode("utf-8"))
        input_ids = self.tokenizer.encode(body, add_special_tokens=True)
        attention_mask = [1] * len(input_ids)

        return {
            "image": image,
            "image_height": image.shape[0],
            "image_width": image.shape[1],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "stem": sample.stem,
            "catalog_id": sample.catalog_id,
        }


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_datasets(
    rendered_dir: Path,
    tokenizer: GABCTokenizer,
    *,
    train_fraction: float = DEFAULT_TRAIN_SPLIT,
    split_seed: int = DEFAULT_SPLIT_SEED,
    augment: bool = False,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    min_body_len: int = DEFAULT_MIN_BODY_LEN,
    overfit_n: int | None = None,
    exclude_test_split: bool = True,
    augmentation_config: AugmentationConfig | None = None,
) -> tuple[ChantOMRDataset, ChantOMRDataset]:
    """Discover pairs and return train/val datasets split by catalog id.

    When *exclude_test_split* is True (the default), samples whose catalog ID
    belongs to the test split (``catalog_id % 20 == 0``) are removed before
    the train/val split.  This ensures the test set is never seen during
    training or validation.

    Augmentation is applied to the **training** set only — validation always
    sees clean images for consistent loss measurement.
    """
    samples = discover_rendered_pairs(rendered_dir, min_body_len=min_body_len)
    if exclude_test_split:
        samples = [s for s in samples if not is_test_split(s.catalog_id)]
    if not samples:
        raise ValueError(f"no rendered training pairs found under {rendered_dir}")
    train_samples, val_samples = split_samples_by_catalog_id(
        samples,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    if not val_samples:
        val_samples = train_samples[-1:]
        train_samples = train_samples[:-1]
    if overfit_n is not None:
        if overfit_n < 1:
            raise ValueError("overfit_n must be >= 1")
        train_samples = train_samples[:overfit_n]
        val_samples = train_samples[:1]
    common = {
        "tokenizer": tokenizer,
        "target_width": target_width,
        "max_height": max_height,
    }
    return (
        ChantOMRDataset(
            train_samples,
            augment=augment,
            augmentation_config=augmentation_config,
            **common,
        ),
        ChantOMRDataset(val_samples, augment=False, **common),
    )


def build_dataloaders(
    train_dataset: ChantOMRDataset,
    val_dataset: ChantOMRDataset,
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    max_seq_len: int = 2048,
    pad_token_id: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders with batch padding collate."""
    pad_id = pad_token_id if pad_token_id is not None else train_dataset.tokenizer.pad_id

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_chant_omr_batch(
            batch,
            pad_token_id=pad_id,
            max_seq_len=max_seq_len,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_dataloaders_from_config(
    config_path: Path,
    *,
    tokenizer: GABCTokenizer | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build dataloaders from ``configs/default.yaml`` (or similar)."""
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    rendered_dir = Path(data_cfg.get("rendered_dir", "data/rendered/"))
    tokenizer_dir = Path(data_cfg.get("tokenizer_dir", "data/tokenizer/"))
    tok = tokenizer or GABCTokenizer.load(tokenizer_dir / TOKENIZER_FILENAME)

    aug_config: AugmentationConfig | None = None
    do_augment = bool(data_cfg.get("augment", False))
    if do_augment:
        aug_kwargs: dict = {}
        texture_dir = data_cfg.get("texture_dir")
        if texture_dir:
            aug_kwargs["texture_dir"] = texture_dir
        aug_config = AugmentationConfig(**aug_kwargs)

    train_ds, val_ds = build_datasets(
        rendered_dir,
        tok,
        train_fraction=float(data_cfg.get("train_split", DEFAULT_TRAIN_SPLIT)),
        augment=do_augment,
        target_width=int(data_cfg.get("target_width", DEFAULT_TARGET_WIDTH)),
        max_height=int(data_cfg.get("max_height", DEFAULT_MAX_HEIGHT)),
        augmentation_config=aug_config,
    )
    return build_dataloaders(
        train_ds,
        val_ds,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
        pad_token_id=tok.pad_id,
    )
