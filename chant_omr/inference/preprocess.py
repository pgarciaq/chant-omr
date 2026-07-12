"""Score image preprocessing for inference (13a variable-height path)."""

from __future__ import annotations

from pathlib import Path

import torch

from chant_omr.data.dataset import (
    DEFAULT_MAX_HEIGHT,
    DEFAULT_TARGET_WIDTH,
    load_score_image_array,
    normalize_pixel_batch,
)


def prepare_inference_tensor(
    image_path: Path,
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and normalize one score image to ``(1, 3, H, W)`` float tensor."""
    arr = load_score_image_array(
        Path(image_path),
        target_width=target_width,
        max_height=max_height,
    ).copy()
    pixels = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    pixels = normalize_pixel_batch(pixels)
    if device is not None:
        pixels = pixels.to(device)
    return pixels
