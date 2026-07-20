"""Score image preprocessing for inference (13a variable-height path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from chant_omr.data.dataset import (
    DEFAULT_MAX_HEIGHT,
    DEFAULT_TARGET_WIDTH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_score_image_array,
    normalize_pixel_batch,
    resize_score_image,
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


def prepare_inference_numpy(
    image_path: Path,
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> np.ndarray:
    """Load and normalize one score image to ``(1, 3, H, W)`` float32 numpy array.

    Numpy-only equivalent of :func:`prepare_inference_tensor` for use with
    ONNX Runtime (avoids device placement).
    """
    arr = load_score_image_array(
        Path(image_path),
        target_width=target_width,
        max_height=max_height,
    ).copy()
    pixels = arr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (pixels - mean) / std


def prepare_inference_numpy_from_array(
    img_array: np.ndarray,
    *,
    target_width: int = DEFAULT_TARGET_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    bgr: bool = False,
) -> np.ndarray:
    """Normalize an in-memory image to ``(1, 3, H, W)`` float32 numpy array.

    Accepts ``(H, W, 3)`` uint8 array (RGB by default, BGR if ``bgr=True``).
    Same output as :func:`prepare_inference_numpy` but without disk I/O.
    """
    if bgr:
        img_array = img_array[:, :, ::-1]
    pil_img = Image.fromarray(img_array, mode="RGB")
    resized = resize_score_image(pil_img, target_width=target_width, max_height=max_height)
    arr = np.asarray(resized, dtype=np.uint8).copy()
    pixels = arr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (pixels - mean) / std
