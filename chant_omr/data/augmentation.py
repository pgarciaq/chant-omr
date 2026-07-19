"""Domain augmentation to make synthetic renders look like real photographs.

Bridges the domain gap between clean Gregorio renders and photographs of
historical manuscripts. Applied during training to create diverse, realistic
training examples from clean synthetic source images.

Augmentation categories:
1. Substrate (parchment texture, aging, foxing, water stains)
2. Ink appearance (fading, staff hue variation)
3. Degradation (iron gall corrosion, salt deposits)
4. Photography (perspective skew, uneven lighting, shadow, barrel distortion)
5. Compression artifacts (JPEG quality variation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AugmentationConfig:
    """Controls augmentation intensity and probability."""

    # Substrate
    parchment_texture_prob: float = 0.8
    parchment_blend_range: tuple[float, float] = (0.15, 0.45)
    aging_prob: float = 0.6
    aging_intensity: tuple[float, float] = (0.05, 0.3)
    foxing_prob: float = 0.3
    foxing_count_range: tuple[int, int] = (5, 40)
    water_stain_prob: float = 0.15

    # Ink
    ink_fade_prob: float = 0.2
    ink_fade_range: tuple[float, float] = (0.7, 0.95)
    staff_hue_prob: float = 0.3
    staff_hue_range: tuple[float, float] = (-15.0, 15.0)

    # Degradation
    iron_gall_prob: float = 0.1
    salt_deposit_prob: float = 0.05
    salt_count_range: tuple[int, int] = (3, 15)

    # Photography
    perspective_prob: float = 0.3
    perspective_max_px: int = 30
    uneven_lighting_prob: float = 0.5
    lighting_intensity_range: tuple[float, float] = (0.7, 1.0)
    shadow_prob: float = 0.3
    shadow_intensity_range: tuple[float, float] = (0.5, 0.85)
    barrel_distortion_prob: float = 0.15
    barrel_k_range: tuple[float, float] = (-0.08, 0.08)

    # Compression
    jpeg_prob: float = 0.5
    jpeg_quality_range: tuple[int, int] = (60, 95)

    # Texture directory (real parchment patches)
    texture_dir: str = "data/textures"

    # Cached texture arrays (loaded lazily)
    _textures: list[np.ndarray] = field(default_factory=list, repr=False)


def _load_textures(config: AugmentationConfig) -> list[np.ndarray]:
    """Lazily load parchment texture patches from disk."""
    if config._textures:
        return config._textures
    tex_dir = Path(config.texture_dir)
    if tex_dir.is_dir():
        for p in sorted(tex_dir.glob("parchment_*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                config._textures.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return config._textures


def _apply_parchment_texture(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Blend a real parchment texture patch onto the image background."""
    textures = _load_textures(config)
    if not textures:
        return image
    tex = textures[rng.integers(len(textures))]
    h, w = image.shape[:2]
    tex_resized = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = rng.uniform(*config.parchment_blend_range)

    # Blend more on white/light areas (background), less on dark areas (ink)
    gray = image.mean(axis=2, keepdims=True) / 255.0
    blend_mask = gray ** 0.5  # stronger blend on lighter pixels
    blended = image.astype(np.float32) * (1 - alpha * blend_mask) + \
              tex_resized.astype(np.float32) * (alpha * blend_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_aging(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add warm yellowish aging tint."""
    intensity = rng.uniform(*config.aging_intensity)
    tint = np.array([intensity * 25, intensity * 15, -intensity * 20], dtype=np.float32)
    aged = image.astype(np.float32) + tint
    return np.clip(aged, 0, 255).astype(np.uint8)


def _apply_foxing(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add small brownish foxing spots (fungal damage)."""
    h, w = image.shape[:2]
    n_spots = rng.integers(*config.foxing_count_range)
    out = image.astype(np.float32)
    for _ in range(n_spots):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        radius = rng.integers(2, 8)
        color = np.array([
            rng.uniform(120, 170),
            rng.uniform(90, 130),
            rng.uniform(50, 90),
        ], dtype=np.float32)
        opacity = rng.uniform(0.2, 0.6)

        y0, y1 = max(0, cy - radius), min(h, cy + radius)
        x0, x1 = max(0, cx - radius), min(w, cx + radius)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = np.clip(1.0 - dist / radius, 0, 1)

        for c in range(3):
            out[y0:y1, x0:x1, c] = (
                out[y0:y1, x0:x1, c] * (1 - mask * opacity) +
                color[c] * mask * opacity
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_water_stain(
    image: np.ndarray, _config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add a large semi-transparent water stain blob."""
    h, w = image.shape[:2]
    cy = rng.integers(h // 4, 3 * h // 4)
    cx = rng.integers(w // 4, 3 * w // 4)
    ry = rng.integers(h // 8, h // 3)
    rx = rng.integers(w // 8, w // 3)

    yy, xx = np.mgrid[0:h, 0:w]
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = np.clip(1.0 - ellipse, 0, 1) ** 2
    # Irregular edge via noise
    noise = cv2.GaussianBlur(
        rng.standard_normal((h, w)).astype(np.float32),
        (0, 0), sigmaX=max(1, min(rx, ry) // 3)
    )
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    mask = mask * (0.5 + 0.5 * noise)

    opacity = rng.uniform(0.08, 0.25)
    stain_color = np.array([
        rng.uniform(160, 200),
        rng.uniform(140, 180),
        rng.uniform(100, 140),
    ], dtype=np.float32)

    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1 - mask * opacity) + stain_color[c] * mask * opacity
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_ink_fade(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Reduce contrast and lighten dark pixels to simulate aged ink."""
    factor = rng.uniform(*config.ink_fade_range)
    gray = image.mean(axis=2)
    dark_mask = (gray < 128).astype(np.float32)
    dark_mask = cv2.GaussianBlur(dark_mask, (0, 0), sigmaX=1.0)

    out = image.astype(np.float32)
    lift = (1 - factor) * 60
    out = out + dark_mask[:, :, np.newaxis] * lift
    out = out * factor + (1 - factor) * 180
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_staff_hue_shift(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Shift the hue of red staff lines."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Red pixels: hue near 0 or 180, high saturation
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    red_mask = ((hue < 15) | (hue > 165)) & (sat > 60)
    shift = rng.uniform(*config.staff_hue_range)
    hsv[:, :, 0] = np.where(red_mask, (hue + shift) % 180, hue)
    hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _apply_iron_gall(
    image: np.ndarray, _config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Simulate iron gall ink corrosion (brownish halo around dark areas)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Find dark regions (ink)
    _, ink_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Dilate to create halo
    kernel_size = rng.integers(3, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    halo = cv2.dilate(ink_mask, kernel, iterations=2)
    # Halo minus original ink = corrosion ring
    corrosion = cv2.subtract(halo, ink_mask).astype(np.float32) / 255.0

    opacity = rng.uniform(0.15, 0.4)
    corrosion_color = np.array([
        rng.uniform(100, 140),
        rng.uniform(70, 100),
        rng.uniform(30, 60),
    ], dtype=np.float32)

    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] = (
            out[:, :, c] * (1 - corrosion * opacity) + corrosion_color[c] * corrosion * opacity
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_salt_deposits(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add small white-ish salt/mineral deposit patches."""
    h, w = image.shape[:2]
    n = rng.integers(*config.salt_count_range)
    out = image.astype(np.float32)
    for _ in range(n):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry = rng.integers(3, 12)
        rx = rng.integers(3, 12)
        y0, y1 = max(0, cy - ry), min(h, cy + ry)
        x0, x1 = max(0, cx - rx), min(w, cx + rx)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
        mask = np.clip(1.0 - dist, 0, 1)
        opacity = rng.uniform(0.3, 0.7)
        brightness = rng.uniform(220, 255)
        for c in range(3):
            out[y0:y1, x0:x1, c] = (
                out[y0:y1, x0:x1, c] * (1 - mask * opacity) +
                brightness * mask * opacity
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_perspective(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply slight perspective warp simulating camera angle."""
    h, w = image.shape[:2]
    max_px = config.perspective_max_px
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.integers(-max_px, max_px + 1)
        dst[i, 1] += rng.integers(-max_px, max_px + 1)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def _apply_uneven_lighting(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply a gradient brightness overlay simulating uneven illumination."""
    h, w = image.shape[:2]
    direction = rng.choice(["lr", "rl", "tb", "bt", "radial"])
    lo, hi = config.lighting_intensity_range

    if direction == "lr":
        grad = np.linspace(lo, 1.0, w, dtype=np.float32)[np.newaxis, :]
    elif direction == "rl":
        grad = np.linspace(1.0, lo, w, dtype=np.float32)[np.newaxis, :]
    elif direction == "tb":
        grad = np.linspace(lo, 1.0, h, dtype=np.float32)[:, np.newaxis]
    elif direction == "bt":
        grad = np.linspace(1.0, lo, h, dtype=np.float32)[:, np.newaxis]
    else:  # radial
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2 + rng.uniform(-h * 0.2, h * 0.2), w / 2 + rng.uniform(-w * 0.2, w * 0.2)
        max_r = np.sqrt(h ** 2 + w ** 2) / 2
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max_r
        grad = (1.0 - dist * (1.0 - lo)).astype(np.float32)

    expanded = grad[:, :, np.newaxis] if grad.ndim == 2 else grad[:, :, np.newaxis]
    grad = np.broadcast_to(expanded, (h, w, 3))
    out = image.astype(np.float32) * grad
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_shadow(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Add a shadow from one edge (simulating a book spine or hand)."""
    h, w = image.shape[:2]
    edge = rng.choice(["left", "right", "top", "bottom"])
    shadow_width_frac = rng.uniform(0.1, 0.35)
    lo = rng.uniform(*config.shadow_intensity_range)

    if edge == "left":
        sw = int(w * shadow_width_frac)
        grad = np.ones(w, dtype=np.float32)
        grad[:sw] = np.linspace(lo, 1.0, sw)
        grad = grad[np.newaxis, :]
    elif edge == "right":
        sw = int(w * shadow_width_frac)
        grad = np.ones(w, dtype=np.float32)
        grad[-sw:] = np.linspace(1.0, lo, sw)
        grad = grad[np.newaxis, :]
    elif edge == "top":
        sw = int(h * shadow_width_frac)
        grad = np.ones(h, dtype=np.float32)
        grad[:sw] = np.linspace(lo, 1.0, sw)
        grad = grad[:, np.newaxis]
    else:
        sw = int(h * shadow_width_frac)
        grad = np.ones(h, dtype=np.float32)
        grad[-sw:] = np.linspace(1.0, lo, sw)
        grad = grad[:, np.newaxis]

    out = image.astype(np.float32) * grad[:, :, np.newaxis] if grad.ndim == 2 else \
          image.astype(np.float32) * np.broadcast_to(grad[:, :, np.newaxis], (h, w, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_barrel_distortion(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply barrel or pincushion distortion simulating lens effects."""
    h, w = image.shape[:2]
    k1 = rng.uniform(*config.barrel_k_range)
    cx, cy = w / 2, h / 2
    camera_matrix = np.array([
        [max(w, h), 0, cx],
        [0, max(w, h), cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.array([k1, 0, 0, 0], dtype=np.float64)
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted


def _apply_jpeg_compression(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Encode/decode as JPEG to introduce compression artifacts."""
    quality = int(rng.integers(*config.jpeg_quality_range))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


_TRANSFORMS: list[tuple[str, callable]] = [
    ("parchment_texture_prob", _apply_parchment_texture),
    ("aging_prob", _apply_aging),
    ("foxing_prob", _apply_foxing),
    ("water_stain_prob", _apply_water_stain),
    ("ink_fade_prob", _apply_ink_fade),
    ("staff_hue_prob", _apply_staff_hue_shift),
    ("iron_gall_prob", _apply_iron_gall),
    ("salt_deposit_prob", _apply_salt_deposits),
    ("perspective_prob", _apply_perspective),
    ("uneven_lighting_prob", _apply_uneven_lighting),
    ("shadow_prob", _apply_shadow),
    ("barrel_distortion_prob", _apply_barrel_distortion),
    ("jpeg_prob", _apply_jpeg_compression),
]


def augment(
    image: np.ndarray,
    config: AugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random augmentations to a clean rendered score image.

    Each of the 13 transforms is applied independently with its own
    probability from the config. On average ~5-6 transforms fire per call.

    Args:
        image: Clean rendered image (H, W, 3), uint8.
        config: Augmentation parameters. Defaults to AugmentationConfig().
        rng: Random number generator for reproducibility.

    Returns:
        Augmented image with same shape, uint8.
    """
    if config is None:
        config = AugmentationConfig()
    if rng is None:
        rng = np.random.default_rng()
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {image.dtype}")

    out = image.copy()
    for prob_attr, transform_fn in _TRANSFORMS:
        prob = getattr(config, prob_attr)
        if rng.random() < prob:
            out = transform_fn(out, config, rng)
    return out
