"""Domain augmentation to make synthetic renders look like real photographs.

Bridges the domain gap between clean Gregorio renders and photographs of
historical manuscripts. Applied during training to create diverse, realistic
training examples from clean synthetic source images.

Augmentation categories:
1. Ink and staff appearance (red hue variation, bleeding, fading, thickness)
2. Parchment/paper substrate (texture, aging, foxing, water stains)
3. Photography artifacts (perspective, barrel distortion, lighting, shadows)
4. Degradation (iron gall corrosion, salt deposits, humidity damage)
5. Compression artifacts (JPEG quality variation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    """Controls augmentation intensity and probability."""

    # Ink appearance
    staff_hue_range: tuple[float, float] = (0.0, 15.0)  # red hue in degrees
    ink_bleed_prob: float = 0.3
    ink_fade_prob: float = 0.2
    thickness_variation: float = 0.15

    # Substrate
    parchment_texture_prob: float = 0.8
    foxing_prob: float = 0.3
    water_stain_prob: float = 0.15
    aging_intensity: tuple[float, float] = (0.0, 0.4)

    # Photography
    perspective_max_angle: float = 5.0
    barrel_distortion_range: tuple[float, float] = (-0.1, 0.1)
    uneven_lighting_prob: float = 0.5
    shadow_prob: float = 0.3
    flash_hotspot_prob: float = 0.1

    # Degradation
    iron_gall_prob: float = 0.1
    salt_deposit_prob: float = 0.05

    # Compression
    jpeg_quality_range: tuple[int, int] = (60, 95)
    jpeg_prob: float = 0.5


def augment(
    image: np.ndarray,
    config: AugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random augmentations to a clean rendered score image.

    Args:
        image: Clean rendered image (H, W, 3), uint8.
        config: Augmentation parameters. Defaults to AugmentationConfig().
        rng: Random number generator for reproducibility.

    Returns:
        Augmented image with same shape, uint8.
    """
    raise NotImplementedError("Augmentation not yet implemented")
