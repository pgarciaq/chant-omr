"""Generate augmented showcase images for the website from the hero score."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chant_omr.data.augmentation import AugmentationConfig, augment

HERO = Path("website/static/images/hero_score.jpg")
OUT_DIR = Path("website/static/images")

VARIANTS: dict[str, dict] = {
    "showcase_parchment.jpg": dict(
        parchment_texture_prob=1.0,
        parchment_blend_range=(0.35, 0.35),
        aging_prob=1.0,
        aging_intensity=(0.15, 0.15),
        foxing_prob=0.0,
        water_stain_prob=0.0,
        ink_fade_prob=0.0,
        staff_hue_prob=0.0,
        iron_gall_prob=0.0,
        salt_deposit_prob=0.0,
        perspective_prob=0.0,
        uneven_lighting_prob=0.0,
        shadow_prob=0.0,
        barrel_distortion_prob=0.0,
        jpeg_prob=0.0,
    ),
    "showcase_foxing.jpg": dict(
        parchment_texture_prob=1.0,
        parchment_blend_range=(0.25, 0.25),
        aging_prob=1.0,
        aging_intensity=(0.2, 0.2),
        foxing_prob=1.0,
        foxing_count_range=(20, 25),
        water_stain_prob=0.0,
        ink_fade_prob=1.0,
        ink_fade_range=(0.85, 0.85),
        staff_hue_prob=0.0,
        iron_gall_prob=0.0,
        salt_deposit_prob=0.0,
        perspective_prob=0.0,
        uneven_lighting_prob=0.0,
        shadow_prob=0.0,
        barrel_distortion_prob=0.0,
        jpeg_prob=0.0,
    ),
    "showcase_iron_gall.jpg": dict(
        parchment_texture_prob=1.0,
        parchment_blend_range=(0.3, 0.3),
        aging_prob=1.0,
        aging_intensity=(0.25, 0.25),
        foxing_prob=0.0,
        water_stain_prob=0.0,
        ink_fade_prob=1.0,
        ink_fade_range=(0.8, 0.8),
        staff_hue_prob=0.0,
        iron_gall_prob=1.0,
        salt_deposit_prob=0.0,
        perspective_prob=0.0,
        uneven_lighting_prob=0.0,
        shadow_prob=0.0,
        barrel_distortion_prob=0.0,
        jpeg_prob=0.0,
    ),
    "showcase_photography.jpg": dict(
        parchment_texture_prob=1.0,
        parchment_blend_range=(0.3, 0.3),
        aging_prob=1.0,
        aging_intensity=(0.15, 0.15),
        foxing_prob=0.0,
        water_stain_prob=0.0,
        ink_fade_prob=0.0,
        staff_hue_prob=0.0,
        iron_gall_prob=0.0,
        salt_deposit_prob=0.0,
        perspective_prob=1.0,
        perspective_max_px=20,
        uneven_lighting_prob=1.0,
        lighting_intensity_range=(0.75, 0.75),
        shadow_prob=1.0,
        shadow_intensity_range=(0.6, 0.6),
        barrel_distortion_prob=0.0,
        jpeg_prob=0.0,
    ),
    "showcase_heavily_aged.jpg": dict(
        parchment_texture_prob=1.0,
        parchment_blend_range=(0.4, 0.4),
        aging_prob=1.0,
        aging_intensity=(0.25, 0.25),
        foxing_prob=1.0,
        foxing_count_range=(28, 35),
        water_stain_prob=1.0,
        ink_fade_prob=1.0,
        ink_fade_range=(0.75, 0.75),
        staff_hue_prob=1.0,
        staff_hue_range=(10.0, 10.0),
        iron_gall_prob=1.0,
        salt_deposit_prob=1.0,
        salt_count_range=(8, 12),
        perspective_prob=1.0,
        perspective_max_px=15,
        uneven_lighting_prob=1.0,
        lighting_intensity_range=(0.8, 0.8),
        shadow_prob=1.0,
        shadow_intensity_range=(0.65, 0.65),
        barrel_distortion_prob=0.0,
        jpeg_prob=1.0,
        jpeg_quality_range=(72, 78),
    ),
}


def main() -> None:
    hero = cv2.imread(str(HERO))
    if hero is None:
        raise FileNotFoundError(f"Cannot read {HERO}")
    hero_rgb = cv2.cvtColor(hero, cv2.COLOR_BGR2RGB)

    rng = np.random.default_rng(42)

    for filename, overrides in VARIANTS.items():
        config = AugmentationConfig(**overrides)
        result = augment(hero_rgb, config=config, rng=rng)
        out_path = OUT_DIR / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  wrote {out_path}")

    print(f"\nGenerated {len(VARIANTS)} showcase images.")


if __name__ == "__main__":
    main()
