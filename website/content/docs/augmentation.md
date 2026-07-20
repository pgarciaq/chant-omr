---
title: "Data Augmentation"
weight: 35
description: "15 domain transforms that bridge the gap between clean renders and real manuscripts"
---

ChantOMR is trained on clean, computer-rendered score images (Gregorio +
LuaLaTeX). Real manuscripts look very different: aged parchment, faded ink,
foxing spots, uneven lighting, perspective distortion from photographing a
bound book. Domain augmentation bridges this gap by transforming clean renders
on-the-fly during training so the model learns to handle real-world conditions.

## How It Works

Each training image passes through 15 independent transforms, each with its
own probability. On average, ~6-7 transforms fire per image. The transforms
are applied using OpenCV and NumPy, with a seeded random generator for
reproducibility.

The augmentation pipeline runs **on-the-fly** during training (not as a
preprocessing step), so each epoch sees different augmented versions of the
same clean source images. This effectively multiplies the training data
diversity without storing augmented copies on disk.

## Transform Catalog

### Substrate (simulating aged parchment)

| Transform | Probability | Description |
|-----------|-------------|-------------|
| **Parchment texture** | 80% | Blends one of 34 real parchment texture patches onto the image background. Stronger on light areas (background), weaker on dark areas (ink) |
| **Aging tint** | 60% | Adds a warm yellowish tint simulating centuries of oxidation |
| **Foxing** | 30% | Scatters small brownish spots simulating fungal damage |
| **Water stains** | 15% | Adds a large semi-transparent elliptical blob with irregular edges |

### Ink appearance

| Transform | Probability | Description |
|-----------|-------------|-------------|
| **Ink fade** | 20% | Reduces contrast and lightens dark pixels, simulating aged ink |
| **Staff hue shift** | 30% | Shifts the hue of red staff lines, simulating different ink batches or aging |
| **Ink bleeding** | 15% | Oriented morphological dilation simulating ink spreading along parchment fibers |
| **Ink thickness variation** | 15% | Gradient-weighted erosion/dilation simulating varying quill pressure across the page |

### Degradation

| Transform | Probability | Description |
|-----------|-------------|-------------|
| **Iron gall corrosion** | 10% | Adds brownish halos around dark ink regions, simulating iron gall ink eating through parchment |
| **Salt deposits** | 5% | Adds small white-ish mineral deposit patches |

### Photography (simulating camera artifacts)

| Transform | Probability | Description |
|-----------|-------------|-------------|
| **Perspective skew** | 30% | Applies a slight perspective warp simulating a non-perpendicular camera angle |
| **Uneven lighting** | 50% | Applies a brightness gradient (left-right, top-bottom, or radial) simulating uneven illumination |
| **Shadow** | 30% | Adds a shadow gradient from one edge, simulating a book spine or photographer's hand |
| **Barrel distortion** | 15% | Applies barrel or pincushion lens distortion |

### Compression

| Transform | Probability | Description |
|-----------|-------------|-------------|
| **JPEG artifacts** | 50% | Encodes and decodes as JPEG at a random quality level (60-95), introducing compression artifacts |

## Parchment Texture Patches

The parchment texture transform uses 34 real photographs of parchment
surfaces, cropped and stored as JPEG patches in `data/textures/`. These were
photographed from actual historical manuscripts and books, providing authentic
surface textures that no procedural generator can replicate.

The texture is blended more strongly on lighter pixels (background) and more
weakly on darker pixels (ink), preserving readability while giving the image
a convincing aged appearance.

## Configuration

All probabilities and intensity ranges are configurable via
`AugmentationConfig` in `chant_omr/data/augmentation.py`. The defaults are
tuned for a good balance between diversity and readability.

To disable augmentation entirely, set `augment: false` in
`configs/default.yaml`. Augmentation only applies to the training split;
the validation split always uses clean images.

## Performance Notes

The 15 transforms are CPU-bound (OpenCV/NumPy). With `num_workers: 0` in
the DataLoader, augmentation runs in the main process and can cause ~4x
slower epoch times. Setting `num_workers: 4` (or higher on bare metal)
allows parallel augmentation across CPU cores.

In containerized environments (e.g., QuickPod), the default `/dev/shm` is
too small for PyTorch multiprocessing. See the
[Training Guide]({{< relref "training-guide" >}}) for how to create a
custom template with `--shm-size=2g`.
