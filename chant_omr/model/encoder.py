"""Visual encoder: ConvNeXt-V2 pretrained on ImageNet.

Extracts a 2D grid of patch embeddings from the input score image.
The encoder is initialized from pretrained ImageNet weights and fine-tuned
during training.

Architecture options (ordered by size):
    - ConvNeXt-V2 Atto:  3.7M params, dim=40   (fastest, lowest quality)
    - ConvNeXt-V2 Femto: 5.2M params, dim=48
    - ConvNeXt-V2 Pico:  9.1M params, dim=64
    - ConvNeXt-V2 Nano: 15.6M params, dim=80
    - ConvNeXt-V2 Tiny: 28.6M params, dim=96   (Transcoda's choice)
    - ConvNeXt-V2 Base: 88.7M params, dim=128  (overkill for chant)

For square notation (~30 neume types vs hundreds in modern notation),
Nano or Pico should suffice. Start with Tiny for comparability with
Transcoda, then ablate downward.
"""

from __future__ import annotations


def build_encoder(
    variant: str = "convnextv2_tiny",
    pretrained: bool = True,
    output_stride: int = 32,
) -> tuple:
    """Build the visual encoder.

    Args:
        variant: timm model name (e.g., "convnextv2_tiny.fcmae_ft_in22k_in1k").
        pretrained: Load pretrained ImageNet weights.
        output_stride: Spatial reduction factor.

    Returns:
        (encoder_model, embed_dim, patch_grid_size)
    """
    raise NotImplementedError("Encoder not yet implemented")
