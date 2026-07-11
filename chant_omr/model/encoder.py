"""Visual encoder: ConvNeXt-V2 pretrained on ImageNet.

Extracts a 2D grid of patch embeddings from score images. Feature maps have
stride 32; width 1050 yields 32 patch columns, while height varies with the
resized image (see dataset resize policy in PLAN §1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

DEFAULT_ENCODER_VARIANT = "convnextv2_tiny"
DEFAULT_OUTPUT_STRIDE = 32

# Short config aliases → full timm checkpoint names.
ENCODER_ALIASES: dict[str, str] = {
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "convnextv2_pico": "convnextv2_pico.fcmae_ft_in22k_in1k",
    "convnextv2_femto": "convnextv2_femto.fcmae_ft_in22k_in1k",
    "convnextv2_atto": "convnextv2_atto.fcmae_ft_in22k_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
}


@dataclass
class EncoderOutput:
    """Encoder forward pass outputs."""

    feature_map: torch.Tensor
    memory: torch.Tensor
    grid_size: tuple[int, int]

    @property
    def batch_size(self) -> int:
        return self.feature_map.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.feature_map.shape[1]

    @property
    def num_patches(self) -> int:
        height, width = self.grid_size
        return height * width


def resolve_encoder_name(variant: str) -> str:
    """Map config alias to a timm model name."""
    return ENCODER_ALIASES.get(variant, variant)


class ChantEncoder(nn.Module):
    """ConvNeXt-V2 backbone returning a flattened patch memory tensor."""

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        *,
        output_stride: int = DEFAULT_OUTPUT_STRIDE,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.output_stride = output_stride

    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput:
        """Encode ``(B, 3, H, W)`` images into patch features."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(f"expected input (B, 3, H, W), got {tuple(pixel_values.shape)}")

        feature_maps = self.backbone(pixel_values)
        if not feature_maps:
            raise RuntimeError("encoder backbone returned no feature maps")
        feature_map = feature_maps[0]
        batch, channels, height, width = feature_map.shape
        if channels != self.embed_dim:
            raise RuntimeError(f"expected {self.embed_dim} channels, got {channels}")

        memory = feature_map.flatten(2).transpose(1, 2).contiguous()
        if memory.shape != (batch, height * width, channels):
            raise RuntimeError(f"unexpected memory shape: {tuple(memory.shape)}")

        return EncoderOutput(
            feature_map=feature_map,
            memory=memory,
            grid_size=(height, width),
        )


def build_encoder(
    variant: str = DEFAULT_ENCODER_VARIANT,
    pretrained: bool = True,
    output_stride: int = DEFAULT_OUTPUT_STRIDE,
) -> tuple[ChantEncoder, int, int]:
    """Build the visual encoder.

    Args:
        variant: Config alias or full timm model name.
        pretrained: Load pretrained ImageNet weights when True.
        output_stride: Expected spatial reduction factor (32 for ConvNeXt-V2).

    Returns:
        ``(encoder, embed_dim, output_stride)``
    """
    timm_name = resolve_encoder_name(variant)
    backbone = timm.create_model(
        timm_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=(-1,),
    )
    channels = backbone.feature_info.channels()
    if not channels:
        raise RuntimeError(f"no feature channels reported for {timm_name}")
    embed_dim = channels[-1]
    encoder = ChantEncoder(backbone, embed_dim, output_stride=output_stride)
    return encoder, embed_dim, output_stride
