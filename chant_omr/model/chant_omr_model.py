"""Full vision-encoder-decoder model for Gregorian chant OMR.

Combines:
    1. ConvNeXt-V2 encoder (image → patch embeddings)
    2. 2D sinusoidal positional encoding on the patch grid
    3. MLP projector (encoder dim → decoder dim)
    4. Transformer decoder (patch embeddings → GABC tokens)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from chant_omr.model.decoder import ChantDecoder, DecoderConfig, build_decoder, count_parameters
from chant_omr.model.encoder import ChantEncoder, build_encoder


@dataclass
class ChantOMRConfig:
    """Full model configuration."""

    encoder_variant: str = "convnextv2_tiny"
    encoder_pretrained: bool = True
    projector_hidden: int = 768
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 8192
    vocab_size: int = 2048
    image_width: int = 1050
    max_height: int = 1600

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ChantOMRConfig:
        """Build config from a YAML ``model:`` section."""
        return cls(
            encoder_variant=str(mapping.get("encoder_variant", "convnextv2_tiny")),
            encoder_pretrained=bool(mapping.get("encoder_pretrained", True)),
            projector_hidden=int(mapping.get("projector_hidden", 768)),
            d_model=int(mapping.get("d_model", 512)),
            n_layers=int(mapping.get("n_layers", 8)),
            n_heads=int(mapping.get("n_heads", 8)),
            d_ff=int(mapping.get("d_ff", 1024)),
            dropout=float(mapping.get("dropout", 0.1)),
            max_seq_len=int(mapping.get("max_seq_len", 8192)),
            vocab_size=int(mapping.get("vocab_size", 2048)),
            image_width=int(mapping.get("image_width", 1050)),
            max_height=int(mapping.get("max_height", 1600)),
        )


@dataclass
class ParameterBreakdown:
    """Trainable parameter counts by submodule."""

    encoder: int
    positional_encoding: int
    projector: int
    decoder: int

    @property
    def total(self) -> int:
        return self.encoder + self.positional_encoding + self.projector + self.decoder


def count_model_parameters(model: ChantOMR) -> ParameterBreakdown:
    """Count trainable parameters for each major submodule."""
    return ParameterBreakdown(
        encoder=count_parameters(model.encoder),
        positional_encoding=count_parameters(model.positional_encoding),
        projector=count_parameters(model.projector),
        decoder=count_parameters(model.decoder),
    )


def _sinusoidal_1d(
    length: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return ``(length, dim)`` sinusoidal table."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal dim must be even")
    position = torch.arange(length, device=device, dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / dim)
    )
    table = torch.zeros(length, dim, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
    table[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
    return table.to(dtype=dtype)


class Sinusoidal2DPositionalEncoding(nn.Module):
    """Fixed 2D sin/cos positional encoding added to encoder patch grids."""

    def __init__(self, embed_dim: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for 2D sinusoidal encoding")
        self.embed_dim = embed_dim
        self.dim_height = embed_dim // 2
        self.dim_width = embed_dim - self.dim_height

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``(B, C, H, W)`` patch features."""
        if feature_map.ndim != 4:
            raise ValueError(f"expected feature_map (B, C, H, W), got {tuple(feature_map.shape)}")
        _batch, channels, height, width = feature_map.shape
        if channels != self.embed_dim:
            raise ValueError(f"expected {self.embed_dim} channels, got {channels}")

        device = feature_map.device
        dtype = feature_map.dtype
        pe_height = _sinusoidal_1d(height, self.dim_height, device=device, dtype=dtype)
        pe_width = _sinusoidal_1d(width, self.dim_width, device=device, dtype=dtype)
        pe = torch.cat(
            [
                pe_height[:, None, :].expand(height, width, self.dim_height),
                pe_width[None, :, :].expand(height, width, self.dim_width),
            ],
            dim=-1,
        )
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        return feature_map + pe


class MLPProjector(nn.Module):
    """Two-layer MLP bridge from encoder dim to decoder ``d_model``."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.net(memory)


class ChantOMR(nn.Module):
    """End-to-end score image → GABC token logits."""

    def __init__(
        self,
        encoder: ChantEncoder,
        positional_encoding: Sinusoidal2DPositionalEncoding,
        projector: MLPProjector,
        decoder: ChantDecoder,
        config: ChantOMRConfig,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.positional_encoding = positional_encoding
        self.projector = projector
        self.decoder = decoder

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return projected encoder memory ``(B, N, d_model)``."""
        encoder_output = self.encoder(pixel_values)
        positioned = self.positional_encoding(encoder_output.feature_map)
        memory = positioned.flatten(2).transpose(1, 2).contiguous()
        projected = self.projector(memory)
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape != (projected.shape[0], projected.shape[1]):
                raise ValueError("encoder_attention_mask must be (B, num_patches)")
        return projected

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-token logits ``(B, T, vocab_size)``."""
        memory = self.encode(pixel_values, encoder_attention_mask=encoder_attention_mask)
        logits, _ = self.decoder(
            input_ids,
            memory,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return logits


def build_model(
    config: ChantOMRConfig | None = None,
    *,
    encoder_pretrained: bool | None = None,
) -> ChantOMR:
    """Build the complete ChantOMR model."""
    cfg = config or ChantOMRConfig()
    pretrained = cfg.encoder_pretrained if encoder_pretrained is None else encoder_pretrained

    encoder, embed_dim, _stride = build_encoder(
        variant=cfg.encoder_variant,
        pretrained=pretrained,
    )
    positional_encoding = Sinusoidal2DPositionalEncoding(embed_dim)
    projector = MLPProjector(embed_dim, cfg.projector_hidden, cfg.d_model)
    decoder = build_decoder(DecoderConfig.from_mapping(asdict(cfg)))

    return ChantOMR(
        encoder=encoder,
        positional_encoding=positional_encoding,
        projector=projector,
        decoder=decoder,
        config=cfg,
    )
