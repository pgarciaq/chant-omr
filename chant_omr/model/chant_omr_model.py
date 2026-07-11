"""Full vision-encoder-decoder model for Gregorian chant OMR.

Combines:
    1. ConvNeXt-V2 encoder (image → patch embeddings)
    2. MLP projector (encoder dim → decoder dim)
    3. Transformer decoder (patch embeddings → GABC tokens)

The model takes a score image and autoregressively generates GABC notation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChantOMRConfig:
    """Full model configuration."""

    # Encoder
    encoder_variant: str = "convnextv2_tiny"
    encoder_pretrained: bool = True

    # Projector
    projector_hidden: int = 768

    # Decoder
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    vocab_size: int = 2048

    # Input (resize policy — not fixed canvas size)
    image_width: int = 1050
    max_height: int = 1600


def build_model(config: ChantOMRConfig | None = None):
    """Build the complete ChantOMR model.

    Args:
        config: Model configuration.

    Returns:
        ChantOMR model.
    """
    raise NotImplementedError("Model not yet implemented")
