"""Autoregressive Transformer decoder for GABC token generation.

Takes the 2D patch grid from the encoder and generates GABC tokens
left-to-right using causal self-attention + cross-attention to encoder
features.

Configuration follows Transcoda's design:
    - 8 layers, d_model=512, d_ff=1024, 8 heads (default)
    - RoPE positional encoding on decoder
    - 2D sinusoidal positional encoding on encoder features
    - BPE vocabulary over GABC tokens (~1000-3000 tokens)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Transformer decoder hyperparameters."""

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    vocab_size: int = 2048


def build_decoder(config: DecoderConfig | None = None):
    """Build the Transformer decoder.

    Args:
        config: Decoder hyperparameters.

    Returns:
        Decoder model.
    """
    raise NotImplementedError("Decoder not yet implemented")
