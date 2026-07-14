"""Autoregressive Transformer decoder for GABC token generation.

Takes projected encoder patch memory and generates GABC tokens left-to-right using
causal self-attention (RoPE) plus cross-attention to encoder features.

Configuration follows Transcoda's design:
    - 8 layers, d_model=512, d_ff=1024, 8 heads (default)
    - Pre-LN, GELU feed-forward
    - RoPE on decoder self-attention
    - 2D sinusoidal positional encoding on encoder features lives in #11 projector

KV cache (#44): during autoregressive inference each layer can reuse previously
computed key/value projections. See :class:`LayerCache` and the ``use_cache``
/ ``past_key_values`` parameters on :class:`ChantDecoder`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCache(NamedTuple):
    """Cached key/value tensors for one decoder layer.

    All tensors are ``(B, H, S, D)`` where ``S`` is the accumulated sequence
    length (self-attention) or the fixed encoder length (cross-attention).
    """

    self_k: torch.Tensor
    self_v: torch.Tensor
    cross_k: torch.Tensor
    cross_v: torch.Tensor


KVCache = list[LayerCache]
"""Per-layer cache: ``past_key_values[i]`` holds :class:`LayerCache` for layer *i*."""


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
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be >= d_model")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DecoderConfig:
        """Build config from a YAML ``model:`` section."""
        return cls(
            d_model=int(mapping.get("d_model", 512)),
            n_layers=int(mapping.get("n_layers", 8)),
            n_heads=int(mapping.get("n_heads", 8)),
            d_ff=int(mapping.get("d_ff", 1024)),
            dropout=float(mapping.get("dropout", 0.1)),
            max_seq_len=int(mapping.get("max_seq_len", 2048)),
            vocab_size=int(mapping.get("vocab_size", 2048)),
            rope_theta=float(mapping.get("rope_theta", 10_000.0)),
        )


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for causal self-attention."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._cached_len and self._cos is not None and self._cos.device == device:
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype=dtype)
        self._sin = emb.sin().to(dtype=dtype)
        self._cached_len = seq_len

    def forward(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._build_cache(min(seq_len, self.max_seq_len), device, dtype)
        assert self._cos is not None and self._sin is not None
        return self._cos[:seq_len], self._sin[:seq_len]


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors shaped ``(B, H, T, D)``."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_self_attention_mask(
    seq_len: int,
    *,
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a bool SDPA mask ``(B, 1, T, T)`` where True allows attention."""
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    if attention_mask is None:
        return causal.unsqueeze(0).unsqueeze(0)

    valid = attention_mask.to(device=device, dtype=torch.bool)
    if valid.shape[-1] != seq_len:
        raise ValueError("attention_mask length must match sequence length")
    key_valid = valid.unsqueeze(1).unsqueeze(2)
    query_valid = valid.unsqueeze(1).unsqueeze(3)
    pad_mask = key_valid & query_valid
    return causal.unsqueeze(0).unsqueeze(0) & pad_mask


def build_cross_attention_mask(
    query_len: int,
    key_len: int,
    *,
    encoder_attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build bool cross-attention mask ``(B, 1, T, N)``."""
    valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
    if valid.shape[-1] != key_len:
        raise ValueError("encoder_attention_mask length must match encoder sequence length")
    return valid.unsqueeze(1).unsqueeze(2).expand(-1, 1, query_len, -1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    When *past_kv* is provided, only the new (last) token positions are
    projected; the cached K/V from previous steps are concatenated before
    the attention computation.  RoPE is applied at the correct absolute
    position via *start_pos*.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, theta=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        if past_kv is not None:
            start_pos = past_kv[0].shape[2]
        else:
            start_pos = 0

        total_len = start_pos + seq_len
        cos, sin = self.rope(total_len, device=hidden_states.device, dtype=hidden_states.dtype)
        cos = cos[start_pos:total_len]
        sin = sin[start_pos:total_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if use_cache:
            new_kv = (k, v)

        full_len = k.shape[2]

        if past_kv is not None:
            attn_mask = None
        else:
            attn_mask = build_self_attention_mask(
                full_len,
                attention_mask=attention_mask,
                device=hidden_states.device,
            )
            if attn_mask is not None and attn_mask.shape[0] == 1 and batch > 1:
                attn_mask = attn_mask.expand(batch, -1, -1, -1)

        dropout_p = self.dropout.p if self.training else 0.0
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(context), new_kv


class CrossAttention(nn.Module):
    """Multi-head cross-attention from decoder tokens to encoder memory.

    When *past_kv* is provided the encoder K/V projections are reused from
    the cache instead of being recomputed.  Since encoder memory is constant
    across generation steps this is always safe.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        if past_kv is not None:
            k, v = past_kv
        else:
            _, enc_len, _ = encoder_memory.shape
            k = (
                self.k_proj(encoder_memory)
                .view(batch, enc_len, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(encoder_memory)
                .view(batch, enc_len, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )

        new_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        if use_cache:
            new_kv = (k, v)

        enc_len = k.shape[2]
        attn_mask = None
        if encoder_attention_mask is not None:
            attn_mask = build_cross_attention_mask(
                seq_len,
                enc_len,
                encoder_attention_mask=encoder_attention_mask,
                device=hidden_states.device,
            )

        dropout_p = self.dropout.p if self.training else 0.0
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(context), new_kv


class FeedForward(nn.Module):
    """Pre-LN feed-forward block with GELU."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class DecoderLayer(nn.Module):
    """One Pre-LN decoder layer: self-attn → cross-attn → FFN."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        self.cross_attn_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.self_attn = CausalSelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_memory: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        layer_cache: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        sa_past = None if layer_cache is None else (layer_cache.self_k, layer_cache.self_v)
        xa_past = None if layer_cache is None else (layer_cache.cross_k, layer_cache.cross_v)

        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        sa_out, sa_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_kv=sa_past,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(sa_out)

        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        xa_out, xa_kv = self.cross_attn(
            hidden_states,
            encoder_memory,
            encoder_attention_mask=encoder_attention_mask,
            past_kv=xa_past,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(xa_out)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.dropout(self.ffn(hidden_states))

        new_cache: LayerCache | None = None
        if use_cache and sa_kv is not None and xa_kv is not None:
            new_cache = LayerCache(
                self_k=sa_kv[0], self_v=sa_kv[1],
                cross_k=xa_kv[0], cross_v=xa_kv[1],
            )
        return hidden_states, new_cache


class ChantDecoder(nn.Module):
    """Autoregressive Transformer decoder for GABC BPE tokens."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / (module.num_embeddings + module.embedding_dim)),
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Return next-token logits ``(B, T, vocab_size)``.

        When *use_cache* is ``True``, also returns a :data:`KVCache` list that
        should be passed back as *past_key_values* on the next call.  During
        training, leave both parameters at their defaults for unchanged
        behavior (the second return value will be ``None``).
        """
        if input_ids.ndim != 2:
            raise ValueError(f"expected input_ids (B, T), got {tuple(input_ids.shape)}")
        if encoder_memory.ndim != 3:
            raise ValueError(
                f"expected encoder_memory (B, N, D), got {tuple(encoder_memory.shape)}"
            )
        if encoder_memory.shape[-1] != self.config.d_model:
            raise ValueError(
                f"encoder_memory last dim must be d_model={self.config.d_model}, "
                f"got {encoder_memory.shape[-1]}"
            )
        if input_ids.shape[0] != encoder_memory.shape[0]:
            raise ValueError("batch size mismatch between input_ids and encoder_memory")

        seq_len = input_ids.shape[1]
        total_len = seq_len
        if past_key_values is not None and len(past_key_values) > 0:
            total_len += past_key_values[0].self_k.shape[2]
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"total sequence length {total_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        hidden_states = self.embed_tokens(input_ids)
        new_caches: KVCache = []
        for i, layer in enumerate(self.layers):
            lc = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_cache = layer(
                hidden_states,
                encoder_memory,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                layer_cache=lc,
                use_cache=use_cache,
            )
            if layer_cache is not None:
                new_caches.append(layer_cache)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_caches if use_cache else None


def build_decoder(config: DecoderConfig | None = None) -> ChantDecoder:
    """Build the Transformer decoder."""
    return ChantDecoder(config or DecoderConfig())
