# 0006. Transcoda-aligned Pre-LN decoder with RoPE

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#10](https://github.com/pgarciaq/chant-omr/issues/10)

## Context

The decoder generates GABC BPE tokens autoregressively from encoder patch memory.
Transcoda (59M, modern notation OMR) validated ConvNeXt-V2 + Transformer for this
pattern. We need layer norm placement, position encoding, and scope vs #11 projector.

## Decision

- **8-layer Pre-LN** decoder: `d_model=512`, `n_heads=8`, `d_ff=1024`, **GELU** FFN.
- **RoPE** on causal **self-attention** only.
- **Cross-attention every layer** to projected encoder memory `(B, N, 512)`.
- **2D sinusoidal** on encoder patches → **#11** (not #10).
- Optional `encoder_attention_mask` API in #10; collate wiring → [#32](https://github.com/pgarciaq/chant-omr/issues/32).
- **No weight tying** (embed ↔ lm_head) in v0.
- PyTorch **SDPA** for attention kernels.

## Consequences

### Positive

- Comparable inductive bias to Transcoda without copying weights.
- Pre-LN + RoPE are modern defaults for autoregressive decoders.

### Negative / trade-offs

- ~25–28M decoder params; full model ~59M ([#34](https://github.com/pgarciaq/chant-omr/issues/34)).
- Custom implementation to maintain vs `nn.TransformerDecoder` (no RoPE).

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Post-LN | Transcoda uses Pre-LN; training stability |
| Absolute sinusoidal on decoder | RoPE preferred for autoregressive LMs |
| 2D sin inside #10 | Split with projector per PLAN diagram |
| Weight tying | Deferred; simpler debugging |
