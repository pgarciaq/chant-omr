# 0009. MLP projector and 2D sinusoidal bridge

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#11](https://github.com/pgarciaq/chant-omr/issues/11)

## Context

The encoder (#9) outputs 768-dim ConvNeXt-V2 patch features on a variable `H'×32`
grid. The decoder (#10) expects 512-dim memory for cross-attention. Transcoda
inserts **2D sinusoidal positional encoding** on the visual grid, then an **MLP
projector**, before the autoregressive decoder.

## Decision

- **`Sinusoidal2DPositionalEncoding`:** add fixed sin/cos features per patch
  `(row, col)` on the **768-dim feature map** before flattening. Split embed dim:
  half for row, half for column (standard 2D extension of Vaswani PE).
- **`MLPProjector`:** `Linear(768, h) → GELU → Linear(h, 512)` with default
  `h=768` (Transcoda-style two-layer bridge).
- **Order:** encoder → 2D sin → flatten → MLP → decoder (not project then add PE).
- **`ChantOMR.encode()`** returns projected memory; `forward()` adds decoder pass.
- Optional `encoder_attention_mask` forwarded to decoder ([#32](https://github.com/pgarciaq/chant-omr/issues/32) collate wiring deferred).
- Target **~59M** total parameters ([#34](https://github.com/pgarciaq/chant-omr/issues/34)).

Complements [ADR 0006](0006-transcoda-decoder-architecture.md) (decoder-side RoPE).

## Consequences

### Positive

- Spatial inductive bias before cross-attention: patches carry explicit grid position.
- Variable-height grids supported by computing sin/cos tables from live `(H', W')`.
- Parameter-free PE; bridge capacity matches Transcoda without widening decoder.

### Negative / trade-offs

- Per-batch patch count still varies with height; padding masks needed for training ([#32](https://github.com/pgarciaq/chant-omr/issues/32)).
- 2D sin assumes roughly axis-aligned strips (true for nomargin Gregorio output).

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| No encoder PE | Weak alignment between image regions and GABC tokens |
| 1D sin after flatten | Loses separate row vs column structure |
| Learned absolute patch embeddings | Poor extrapolation to unseen grid heights |
| RoPE on patches | RoPE suits autoregressive **token** sequences, not 2D fields |
| Single linear 768→512 | Less bridge capacity than Transcoda |
| Project before adding PE | Transcoda adds PE in encoder feature space first |
| Coord-conv in backbone | Would modify pretrained ConvNeXt input channels |
