# 0010. Encoder padding mask in dataset collate

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#32](https://github.com/pgarciaq/chant-omr/issues/32)

## Context

Batch collation bottom-pads score images to the tallest (and widest) item. The
ConvNeXt encoder produces patch features over the **full padded canvas**, including
white empty rows. Without a mask, decoder cross-attention treats padded patches as
real score content ([ADR 0002](0002-variable-height-patch-grid.md),
[ADR 0009](0009-mlp-projector-and-2d-sinusoidal-bridge.md)).

The decoder and `ChantOMR` already accept optional `encoder_attention_mask`; #32
wires it from the data pipeline.

## Decision

- Record **pre-pad** `image_height` / `image_width` in `ChantOMRDataset.__getitem__`.
- In `collate_chant_omr_batch`, compute patch grid from **padded** batch dimensions
  using shared `patch_grid_size(h, w, stride=32)` (matches encoder output).
- Emit `encoder_attention_mask` `(B, H'×W')` with `1` for patches from valid
  `(h_orig // 32) × (w_orig // 32)` region, `0` for bottom/right pad patches.
- Pass mask through Lightning #12 into `ChantOMR.forward()`.

## Consequences

### Positive

- Correct cross-attention when batching mixed-height strips (`batch_size > 1`).
- Single stride helper avoids drift between collate and encoder tests.

### Negative / trade-offs

- Slightly larger batch dict; mask computation each collate step.
- Assumes encoder stride 32 stays aligned with ConvNeXt-V2 (documented in encoder).

### Inference (single image)

At predict time there is **no batch padding**: one resized image, all encoder patches
are valid. `encoder_attention_mask` may be omitted or all-ones; #32 mask logic applies
only when batching mixed heights in training ([ADR 0012](0012-openvino-export-and-inference-deployment.md)
fixed-height pad at export is separate).

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Compute mask in Lightning from pixel tensor | Duplicates stride math; collate already knows pre-pad sizes |
| Skip mask; use `batch_size=1` only | Works for smoke test but wrong default for training |
| Height bucketing instead of pad+mask | Deferred optimization; mask is simpler v0 fix |
