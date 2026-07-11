# 0002. Variable-height score strips and patch grid

- **Status:** Accepted
- **Date:** 2026-07-11
- **Issue:** [#8](https://github.com/pgarciaq/chant-omr/issues/8)

## Context

Gregorio nomargin renders are **wide score strips** (width ~1182 px, height
400–1600+ px), not full book pages. Transcoda uses a **fixed** 1485×1050 canvas
→ fixed 47×33 patch grid. We must choose resize policy and encoder patch layout.

## Decision

- Scale to **`target_width=1050`**, preserve aspect ratio.
- Cap height at **`max_height=1600`** (scale down longer chants uniformly).
- **No portrait letterboxing** (no fake book-page canvas).
- Encoder outputs a **variable** patch grid (32 columns at stride 32; rows vary).
- Batch collate bottom-pads images to max height in batch ([#32](https://github.com/pgarciaq/chant-omr/issues/32) wires encoder mask).

## Consequences

### Positive

- Matches nomargin strip geometry; no wasted empty staff on short chants.
- Compute scales with actual score size.

### Negative / trade-offs

- More complex than fixed canvas: per-sample grid size, 2D sin emb in #11, encoder padding masks.
- Differs from Transcoda's fixed grid — not apples-to-apples on architecture alone.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Fixed 1050×1485 portrait canvas | Letterbox or crop; retired ghh assumption |
| Fixed Transcoda 1485×1050 | Strips are not full pages; distorts or wastes patches |
| Content-area crop (ghh Stage 6) | Explicitly out of scope for OMR training/inference |
