# 0004. PNG-first dataset pairing and catalog-id split

- **Status:** Accepted
- **Date:** 2026-07-11
- **Issue:** [#8](https://github.com/pgarciaq/chant-omr/issues/8)

## Context

Training needs aligned (image, GABC) pairs. The rendered directory may contain
legacy slug filenames, orphan PNGs, and multiple editorial variants per catalog
id (`5000.gabc`, `5000_elem1.gabc`).

## Decision

- **PNG-first indexing:** discover `*.png` in `data/rendered/`, require matching
  `{stem}.gabc` sidecar.
- Labels: GABC **body** from sidecar via `extract_gabc_body()`.
- **Train/val split by catalog id** (seeded shuffle) so variants of the same
  chant do not leak across splits.
- Keep **all SHA256-unique variants** as separate samples (editorial diversity).
- `collate_fn` pads images (bottom) and token sequences; teacher-forcing shift in #12.

## Consequences

### Positive

- Only train on pairs that actually rendered to PNG.
- Split integrity prevents optimistic val metrics.

### Negative / trade-offs

- Orphan slug `.gabc` without PNG excluded until [#29](https://github.com/pgarciaq/chant-omr/issues/29).
- Variant leakage within train split still possible (acceptable for v0).

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| GABC-first indexing | Includes non-rendered entries |
| Random file-level split | Leaks variants across train/val |
| One variant per id only | Throws away editorial diversity |
