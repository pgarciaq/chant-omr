# 0013. GABC output assembly and header policy

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#13](https://github.com/pgarciaq/chant-omr/issues/13)

## Context

The BPE tokenizer trains and predicts **GABC bodies only** — text after the final
`%%` marker ([ADR 0003](0003-bpe-body-only-tokenizer.md), [PLAN.md §1.3](../../PLAN.md)).
Gregorio and ghh expect a **full `.gabc` file**: header fields, `%%`, then body.

Issue #13 must define how predicted token IDs become a valid on-disk GABC file for
v0 (chant-omr CLI) and v1 (ghh consumer, [#15](https://github.com/pgarciaq/chant-omr/issues/15)).

## Decision

### Model output scope (v0 and v1)

The vision-encoder-decoder **always predicts the body string** (neume groups in
parentheses + syllable text). It does **not** predict header fields (`name:`,
`mode:`, `annotation:`, etc.) in v0 or v1.

### v0 header assembly (chant-omr `#13`)

Wrap the decoded body in a **minimal valid GABC template**:

```gabc
name: OMR output;
%%
{decoded body}
```

- CLI may accept optional `--name "Kyrie XVII"` to override the default `name:` field.
- No `mode:` / `annotation:` unless explicitly passed via future CLI flags.
- Validation: assembled file passes `parse_gabc()` and `plain_gabc_reject_reason()`.

### v1 header assembly (ghh `#15`)

**ghh injects metadata**; the model still outputs body only.

- `name:` from book/project config, filename, or Stage 11 OCR incipit when available.
- Page number and catalog hints from ghh `pipeline.json` metadata when available.
- `mode:` / `annotation:` only when known from user config or catalog — not from the
  vision model unless a future issue adds header prediction.

chant-omr `#13` exposes a helper (e.g. `assemble_gabc(body, headers=...)`) so ghh
can supply headers without duplicating GABC formatting rules.

### Deferred

- **Grammar-constrained decoding** during beam search ([#37](https://github.com/pgarciaq/chant-omr/issues/37)).
- **HuggingFace model upload** (distribution only; no header change).
- Predicting headers from the image (possible v2+ if needed).

## Consequences

### Positive

- Matches training data (body-only BPE) — no train/serve skew on headers.
- v0 files are Gregorio-parseable with zero external metadata.
- v1 keeps a clean split: OMR = neumes + syllables; ghh = document context.

### Negative / trade-offs

- v0 output files lack liturgical metadata (mode, annotation) unless user adds them.
- ghh integration must implement header injection (#15 scope).

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Model predicts full file including headers | Not trained on headers; wastes capacity |
| Copy headers from sidecar `.gabc` at predict time | Cheats eval; unavailable for real scans |
| Body-only output (no `%%`) | Invalid GABC; breaks Gregorio and ghh |
