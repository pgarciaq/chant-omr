# 0008. End-to-end vision-encoder-decoder over classical OMR

- **Status:** Accepted
- **Date:** 2026-07-10
- **Issue:** [#2](https://github.com/pgarciaq/chant-omr/issues/2)

## Context

OMR can be built as a **classical pipeline** (staff removal → segment → classify →
assemble) or an **end-to-end** image-to-sequence model. Existing tools (Audiveris,
Kraken neume mode) follow classical or non-GABC paths.

## Decision

Build **chant-omr** as an end-to-end **ConvNeXt-V2 + Transformer** model (~59M)
that maps score images directly to **GABC** token sequences, following Transcoda's
proven pattern adapted for square notation.

Training data: synthetic Gregorio renders + optional ghh augmentation (#30).
Inference: beam search → GABC body + header assembly → ghh consumer (#15).

## Consequences

### Positive

- Only need (image, GABC) pairs — no bounding boxes or symbol catalogs.
- Ligatures and spacing learned implicitly.
- Single model to export (OpenVINO) for Intel deployment.

### Negative / trade-offs

- Needs more paired data than rule-based systems for coverage.
- Domain gap from clean renders to parchment (#30 augmentation).
- Less interpretable failures than staged pipelines.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Classical neume classifier + assembler | Error cascade; heavy manual rules |
| Fine-tune LEGATO (943M) | Frozen vision encoder; no square notation; too large for OpenVINO |
| Use Transcoda weights directly | Wrong notation and `**kern` output |
| Kraken line OCR on neumes | Not structured GABC output |
