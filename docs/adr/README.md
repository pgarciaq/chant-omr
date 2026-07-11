# Architecture Decision Records

This directory holds **Architecture Decision Records (ADRs)** for chant-omr: durable
notes on *why* we chose one approach over another.

| Artifact | Role |
|----------|------|
| [PLAN.md](../PLAN.md) | What we are building (spec + implementation status) |
| [GitHub issues](https://github.com/pgarciaq/chant-omr/issues) | Actionable work items |
| **ADRs** (`docs/adr/`) | Rationale for major forks |

## When to write an ADR

Write an ADR when a decision:

- Involves meaningful trade-offs between alternatives
- Might not be obvious from code or PLAN.md alone
- Constrains future work or closes off other paths

Do **not** write an ADR for routine bugfixes, naming tweaks, or following an
already-documented pattern.

## Process

1. Copy [template.md](template.md) to `NNNN-short-title.md` (next number in index).
2. Fill in Context, Decision, Consequences, and link the GitHub issue/PR.
3. Add the ADR to the index below.
4. Reference the ADR from PLAN.md or issue comments when relevant.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-variable-height-patch-grid.md) | Variable-height score strips and patch grid | Accepted |
| [0003](0003-bpe-body-only-tokenizer.md) | BPE tokenizer on GABC bodies only | Accepted |
| [0004](0004-png-first-dataset-pairing.md) | PNG-first dataset pairing and catalog-id split | Accepted |
| [0005](0005-parallel-render-workers.md) | Parallel render workers and shared TeX cache | Accepted |
| [0006](0006-transcoda-decoder-architecture.md) | Transcoda-aligned Pre-LN decoder with RoPE | Accepted |
| [0007](0007-nabc-deferred-for-v0.md) | Defer NABC support for v0 | Accepted |
| [0008](0008-end-to-end-over-classical-omr.md) | End-to-end vision-encoder-decoder over classical OMR | Accepted |

## Template

See [template.md](template.md).
