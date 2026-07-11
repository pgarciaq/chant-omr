# 0001. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#35](https://github.com/pgarciaq/chant-omr/issues/35)

## Context

chant-omr accumulated many design choices across PLAN.md sections, GitHub issue
threads, and commits (image sizing, tokenizer scope, parallel render, decoder
design, NABC deferral, etc.). PLAN.md describes *what* we build; issues track
*work* — neither reliably answers *why* we rejected alternatives.

## Decision

Adopt lightweight **Architecture Decision Records** under `docs/adr/` with a
template, index, and selective backfill of major forks. Reference ADRs from
PLAN.md and issues when a rationale matters.

## Consequences

### Positive

- Future contributors can find rationale without archaeology in chat logs.
- Clear split: PLAN = spec, issues = tasks, ADRs = why.

### Negative / trade-offs

- Extra maintenance when reversing a decision (must supersede ADR).
- Not every choice warrants an ADR — judgment required.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| PLAN.md only | Already long; mixes spec with historical rationale |
| Issue comments only | Hard to discover; closed issues bury context |
| ADR for every commit | Too heavy for a solo/small project |
