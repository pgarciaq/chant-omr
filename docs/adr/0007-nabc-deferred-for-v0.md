# 0007. Defer NABC support for v0

- **Status:** Accepted
- **Date:** 2026-07-11
- **Issue:** [#22](https://github.com/pgarciaq/chant-omr/issues/22)

## Context

GregoBase includes chants in **NABC** (Nuances Artificielles) notation, a
different encoding from plain GABC. NABC requires different rendering, parsing,
and training paths.

## Decision

- **v0 trains on plain GABC only.** Skip NABC at download (#21), render (#21), and
  dataset time via `is_nabc_notation()` / `plain_gabc_reject_reason()`.
- Track full NABC work under **Epic 5** ([#23](https://github.com/pgarciaq/chant-omr/issues/23)–[#26](https://github.com/pgarciaq/chant-omr/issues/26)).
- Prefer fetching plain twins when available ([#26](https://github.com/pgarciaq/chant-omr/issues/26)).

## Consequences

### Positive

- Smaller scope for first trainable model; plain GABC is ghh's target format.
- Avoids blocking v0 on NABC renderer research.

### Negative / trade-offs

- Corpus excludes NABC-only entries until Epic 5.
- Collapse-NABC-to-plain ([#24](https://github.com/pgarciaq/chant-omr/issues/24)) only as last resort.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Train on NABC from day one | Unbounded scope; no NABC renderer yet |
| Collapse all NABC to plain | Lossy; only when plain unavailable |
| Ignore NABC silently | Explicit skip + manifest counters instead |
