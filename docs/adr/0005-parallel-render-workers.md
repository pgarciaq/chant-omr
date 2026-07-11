# 0005. Parallel render workers and shared TeX cache

- **Status:** Accepted
- **Date:** 2026-07-11
- **Issue:** [#31](https://github.com/pgarciaq/chant-omr/issues/31)

## Context

Bulk `chant-omr render` took ~13–15 s/score sequentially (`--workers 1`). Each
score runs a full cold LuaLaTeX + Gregorio autocompile; CPU time dominates.

## Decision

- Default **`--workers 0` (auto)** → `min(cpu_count, 8)`; cap via
  `CHANT_OMR_RENDER_WORKERS_MAX`.
- **`ProcessPoolExecutor`** for parallel LuaLaTeX (not threads).
- Shared **`TEXMFCACHE`** under `data/rendered/.texcache/`.
- Writable per-job cache fallback for standalone renders / CI.
- **`fcntl`** lock on parallel `render_failures.jsonl` appends.

Observed bulk throughput on 22-core host with warm cache: **~1.2 s/score effective**
(vs ~13 s sequential).

## Consequences

### Positive

- ~10× throughput improvement for corpus builds.
- No change to per-job compile correctness.

### Negative / trade-offs

- First batch still slow (cold start); RAM/disk contention if cap too high.
- Per-score CPU time unchanged — parallelism improves throughput only.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Docs-only (`--workers 8`) | Undiscoverable; no shared cache |
| Reuse temp dir without cleanup | Benchmarked; no gain |
| Persistent LuaLaTeX daemon | Large new subsystem; deferred |
| Thread pool | LuaLaTeX not thread-safe in one process |
