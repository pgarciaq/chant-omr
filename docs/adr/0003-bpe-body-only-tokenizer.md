# 0003. BPE tokenizer on GABC bodies only

- **Status:** Accepted
- **Date:** 2026-07-11
- **Issue:** [#7](https://github.com/pgarciaq/chant-omr/issues/7)

## Context

Full GABC files have headers (`name:`, `office-part:`, etc.) before the final
`%%` marker, then the notation body. Headers are metadata; bodies are what
Gregorio typesets and what the model must predict from the image.

## Decision

- Train ByteLevel BPE on **plain GABC bodies only** (after final `%%`).
- Exclude NABC notation and empty/short bodies (< 20 chars).
- Corpus: all manifest `ok` plain GABC under `data/gregobase/` (not limited to
  rendered PNGs).
- Vocab size **2048**; special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- At inference, headers are **prepended outside** the decoder output.

## Consequences

### Positive

- Vocabulary focuses on notation glyphs, not metadata keys.
- Larger text corpus improves BPE merges even before all pairs are rendered.

### Negative / trade-offs

- Model never learns headers; downstream must add them.
- NABC bodies excluded until Epic 5.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Tokenize full file including headers | Wastes vocab; headers not visible in score image |
| Character-level encoding | Longer sequences; BPE matches Transcoda pattern |
| Smaller vocab (1k) | 2048 headroom for GABC punctuation and merges |
