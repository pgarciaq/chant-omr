# 0011. Teacher-forcing cross-entropy training loss

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#12](https://github.com/pgarciaq/chant-omr/issues/12)

## Context

`ChantOMRDataset` returns full tokenized GABC bodies with `<bos>` / `<eos>`. The
model outputs next-token logits at every decoder position. Training (#12) must define
how batches become a loss without autoregressive inference (that is #13 beam search).

Shift-left teacher forcing was explicitly deferred from #8 dataset to the Lightning
module.

## Decision

- **Teacher forcing:** feed `input_ids[:, :-1]` to the decoder; predict `input_ids[:, 1:]`.
- **Loss:** token cross-entropy over flattened logits; `ignore_index=pad_id`.
- **Decoder attention mask:** `attention_mask[:, :-1]` aligned with decoder input.
- **Encoder mask:** `encoder_attention_mask` from collate ([#32](https://github.com/pgarciaq/chant-omr/issues/32)).
- **Optimizer:** AdamW `lr=1e-4`, `weight_decay=0.05`, cosine schedule with 5% linear warmup, `grad_clip=1.0` (Transcoda recipe, PLAN §3.1).
- **Overfit gate:** `--overfit-n 10` trains on a fixed small subset before cloud GPU spend.

## Consequences

### Positive

- Standard seq2seq training; easy to debug (loss should → ~0 on 10-sample overfit).
- Keeps dataset free of label-shift logic.

### Negative / trade-offs

- Exposure bias (train on gold prefix, infer autoregressively) — accepted; mitigated later by beam search / more data.
- Full fine-tune of ~59M params needs GPU for practical overfit timing.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Shift in dataset collate | Couples data layer to training; harder to reuse dataset |
| Per-token loss mask only (no ignore_index) | Redundant when pad positions use `pad_id` labels |
| Freeze encoder for overfit | Would not validate full pipeline |
| CTC / non-autoregressive | Wrong architecture for BPE decoder |
