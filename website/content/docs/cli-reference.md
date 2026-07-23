---
title: "CLI Reference"
weight: 20
description: "Complete reference for the chant-omr command-line interface"
---

ChantOMR provides a single entry point: `chant-omr`. All subcommands are
listed below.

## `chant-omr download`

Download GABC files from GregoBase.

```bash
chant-omr download [--output-dir data/gregobase/]
```

Fetches GABC transcriptions from the GregoBase API. The download is
incremental — existing files are skipped. Uses a polite User-Agent header
and respects rate limits.

## `chant-omr render`

Render GABC files into score images.

```bash
chant-omr render [--gabc-dir data/gregobase/] [--output-dir data/rendered/] [--workers 4]
```

Invokes Gregorio + LuaLaTeX to render each GABC file into a PNG image.
Renders are deterministic. The `--workers` flag controls parallel rendering.

## `chant-omr train-tokenizer`

Train a BPE tokenizer on GABC bodies.

```bash
chant-omr train-tokenizer [--gabc-dir data/gregobase/] [--output-dir data/tokenizer/]
```

Strips GABC headers and trains a Byte Pair Encoding tokenizer on the
notation bodies. Produces `tokenizer.json`.

## `chant-omr cleanup`

Remove orphan files from the rendered directory.

```bash
chant-omr cleanup [--rendered-dir data/rendered/] [--no-dry-run]
```

Identifies and removes:
- Orphan `.gabc` files with no matching `.png`
- Orphan `.png` files with no matching `.gabc`
- Invalid GABC files that would be rejected during training

By default runs in dry-run mode (reports but does not delete).

## `chant-omr predict`

Run OMR on a single image.

```bash
# Using the HuggingFace model (recommended)
chant-omr predict IMAGE_PATH --model pgquiles/chant-omr [--device openvino]

# Using a local checkpoint
chant-omr predict IMAGE_PATH --checkpoint PATH [--device auto]
```

Loads the trained model and decodes the score image into GABC notation,
printed to stdout (or written to a file with `--output`).

Provide either `--model` (HuggingFace repo ID, auto-downloads) or
`--checkpoint` (local `.ckpt` file). These are mutually exclusive.

With `--model`, the CLI downloads the model on first use and selects the
appropriate format based on `--device`: OpenVINO IR for `openvino`,
ONNX for `onnx`, or safetensors for PyTorch backends.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | &mdash; | HuggingFace model repo (e.g. `pgquiles/chant-omr`); downloads automatically |
| `--checkpoint` | &mdash; | Path to a local `.ckpt` file (mutually exclusive with `--model`) |
| `--device` | `auto` | Inference backend: `auto`/`cuda`/`xpu`/`cpu` (PyTorch), `onnx`, or `openvino` |
| `--model-dir` | auto | Exported model directory (overrides `--model` subfolder for onnx/openvino) |
| `--beam-width` | from config | Beam search width (1 = greedy) |
| `--max-length` | from config | Maximum decoder sequence length |
| `--repetition-penalty` | from config | Repetition penalty for autoregressive decoding |
| `--grammar-constrained` | off | Enable balanced-parenthesis grammar mask during decoding |
| `--grammar-penalty` | `-inf` | Logit penalty for grammar-forbidden tokens. `-inf` = hard mask; a finite value like `-10.0` = soft penalty that can be overridden by strong model confidence |
| `--output` / `-o` | stdout | Write GABC to a file instead of stdout |
| `--name` | `OMR output` | GABC `name:` header value |
| `--dump-metrics` | off | Print teacher-forcing vs greedy diagnostics (uses sidecar `.gabc` when present) |

## `chant-omr evaluate`

Evaluate model on benchmark (image, GABC) pairs.

```bash
chant-omr evaluate CHECKPOINT [--benchmark-dir benchmarks/] \
    [--device auto] [--beam-width 3] [--max-length 8192] \
    [--test-split-only] [--limit N] \
    [--grammar-constrained] [--grammar-penalty FLOAT] [--gregorio-check]
```

Reports aggregate metrics: GABC Edit Distance, neume accuracy, structural
validity, and optionally Gregorio compilation success rate.

| Flag | Default | Description |
|------|---------|-------------|
| `CHECKPOINT` | required | Path to a `.ckpt` file (positional argument) |
| `--benchmark-dir` | auto-detect | Directory with paired `.png` + `.gabc` files (tries `benchmarks/`, then `data/rendered/`) |
| `--device` | `auto` | Inference device: `auto`/`cuda`/`xpu`/`cpu` |
| `--beam-width` | `3` | Beam search width |
| `--max-length` | `8192` | Maximum decoder sequence length |
| `--repetition-penalty` | `1.1` | Repetition penalty |
| `--test-split-only` | off | Only evaluate test-split samples (`catalog_id % 20 == 0`) |
| `--limit` | all | Evaluate only first N pairs |
| `--grammar-constrained` | off | Enable grammar-constrained decoding during evaluation |
| `--grammar-penalty` | from config | Logit penalty for grammar-forbidden tokens |
| `--gregorio-check` | off | Compile each prediction through Gregorio to check structural validity (requires `gregorio` binary) |

## `chant-omr export`

Export the model for deployment.

```bash
chant-omr export CHECKPOINT [--format onnx|openvino|safetensors] \
    [--output-dir models/] [--verify]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `onnx` | Export format (see table below) |
| `--output-dir` | `models/` | Output directory for exported artifacts |
| `--verify` | off | Run numeric parity check after export (PyTorch vs exported model) |

| Format | Artifacts | Description |
|--------|-----------|-------------|
| `onnx` | `encoder.onnx`, `decoder.onnx`, `decoder_init.onnx`, `decoder_step.onnx`, `tokenizer.json`, `manifest.json` | **Primary.** Portable inference on any hardware via ONNX Runtime (CUDA, DirectML, CoreML, CPU). Dual-path decoding: cached (init + step) for greedy, non-cached for beam search. |
| `openvino` | `encoder.xml/.bin`, `decoder.xml/.bin`, `decoder_init.xml/.bin`, `decoder_step.xml/.bin`, `manifest.json` | Intel-optimized path for Arc GPUs and NPUs. Same dual-path layout as ONNX. |
| `safetensors` | `model.safetensors`, `manifest.json` | Full weights for HuggingFace distribution |

The decoder uses **dual-path decoding**: greedy mode uses a KV-cached path
(`decoder_init` for the first step, `decoder_step` for subsequent steps)
with 4 stacked cache tensors `(n_layers, B, H, S, head_dim)`. Beam search
uses the non-cached `decoder` model that recomputes attention each step.

## `chant-omr audit-tokens`

Report token-length distribution over the rendered corpus.

```bash
chant-omr audit-tokens [--rendered-dir data/rendered/]
```

Useful for validating that the tokenizer and max sequence length
are configured correctly.

## `chant-omr train`

Train the OMR model (alternative to `python scripts/train.py`).

```bash
chant-omr train [--config configs/default.yaml] [--accelerator cuda]
```

## `chant-omr upload`

Upload model to HuggingFace Hub.

```bash
chant-omr upload CHECKPOINT --repo-id pgquiles/chant-omr [--config configs/default.yaml] [--private]
```

Exports the checkpoint in all formats (safetensors, OpenVINO IR, ONNX),
generates a model card, and uploads everything to HuggingFace Hub.
Requires prior authentication via `hf auth login`.

| Flag | Default | Description |
|------|---------|-------------|
| `CHECKPOINT` | required | Path to a `.ckpt` file (positional argument) |
| `--repo-id` | required | HuggingFace repo (e.g. `pgquiles/chant-omr`) |
| `--config` | `configs/default.yaml` | Model configuration file |
| `--private` | off | Create a private repository |

## `chant-omr manifest`

Manifest management commands for tracking dataset state.
