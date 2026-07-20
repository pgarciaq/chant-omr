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
chant-omr predict IMAGE_PATH [--checkpoint PATH] [--beam-width 3] \
    [--grammar-constrained] [--grammar-penalty FLOAT]
```

Loads the trained model and decodes the score image into GABC notation,
printed to stdout.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | auto-detect | Path to a `.ckpt` file |
| `--beam-width` | 3 | Beam search width (1 = greedy) |
| `--grammar-constrained` | off | Enable balanced-parenthesis grammar mask during decoding |
| `--grammar-penalty` | `-inf` | Logit penalty for grammar-forbidden tokens. `-inf` = hard mask; a finite value like `-10.0` = soft penalty that can be overridden by strong model confidence |

## `chant-omr evaluate`

Evaluate model on benchmark (image, GABC) pairs.

```bash
chant-omr evaluate [--checkpoint PATH] [--benchmark-dir benchmarks/] \
    [--grammar-constrained] [--grammar-penalty FLOAT] [--gregorio-check]
```

Reports aggregate metrics: GABC Edit Distance, neume accuracy, and
structural validity.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | auto-detect | Path to a `.ckpt` file |
| `--benchmark-dir` | `benchmarks/` | Directory with paired `.png` + `.gabc` files |
| `--grammar-constrained` | off | Enable grammar-constrained decoding during evaluation |
| `--grammar-penalty` | `-inf` | Logit penalty for grammar-forbidden tokens |
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
| `onnx` | `encoder.onnx`, `decoder_init.onnx`, `decoder_step.onnx`, `tokenizer.json`, `manifest.json` | **Primary.** Portable inference on any hardware via ONNX Runtime (CUDA, DirectML, CoreML, CPU). Uses KV cache with a two-model decoder strategy (init + step). |
| `openvino` | `encoder.xml/.bin`, `decoder.xml/.bin`, `manifest.json` | Intel-optimized path for Arc GPUs and NPUs |
| `safetensors` | `model.safetensors`, `manifest.json` | Full weights for HuggingFace distribution |

The ONNX decoder uses **4 stacked KV cache tensors** `(n_layers, B, H, S, head_dim)`
for efficient autoregressive generation. `decoder_init.onnx` runs the first
step (computing cross-attention K/V from encoder memory), and `decoder_step.onnx`
runs subsequent steps (reusing cached cross-attention K/V).

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

## `chant-omr manifest`

Manifest management commands for tracking dataset state.
