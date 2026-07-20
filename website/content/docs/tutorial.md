---
title: "Tutorial"
weight: 10
description: "Install ChantOMR, prepare training data, train a model, and run inference"
---

## Installation

ChantOMR requires Python 3.11 or later (up to 3.13). Python 3.14 is not yet
supported.

```bash
git clone https://github.com/pgarciaq/chant-omr.git
cd chant-omr
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For OpenVINO export support, install the `export` extra:

```bash
pip install -e ".[dev,export]"
```

## Preparing the Training Data

The training pipeline has three stages, each a CLI command:

### 1. Download the GABC corpus

```bash
chant-omr download
```

This downloads ~10,000 GABC files from [GregoBase](https://gregobase.selapa.net/)
into `data/gregobase/`. The download is incremental — re-running skips
files already present.

### 2. Render to PNG

```bash
chant-omr render
```

Renders each GABC file into a score image using Gregorio + LuaLaTeX.
Output goes to `data/rendered/` as paired `.gabc` + `.png` files.
Requires TeX Live with the Gregorio package installed.

### 3. Train the BPE tokenizer

```bash
chant-omr train-tokenizer
```

Trains a Byte Pair Encoding tokenizer on the GABC bodies (notation only,
not headers). Produces `data/tokenizer/tokenizer.json` with ~2000 tokens.

## Training

### On a cloud GPU (recommended)

The cheapest path is a cloud GPU instance. See the [Training Guide](../training-guide/)
for step-by-step instructions with QuickPod, Vast.ai, or Lambda Labs.

```bash
python scripts/train.py \
  --accelerator cuda \
  --precision bf16-mixed \
  --epochs 50
```

### On Intel Arc (XPU)

ChantOMR supports Intel Arc GPUs via PyTorch XPU:

```bash
python scripts/train.py \
  --accelerator xpu \
  --precision bf16-mixed \
  --epochs 50
```

### Overfit smoke test

Before a full training run, verify the pipeline works:

```bash
python scripts/train.py \
  --accelerator cuda \
  --overfit-n 10 \
  --epochs 20 \
  --batch-size 2
```

Loss should decrease rapidly on 10 overfitted samples.

## Inference

Run OMR on a single image:

```bash
chant-omr predict manuscript_page.png
```

This loads the trained model (from `checkpoints/`) and outputs GABC notation
to stdout.

## Evaluation

Evaluate the model on a benchmark set of (image, GABC) pairs:

```bash
chant-omr evaluate --checkpoint checkpoints/best.ckpt --benchmark-dir benchmarks/
```

Reports GABC Edit Distance (GED), neume accuracy, and structural validity.

## ghh Integration

ChantOMR integrates with [Guido's Helping Hand](https://pgarciaq.github.io/ghh/)
as Stage 14 (OMR). After ghh processes your book photos through its pipeline
(crop, deskew, perspective correction, etc.), the OMR stage runs ChantOMR
inference on each music page and produces `.gabc` files.

### Setup

1. Install ghh with the OMR extra:

```bash
pip install ghh[omr]
```

2. Install chant-omr (until it's published on PyPI):

```bash
pip install -e /path/to/chant-omr
```

3. Export your trained model to OpenVINO:

```bash
chant-omr export checkpoints/best.ckpt --format openvino --output-dir models/
```

### Running OMR in ghh

```bash
ghh run /path/to/photos --model-dir /path/to/models/
```

The `--model-dir` flag points to the directory containing the exported
OpenVINO IR files. Only pages classified as "music" by Stage 4 (Page
Detect) are processed; text and blank pages pass through unchanged.

See the [ghh Pipeline Stages](https://pgarciaq.github.io/ghh/docs/pipeline/)
documentation for details on Stage 14.
