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

## Using the Pre-Trained Model

The fastest way to get started is with the pre-trained model from
[HuggingFace](https://huggingface.co/pgquiles/chant-omr). No training
needed -- just install and run:

```bash
pip install chant-omr
chant-omr predict score.png --model pgquiles/chant-omr
```

The model is downloaded automatically on first use and cached locally.
Choose a backend with `--device`:

```bash
# OpenVINO (Intel Arc GPUs/NPUs)
chant-omr predict score.png --model pgquiles/chant-omr --device openvino

# ONNX Runtime (any hardware)
chant-omr predict score.png --model pgquiles/chant-omr --device onnx

# PyTorch (auto-selects CUDA > XPU > CPU)
chant-omr predict score.png --model pgquiles/chant-omr --device auto
```

The HuggingFace repository contains safetensors weights, OpenVINO IR,
and ONNX models. The CLI automatically selects the right format based
on `--device`.

### From Python

```python
from chant_omr.hub import download_from_hub
from chant_omr.inference.ov_decode import load_openvino_models, ov_predict_gabc

model_dir = download_from_hub("pgquiles/chant-omr")
gabc = ov_predict_gabc("score.png", model_dir / "openvino", beam_width=3)
print(gabc)
```

## Training from Scratch

The following sections describe how to prepare data and train your own
model. Skip this if you only need inference with the pre-trained model.

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

Run OMR on a single image using the HuggingFace model:

```bash
chant-omr predict manuscript_page.png --model pgquiles/chant-omr
```

Or using a local checkpoint:

```bash
chant-omr predict manuscript_page.png --checkpoint checkpoints/best.ckpt
```

Both output GABC notation to stdout. Use `--output file.gabc` to write
to a file instead.

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
