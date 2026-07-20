---
title: "Architecture"
weight: 30
description: "Model architecture: encoder, projector, decoder, and tokenizer"
---

ChantOMR follows the [Transcoda](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1)
vision-encoder-decoder architecture (~59M params), retrained from scratch
for Gregorian square notation with GABC output. The image goes in,
autoregressive GABC tokens come out. There are no intermediate stages
(no staff detection, no symbol segmentation, no classification pipeline).

No Transcoda weights are reused -- only the architecture design
(ConvNeXt-V2 Tiny encoder, 8-layer Transformer decoder, BPE tokenizer).
See [ADR-0008](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0008-end-to-end-over-classical-omr.md)
for the full rationale.

## Overview

| Component | Implementation | Parameters |
|-----------|---------------|------------|
| Encoder | ConvNeXt-V2 Tiny | 28.6M |
| Projector | 2D sinusoidal + MLP | ~0.8M |
| Decoder | Transformer (Pre-LN, RoPE) | ~29.8M |
| **Total** | | **~59M** |

## Encoder: ConvNeXt-V2 Tiny

The encoder is a ConvNeXt-V2 Tiny backbone (`convnextv2_tiny.fcmae_ft_in22k_in1k`)
from the `timm` library, pretrained on ImageNet.

**Input:** Score images at width 1050, variable height. Images are
ImageNet-normalized (mean/std per channel).

**Output:** A grid of feature patches at stride 32, with dimension 768.
For a 1050×H image, the patch grid is approximately 33×(H/32) patches.

The variable-height design avoids padding and information loss. See
[ADR-0002](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0002-variable-height-patch-grid.md).

## Projector: 2D Sinusoidal + MLP

The encoder outputs are 768-dimensional; the decoder expects 512-dimensional
input. The projector bridges them:

1. **2D sinusoidal positional encoding** adds spatial awareness (row and
   column position in the patch grid)
2. **MLP** projects 768 → 512 dimensions

See [ADR-0009](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0009-mlp-projector-and-2d-sinusoidal-bridge.md).

## Decoder: Transformer with RoPE

An 8-layer Transformer decoder with:

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `n_heads` | 8 |
| `d_ff` | 1024 |
| `dropout` | 0.1 |
| `max_seq_len` | 8192 |
| Positional encoding | RoPE (Rotary Position Embedding) |
| Normalization | Pre-LayerNorm |

The decoder uses **causal self-attention** (each token attends only to
previous tokens) and **cross-attention** to the encoder output. At inference
time, a **KV cache** enables O(n) autoregressive generation instead of
O(n²).

See [ADR-0006](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0006-transcoda-decoder-architecture.md).

## Tokenizer: BPE on GABC Bodies

A Byte Pair Encoding tokenizer trained on the notation bodies of ~20,000
GregoBase GABC files. The vocabulary is ~2000 tokens.

GABC headers (name, mode, book, etc.) are stripped — the tokenizer operates
only on the notation body (the part after `%%`).

Special tokens: `<bos>` (beginning of sequence), `<eos>` (end of sequence),
`<pad>` (padding).

See [ADR-0003](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0003-bpe-body-only-tokenizer.md).

## Training

- **Loss:** Teacher-forcing cross-entropy
  ([ADR-0011](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0011-teacher-forcing-cross-entropy-loss.md))
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.05)
- **Precision:** bf16-mixed on Ampere+ GPUs
- **Batch size:** 2 (effective 8 via gradient accumulation)
- **Early stopping:** patience=10 epochs, halts training when val_loss
  plateaus (min_delta=0.001)
- **Domain augmentation:** 13 on-the-fly OpenCV transforms simulate aged
  manuscripts. See the [Augmentation Guide]({{< relref "augmentation" >}}).
- **Framework:** PyTorch Lightning

## Inference

- **Greedy decode** with KV cache for O(n) autoregressive generation (default)
- **Beam search** (configurable beam width, default 3)
- **Grammar-constrained decoding:** Optional balanced-parenthesis mask
  prevents structurally invalid GABC. Supports hard masking (`-inf`) or
  soft penalty mode (configurable finite penalty, e.g. `-10.0`)
- **Encoding-equivalence normalization:** Gregorio round-trip normalizes
  predictions and references before metric computation, ensuring fair
  comparison across equivalent GABC encodings
- **Gregorio compilation check:** Gold-standard structural validation by
  compiling predictions through the Gregorio binary
- **Dual-path decoding:** Greedy decoding uses a fast KV-cached path
  (`decoder_init` + `decoder_step`); beam search uses a non-cached
  `decoder` model that recomputes attention each step for correctness
- **ONNX export** (primary) for deployment on any hardware via ONNX Runtime.
  Exports four decoder variants: `decoder.onnx` (non-cached, beam search),
  `decoder_init.onnx` (first cached step), and `decoder_step.onnx`
  (subsequent cached steps). KV cache uses 4 stacked tensors
  `(n_layers, B, H, S, head_dim)` for clean ONNX I/O
- **OpenVINO export** for accelerated inference on Intel Arc GPUs and NPUs.
  Same four-model layout: `encoder.xml`, `decoder.xml` (non-cached),
  `decoder_init.xml` (cached init), `decoder_step.xml` (cached step)
  ([ADR-0012](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0012-openvino-export-and-inference-deployment.md))

## ghh Integration API

ChantOMR exposes a numpy-array API for integration with
[Guido's Helping Hand](https://pgarciaq.github.io/ghh/) (ghh), which
uses it in Stage 14 (OMR) to transcribe music pages.

| Function | Module | Purpose |
|----------|--------|---------|
| `prepare_inference_numpy_from_array()` | `preprocess` | Convert an in-memory `(H, W, 3)` uint8 image to `(1, 3, H, W)` float32 |
| `load_openvino_models()` | `ov_decode` | Load and compile all OpenVINO IRs, returns an `OvModelBundle` |
| `ov_predict_gabc_from_array()` | `ov_decode` | Run inference on a preprocessed array using a pre-loaded `OvModelBundle` |

The `OvModelBundle` dataclass holds the compiled encoder, decoder (cached
and non-cached), manifest, and tokenizer. Loading models once and reusing
the bundle across many images avoids repeated compilation overhead.

```python
from chant_omr.inference.ov_decode import load_openvino_models, ov_predict_gabc_from_array
from chant_omr.inference.preprocess import prepare_inference_numpy_from_array

models = load_openvino_models(Path("models/"), device="AUTO")
pixels = prepare_inference_numpy_from_array(rgb_array)
gabc = ov_predict_gabc_from_array(pixels, models, beam_width=1, name="page_042")
```
