---
title: "Architecture"
weight: 30
description: "Model architecture: encoder, projector, decoder, and tokenizer"
---

ChantOMR uses a vision-encoder-decoder architecture: the image goes in,
autoregressive GABC tokens come out. There are no intermediate stages
(no staff detection, no symbol segmentation, no classification pipeline).

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
| `max_seq_len` | 2048 |
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
- **Framework:** PyTorch Lightning

## Inference

- **Greedy decode** with KV cache (default)
- **Beam search** (configurable beam width, default 3)
- **OpenVINO export** for deployment on Intel hardware
  ([ADR-0012](https://github.com/pgarciaq/chant-omr/blob/master/docs/adr/0012-openvino-export-and-inference-deployment.md))
