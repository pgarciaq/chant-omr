---
title: "Design Decisions"
weight: 50
description: "Architecture Decision Records (ADRs) documenting key technical choices"
---

ChantOMR documents major technical decisions as Architecture Decision Records
(ADRs). Each ADR captures the context, decision, and consequences of a
design choice.

The full ADR files are in
[`docs/adr/`](https://github.com/pgarciaq/chant-omr/tree/master/docs/adr)
on GitHub.

## ADR Index

| # | Decision | Summary |
|---|----------|---------|
| 0001 | Record architecture decisions | Use lightweight ADR format for all significant technical choices |
| 0002 | Variable-height score strips | Accept variable-height images (fixed width 1050) with a variable-size patch grid — no padding, no cropping |
| 0003 | BPE tokenizer on GABC bodies | Train BPE on notation bodies only (after `%%`), not headers — ~2000 token vocabulary |
| 0004 | PNG-first dataset pairing | Index by rendered PNGs, require matching GABC sidecar — reject orphans in either direction |
| 0005 | Parallel render workers | Use multiprocessing for Gregorio rendering with a shared TeX cache directory |
| 0006 | Transcoda-aligned decoder | Pre-LN Transformer decoder with RoPE, aligned with Transcoda's proven architecture |
| 0007 | Defer NABC for v0 | NABC (neume-level) notation support deferred — v0 targets standard GABC only |
| 0008 | End-to-end over classical OMR | Single encoder-decoder model instead of staff detection → segmentation → classification pipeline |
| 0009 | MLP projector with 2D sinusoidal | Bridge encoder (768-dim) to decoder (512-dim) with spatial positional encoding |
| 0010 | Encoder padding mask in collate | Handle variable-height images via padding masks created during batch collation |
| 0011 | Teacher-forcing cross-entropy | Standard teacher-forcing training loss — simple, effective, well-understood |
| 0012 | OpenVINO export and deployment | Export to OpenVINO IR for Intel hardware inference — no PyTorch at runtime |
| 0013 | GABC output assembly | Assemble decoded tokens into valid GABC with synthetic headers and structural validation |
