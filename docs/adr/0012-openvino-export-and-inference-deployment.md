# 0012. OpenVINO export and inference deployment

- **Status:** Accepted
- **Date:** 2026-07-12
- **Issue:** [#13](https://github.com/pgarciaq/chant-omr/issues/13)

## Context

Training runs in **PyTorch** (cloud NVIDIA GPU, e.g. Grace Hopper). Production
inference runs in **ghh** on user **Intel Arc GPU/NPU** via **OpenVINO** — no
PyTorch at runtime ([PLAN.md](../../PLAN.md) design principle 7, [#15](https://github.com/pgarciaq/chant-omr/issues/15)).

The model has two export challenges:

1. **Variable image height** → variable encoder patch count (`H'×W'`, stride 32).
2. **Autoregressive decoder** → generation is a loop, not one static forward pass.

Issue #13 must deliver both a **PyTorch dev path** (iterate on Arc/CPU during
development) and an **OpenVINO production path** (ghh consumer).

## Decision

### Two-phase #13 delivery

| Phase | Deliverable | Runtime |
|-------|-------------|---------|
| **13a** | `predict_gabc()`, beam search, checkpoint load, GABC assembly | PyTorch (CPU/CUDA/XPU) |
| **13b** | OpenVINO export + safetensors weights | OpenVINO on Arc/NPU (#15) |

Implement **13a first**; do not block predict/beam tests on export tooling.

### OpenVINO v0 export strategy

- **Pad every inference image to fixed canvas** `1050×1600` (width × max height from
  config) using the same resize policy as training, then bottom-pad with white if
  shorter. Encoder patch grid becomes **fixed** `50×32 = 1600` patches at stride 32.
- Export **encoder + projector path** (image → projected memory `(1, N, d_model)`)
  as OpenVINO IR where practical for v0.
- Run **decoder beam search in Python** (PyTorch or OpenVINO Runtime with a
  single-step decoder graph added incrementally). Full end-to-end autoregressive IR
  with dynamic sequence length is **deferred** until [#36](https://github.com/pgarciaq/chant-omr/issues/36)
  (KV cache) clarifies the stateful graph.
- Also export **safetensors** weights for distribution and HuggingFace upload (deferred
  as a separate task).

### Development vs production runtimes

| Activity | Hardware | Stack |
|----------|----------|-------|
| Full training, overfit gate | NVIDIA cloud (Grace Hopper) or local Arc (slow) | PyTorch Lightning |
| chant-omr dev / debug predict | Fedora + Arc or CPU | PyTorch |
| ghh end-user OMR | Intel Arc GPU/NPU | OpenVINO IR (#15) |

## Consequences

### Positive

- Matches ghh's existing OpenVINO usage (e.g. DocTr dewarp).
- Small ~59M model fits Arc inference budget; no CUDA dependency for users.
- Fixed-height padding simplifies v0 export without changing training collation.
- PyTorch predict unblocks overfit validation before export is perfect.

### Negative / trade-offs

- v0 OpenVINO may not be a single "one-shot" IR for the full generate loop.
- Fixed 1600px canvas wastes compute on short strips (acceptable v0).
- Export pipeline must stay in sync with `ChantOMR.encode()` and config dims.
- KV cache (#36) may require revisiting decoder export shape.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| PyTorch-only inference in ghh | Multi-GB deps; poor Arc path; contradicts ghh footprint goals |
| Full dynamic-shape E2E ONNX → OpenVINO | RoPE + cross-attn + variable seq brittle; high v0 risk |
| OpenVINO only, no PyTorch predict | Slows #13 development and CI |
| Always variable-height IR (no pad) | Harder export; fixed pad is simpler v0 |
