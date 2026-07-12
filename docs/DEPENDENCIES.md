# Dependencies

Python packages, PyTorch backends, and Fedora RPMs for chant-omr.

## Python (pip)

### Base install

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Core deps are in `pyproject.toml` (`torch`, `lightning`, `timm`, etc.).

### PyTorch backends (pick one)

`pip install -e .` pulls a default PyTorch wheel from PyPI (often CPU or CUDA
depending on platform). For training, **reinstall torch for your hardware** after
the base install:

| Backend | When | Install |
|---------|------|---------|
| **XPU** (Intel Arc) | Local training / overfit on laptop | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu` |
| **CUDA** (NVIDIA) | Cloud GPU (A100, Grace Hopper, …) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` (match CUDA version) |
| **CPU** | CI, debugging only | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

Verify:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); \
print('xpu', hasattr(torch,'xpu') and torch.xpu.is_available())"
```

Train with `--accelerator auto|cuda|xpu|cpu` (see README).

**Do not install IPEX** (`intel-extension-for-pytorch`) — retired; use stock
`torch.xpu` ([Intel notice](https://intel.github.io/intel-extension-for-pytorch/)).

### Optional extras

```bash
pip install -e ".[export]"   # OpenVINO + ONNX export (#13b)
pip install -e ".[dev]"      # pytest, ruff
```

## Fedora RPMs

### Rendering (Gregorio → PNG)

Required for `chant-omr render` / `scripts/render_dataset.py`:

```bash
sudo dnf install texlive-gregoriotex texlive-luatex texlive-libertinus-fonts \
  texlive-metapost poppler-utils
```

### Intel Arc GPU (PyTorch XPU training)

GPU compute stack (Level Zero + NEO). Sufficient for `torch.xpu` with XPU wheels:

```bash
sudo dnf install intel-compute-runtime oneapi-level-zero
```

Recommended:

```bash
# GPU device access
sudo usermod -aG render "$USER"   # re-login required

# Optional: OneAPI CLI tools (sycl-ls) — not required for pip PyTorch XPU
sudo dnf install intel-oneapi-base-toolkit
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

You do **not** need `intel-oneapi-neural-compressor`, `intel-oneapi-intelpython`,
or bundled OneAPI PyTorch conda envs for chant-omr.

### OpenVINO export (optional, #13b)

Runtime is installed via pip (`.[export]`). GPU inference still uses the RPMs above.

## ghh consumer (#15)

End-user OMR in ghh uses **OpenVINO only** (no PyTorch). See ghh `docs/DEPENDENCIES.md`
and `pip install ghh[omr]`.
