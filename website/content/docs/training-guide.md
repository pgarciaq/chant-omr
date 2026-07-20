---
title: "Training Guide"
weight: 40
description: "Step-by-step guide for training ChantOMR on a cloud GPU"
---

This is a condensed version of the full
[TRAINING-PLAN.md](https://github.com/pgarciaq/chant-omr/blob/master/TRAINING-PLAN.md).

## Recommended: Cloud GPU (QuickPod)

The cheapest path. The model is ~59M params and the dataset is ~20k images.
Training takes **~24 hours on a 16 GB GPU** and costs **~$4-12** depending
on GPU and provider.

### GPU recommendations

| GPU | $/hr | VRAM | Est. total |
|-----|------|------|------------|
| RTX 5080 | $0.16 | 16 GB | ~$4 |
| RTX 5090 | $0.50 | 32 GB | ~$6 |
| A100 PCIe | $0.39 | 40 GB | ~$5-9 |

Any GPU with **Ampere or newer** architecture and **16+ GB VRAM** works
with the default config (`batch_size: 2`, `accumulate_grad_batches: 4`,
effective batch size 8).

### Quick start

```bash
# SSH into the pod
ssh root@INSTANCE_IP -p 34200

# Install Python 3.13 (QuickPod ships 3.10)
apt update && apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y && apt update
apt install python3.13 python3.13-venv python3.13-dev -y

# Clone, create venv, install
git clone https://github.com/pgarciaq/chant-omr.git && cd chant-omr
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"

# Transfer dataset from your laptop (run on LAPTOP, not pod)
rsync -avz --progress -e 'ssh -p 34200' \
  data/gregobase/ data/rendered/ data/tokenizer/ \
  root@INSTANCE_IP:~/chant-omr/data/

# Train (on the pod, inside tmux)
tmux new -s train
source .venv/bin/activate
python scripts/train.py --accelerator cuda --precision bf16-mixed --epochs 50

# Copy best checkpoint back (run on LAPTOP)
scp -P 34200 root@INSTANCE_IP:~/chant-omr/checkpoints/best.ckpt \
  ~/dev/lpacleaner/chant-omr/checkpoints/
```

### Known constraints

| Constraint | Workaround |
|------------|------------|
| `/dev/shm` too small | Create a custom QuickPod template with `--shm-size=2g` in Docker Options (see below) |
| 16 GB VRAM OOM | `batch_size: 2` + `accumulate_grad_batches: 4` |
| Python 3.10 only | Install Python 3.13 from deadsnakes |
| SSH port 34200 | `-p 34200` for ssh, `-P 34200` for scp |

**Custom QuickPod template ("Pytorch Latest-largeSHM"):**
The default QuickPod `/dev/shm` is too small for PyTorch multiprocessing
DataLoaders. With 13 augmentation transforms, `num_workers: 0` causes ~4x
slower epochs. To fix this:

1. In QuickPod, go to Templates and clone "Pytorch Latest"
2. Name the clone "Pytorch Latest-largeSHM"
3. In the **Docker Options** field, add: `--shm-size=2g`
4. Save and use this template when creating pods

This gives 2 GB of shared memory, enabling `num_workers: 4` in
`configs/default.yaml` for parallel augmentation.

## Alternative: Bare metal (NVIDIA GPU)

For dedicated hardware (A100, H100, Grace Hopper), see the full
[TRAINING-PLAN.md](https://github.com/pgarciaq/chant-omr/blob/master/TRAINING-PLAN.md)
which covers RHEL 9 setup, NVIDIA driver installation, and CUDA configuration.

## Configuration

Training parameters are in `configs/default.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 2 | Fits 16 GB VRAM |
| `accumulate_grad_batches` | 4 | Effective batch size = 8 |
| `precision` | `bf16-mixed` | Requires Ampere+ GPU |
| `epochs` | 50 | Ceiling; early stopping halts sooner |
| `early_stopping.patience` | 10 | Stop after 10 epochs with no val_loss improvement |
| `learning_rate` | 1e-4 | |
| `num_workers` | 4 | Requires `--shm-size=2g` in container; set to 0 as fallback |
| `augment` | `true` | 13 domain transforms applied on-the-fly during training |

## After Training

1. Copy the best checkpoint (lowest `val_loss`) to your development machine
2. **Terminate the cloud instance** to stop billing
3. Run evaluation:
   ```bash
   chant-omr evaluate --checkpoint checkpoints/best.ckpt --gregorio-check
   ```
4. Export for deployment:
   ```bash
   # ONNX (primary — runs on any hardware via ONNX Runtime)
   pip install onnx onnxruntime   # if not already installed
   chant-omr export checkpoints/best.ckpt --format onnx --output-dir models/ --verify

   # OpenVINO (optional — optimized for Intel Arc GPUs and NPUs)
   pip install -e ".[export]"    # installs openvino + nncf
   chant-omr export checkpoints/best.ckpt --format openvino --output-dir models/ --verify
   ```
   ONNX produces `encoder.onnx`, `decoder.onnx`, `decoder_init.onnx`,
   `decoder_step.onnx`, `tokenizer.json`, and `manifest.json`. OpenVINO
   produces the equivalent `.xml/.bin` IR files. The `--verify` flag runs
   numeric parity checks against the PyTorch model.
