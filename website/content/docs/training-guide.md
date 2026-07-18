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
| `/dev/shm` too small | `num_workers: 0` in config |
| 16 GB VRAM OOM | `batch_size: 2` + `accumulate_grad_batches: 4` |
| Python 3.10 only | Install Python 3.13 from deadsnakes |
| SSH port 34200 | `-p 34200` for ssh, `-P 34200` for scp |

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
| `epochs` | 50 | |
| `learning_rate` | 1e-4 | |
| `num_workers` | 0 | Safe for containers |

## After Training

1. Copy the best checkpoint (lowest `val_loss`) to your development machine
2. **Terminate the cloud instance** to stop billing
3. Run evaluation: `chant-omr evaluate --checkpoint checkpoints/best.ckpt`
4. Export for deployment: `chant-omr export --checkpoint checkpoints/best.ckpt`
