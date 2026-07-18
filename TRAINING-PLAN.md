# ChantOMR Training Plan

How to run the full training on an NVIDIA GPU.  Two paths are documented:

- **Option A** — Cloud GPU rental (Lambda Labs, RunPod, etc.) — cheapest,
  fastest to get started, ~$5-10 total.
- **Option B** — NVIDIA Grace Hopper (GH200) bare metal on RHEL 9 — for when
  you have access to dedicated hardware.

Related: [#49 Full training run on NVIDIA GPU](https://github.com/pgarciaq/chant-omr/issues/49)

---

# Option A: Cloud GPU

The fastest path.  Cloud GPU providers offer Ubuntu instances with CUDA +
PyTorch pre-installed.  No driver setup needed — just SSH in, clone, and train.

## Estimated cost

The ChantOMR model is small (~56M params) and the dataset is ~20k images.
Training takes **3-6 hours on a consumer/prosumer GPU**, costing under $5 total.

### Recommended providers and GPUs

| Provider | GPU | VRAM | $/hr | Est. total | Notes |
|----------|-----|------|------|------------|-------|
| **QuickPod** | RTX 5080 | 16 GB | $0.16 | **~$0.50-1** | Best value. Blackwell, bf16. |
| QuickPod | RTX PRO 4000 | 20 GB | $0.19 | ~$0.60-1 | Blackwell professional. |
| QuickPod | A100 PCIe | 40 GB | $0.39 | ~$1-2 | Datacenter class, overkill but fast. |
| QuickPod | RTX 5090 | 32 GB | $0.51 | ~$2-3 | Flagship consumer. |
| QuickPod | RTX 6000 Ada | 48 GB | $0.56 | ~$2-3 | Professional Ada. |
| Vast.ai | A100 (spot) | 40-80 GB | ~$0.60-0.80 | ~$2-3 | Interruptible. |
| Lambda Labs | A100 80GB | 80 GB | $1.29 | ~$4-8 | Reliable, reserved available. |
| RunPod | A100 80GB | 80 GB | $1.39 | ~$4-8 | Per-second billing. |

Prices as of July 2026.

### GPU compatibility

Any GPU with **Ampere or newer** architecture and **12+ GB VRAM** works:

| Architecture | GPUs | bf16 | Precision flag |
|-------------|------|------|----------------|
| Blackwell (2025) | RTX 5060 Ti/5070 Ti/5080/5090, PRO 4000 | Yes | `--precision bf16-mixed` |
| Ada Lovelace (2023) | RTX 4070 Super, L40S, RTX 6000 Ada | Yes | `--precision bf16-mixed` |
| Ampere (2021) | A100, A40, A10, RTX 3080, A4000, A2000 | Yes | `--precision bf16-mixed` |
| Volta (2018) | V100 | **No** | `--precision 16-mixed` |
| Turing (2019) | T4 | **No** | Avoid — very slow for training |

For 12 GB GPUs (RTX 3080, 4070 Super, A2000), use `--batch-size 4`.
For 16+ GB GPUs, `--batch-size 8` (default) works fine.

### Avoid

- **AWS/GCP/Azure** — same hardware at 5-10x the price.
- **Google Colab** — sessions disconnect after 3-5 hours, no SSH, awkward
  data transfer, A100 burns ~15 compute units/hr.
- **T4** — inference GPU, very slow for training (12-24h+).

## Step-by-step (QuickPod)

QuickPod pods run **Ubuntu 22.04 (Jammy)** with **CUDA 13.2** pre-installed
but only **Python 3.10** (which is too old — ChantOMR requires `>=3.11`).
Install Python 3.13 from the deadsnakes PPA before creating the venv.

Disk space needed: **~15 GB** (dataset ~2 GB, venv ~6 GB, checkpoints ~3 GB,
code + logs ~1 GB).  QuickPod default is 30 GB — just enough; no need to
increase it.

### Known QuickPod constraints

| Constraint | Impact | Workaround (already in default config) |
|------------|--------|----------------------------------------|
| `/dev/shm` too small, `mount -o remount` denied | PyTorch DataLoader multiprocessing crashes with "No space left on device" | `num_workers: 0` in config (loads data in main process) |
| 16 GB VRAM (RTX 5080/5090) | `batch_size: 8` OOMs during encoder forward pass | `batch_size: 4` + `accumulate_grad_batches: 2` (effective batch = 8) |
| Python 3.10 only | ChantOMR requires `>=3.11` | Install Python 3.13 from deadsnakes PPA (step 3) |
| SSH on port 34200 | Standard rsync/scp commands fail | Pass `-p 34200` / `-P 34200` / `-e 'ssh -p 34200'` |

All workarounds are already applied in `configs/default.yaml`.  No manual
config edits needed — just `git pull` and run.

### 1. Create a pod

On [console.quickpod.io](https://console.quickpod.io/):

1. Select a **GPU** — RTX 5080 ($0.16/hr) or RTX 5090 ($0.50/hr) recommended.
2. Select any available **template** (PyTorch preferred, or vanilla Ubuntu).
3. Click **Create**.  Note the SSH connection command from the pod dashboard.

### 2. SSH in and clone the repo

```bash
ssh root@INSTANCE_IP -p 34200   # connection details from QuickPod dashboard
git clone https://github.com/pgarciaq/chant-omr.git
cd chant-omr
```

### 3. Install Python 3.13 and create venv

QuickPod pods ship Python 3.10 which is too old.  Install 3.13 from
deadsnakes (takes ~2 minutes):

```bash
apt update && apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.13 python3.13-venv python3.13-dev -y
```

Create the venv with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Transfer the dataset from your laptop

From your **laptop** (not the pod).  QuickPod SSH uses a non-standard port,
so pass `-e 'ssh -p 34200'` to rsync:

```bash
rsync -avz --progress -e 'ssh -p 34200' \
  ~/dev/lpacleaner/chant-omr/data/gregobase/ \
  ~/dev/lpacleaner/chant-omr/data/rendered/ \
  ~/dev/lpacleaner/chant-omr/data/tokenizer/ \
  root@INSTANCE_IP:~/chant-omr/data/
```

The dataset is ~2 GB and transfers in 1-3 minutes.

### 5. Set num_workers to 0

QuickPod containers have a tiny `/dev/shm` and block `mount -o remount`,
so PyTorch multiprocessing DataLoader crashes.  The default config already
has `num_workers: 0`, but if you had previously changed it:

```bash
sed -i 's/num_workers: [0-9]*/num_workers: 0/' configs/default.yaml
```

### 6. Smoke test

```bash
python scripts/train.py \
  --accelerator cuda \
  --overfit-n 10 \
  --epochs 20 \
  --batch-size 2
```

Verify loss decreases.  Check GPU utilization: `nvidia-smi`.

### 7. Full training run

The default config uses `batch_size: 4` with `accumulate_grad_batches: 2`
(effective batch size 8), which fits in 16 GB VRAM.  No overrides needed:

```bash
tmux new -s train
source .venv/bin/activate
python scripts/train.py \
  --accelerator cuda \
  --precision bf16-mixed \
  --epochs 50
```

Use `tmux` so the training survives SSH disconnections.

With an RTX 5080/5090, expect **4-7 hours**.

### 8. Copy the best checkpoint back

From your **laptop**, pull the best checkpoint (note the `-P 34200` for scp):

```bash
scp -P 34200 \
  root@INSTANCE_IP:~/chant-omr/checkpoints/chant-omr-epoch=XX-val_loss=X.XXXX.ckpt \
  ~/dev/lpacleaner/chant-omr/checkpoints/
```

Copy the **1 best checkpoint** (lowest val_loss).  Then **terminate the
pod** from the QuickPod dashboard to stop billing.

---

# Option B: NVIDIA Grace Hopper (GH200) on RHEL 9

For when you have access to dedicated GH200 hardware.  This path requires
installing RHEL 9, NVIDIA drivers, and CUDA from scratch.  The machine is
**aarch64** (ARM) — this affects package names and repos.

---

## Phase 0: Install RHEL 9 on bare metal

1. Download the RHEL 9 **aarch64** ISO from the
   [Red Hat Customer Portal](https://access.redhat.com/downloads/content/rhel).
   Do NOT use the x86_64 variant.
2. Boot from the ISO and run the Anaconda installer.  Defaults are fine for a
   training workstation.  Select **Server with GUI** or **Minimal Install**
   (GUI is optional — you will work via SSH).
3. After install, register the system and attach a subscription:

```bash
sudo subscription-manager register --username YOUR_RH_USERNAME
sudo subscription-manager attach --auto
```

4. Update the system:

```bash
sudo dnf update -y
sudo reboot
```

---

## Phase 1: NVIDIA GPU driver + CUDA toolkit

Grace Hopper requires the **open-source (OpenRM) driver**.  The proprietary
driver does not work on Hopper GPUs.

### 1.1 Install kernel development headers

```bash
sudo dnf install kernel-headers-$(uname -r | sed 's/+64k//g') -y
sudo dnf install kernel-devel-matched-$(uname -r | sed 's/+64k//g') -y
sudo dnf install kernel-64k-devel-matched-$(uname -r | sed 's/+64k//g') -y
```

### 1.2 Add the NVIDIA CUDA repository

`sbsa` in the URL stands for Server Base System Architecture (ARM server).
This is the correct repo for Grace Hopper — do NOT use the x86_64 repo.

```bash
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm -y
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo
sudo dnf clean expire-cache
```

### 1.3 Install CUDA toolkit + open-source GPU driver

```bash
sudo dnf install cuda-toolkit-13-0 -y
sudo dnf module install nvidia-driver:580-open -y
```

### 1.4 Enable persistence daemon and reboot

```bash
sudo systemctl enable nvidia-persistenced
sudo reboot
```

### 1.5 Verify (after reboot)

```bash
nvidia-smi
```

Expected output:

```
+-------------------------------------------------------------------------+
| NVIDIA-SMI 580.xx.xx    Driver Version: 580.xx.xx    CUDA Version: 13.0 |
|  GPU   Name          ...   Memory-Usage   ...                           |
|  0     GH200 480GB   ...   0MiB / 98304MiB (or 98304+480GB unified)    |
+-------------------------------------------------------------------------+
```

If `nvidia-smi` fails, the driver did not load.  Check `dmesg | grep nvidia`
for errors.

### 1.6 Add CUDA to your PATH

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Then `source ~/.bashrc`.

Verify: `nvcc --version` should print CUDA 13.0.

---

## Phase 2: Python 3.13 + project venv

RHEL 9 ships Python 3.9/3.11/3.12 in AppStream.  Python 3.13 is available
from EPEL (already enabled in Phase 1).

### 2.1 Enable CRB and install Python 3.13

```bash
sudo subscription-manager repos \
  --enable codeready-builder-for-rhel-9-aarch64-rpms
sudo dnf install python3.13 python3.13-pip python3.13-devel -y
```

Verify: `python3.13 --version` should show 3.13.x.

### 2.2 Install system dependencies for rendering (optional)

Only needed if you plan to render GABC on the GH200 rather than transferring
pre-rendered data from your laptop (see Phase 3).

```bash
sudo dnf install texlive-gregoriotex texlive-luatex texlive-libertinus-fonts \
  texlive-metapost poppler-utils -y
```

If `texlive-gregoriotex` is not available in RHEL 9 AppStream, install the
full TeX Live collection:

```bash
sudo dnf install texlive-scheme-full -y
```

(Large, ~4 GB, but guarantees all TeX dependencies.)

### 2.3 Install git and clone the repo

```bash
sudo dnf install git -y
git clone https://github.com/pgarciaq/chant-omr.git
cd chant-omr
```

### 2.4 Create venv and install dependencies

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Since PyTorch 2.11.0, `pip install torch` on aarch64 automatically pulls
CUDA-enabled wheels from PyPI — no `--index-url` needed:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2.5 Verify PyTorch sees the GPU

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memory:', torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')
"
```

Expected output: CUDA available = True, GPU = GH200 or similar.

**If `torch.cuda.is_available()` returns False:**
- Check that `nvidia-smi` works (driver loaded)
- Check `nvcc --version` matches the CUDA version PyTorch was built for
- Try explicit install: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

---

## Phase 3: Prepare the dataset

The GABC corpus and rendered score images already exist on the dev laptop.
**Transfer them** rather than re-downloading/re-rendering (saves hours).

### 3.1 Transfer data from your laptop

From your **laptop** (not the GH200), run:

```bash
rsync -avz --progress \
  ~/dev/lpacleaner/chant-omr/data/gregobase/ \
  ~/dev/lpacleaner/chant-omr/data/rendered/ \
  ~/dev/lpacleaner/chant-omr/data/tokenizer/ \
  you@gh200-machine:~/chant-omr/data/
```

Replace `you@gh200-machine` with your actual SSH user and hostname.

This copies three directories:
- `data/gregobase/` — GABC files + `manifest.json`
- `data/rendered/` — paired `.png` + `.gabc` score images (~20k pairs)
- `data/tokenizer/` — trained BPE tokenizer

`rsync` preserves timestamps and skips already-transferred files on subsequent
runs, so it is safe to re-run if interrupted.

### 3.2 Verify the dataset on the GH200

```bash
source .venv/bin/activate
ls data/rendered/*.png | wc -l     # should be ~19-20k files
ls data/tokenizer/                  # should contain tokenizer.json
```

### 3.3 Audit token lengths (optional)

```bash
chant-omr audit-tokens
```

Checks that the dataset fits within `max_seq_len=2048`.

### 3.4 Alternative: re-download/re-render on the GH200

Only if you cannot transfer data, or need a fresh dataset:

```bash
chant-omr download                # ~5.5h at 1 req/s
chant-omr render --workers 0      # several hours
chant-omr train-tokenizer
```

This requires the TeX dependencies from Phase 2.2.

---

## Phase 4: Train the model

### 4.1 Smoke test (overfit on 10 samples)

Before committing to a full run, verify everything works end-to-end:

```bash
python scripts/train.py \
  --accelerator cuda \
  --overfit-n 10 \
  --epochs 20 \
  --batch-size 2
```

Watch for:
- Training starts without errors
- Loss decreases (should drop significantly on 10 overfitted samples)
- GPU utilization visible in `nvidia-smi` (run in another terminal)

### 4.2 Full training run

```bash
python scripts/train.py \
  --accelerator cuda \
  --precision bf16-mixed \
  --batch-size 8 \
  --epochs 50
```

**Key parameters** (from `configs/default.yaml`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `precision` | `bf16-mixed` | GH200 Hopper GPU has native bf16 support |
| `batch_size` | 8 | Increase if GPU memory allows (GH200 has 96+ GB) |
| `epochs` | 50 | Adjust based on convergence |
| `learning_rate` | 1e-4 | AdamW with cosine warmup |
| `encoder_pretrained` | true | Uses ImageNet-pretrained ConvNeXt-V2 Tiny |

**To increase batch size** (GH200 has massive memory):

```bash
python scripts/train.py \
  --accelerator cuda \
  --precision bf16-mixed \
  --batch-size 32 \
  --epochs 50
```

### 4.3 Monitor training

In a second terminal:

```bash
# GPU utilization (refreshes every 1s)
watch -n 1 nvidia-smi

# TensorBoard (if Lightning logs are enabled)
pip install tensorboard
tensorboard --logdir lightning_logs/
```

### 4.4 Checkpoints

Checkpoints are saved to `checkpoints/` (configurable in `configs/default.yaml`).
The top 3 by `val_loss` are kept.  To resume from a checkpoint:

```bash
python scripts/train.py \
  --accelerator cuda \
  --resume checkpoints/chant-omr-epoch=XX-val_loss=X.XXXX.ckpt
```

---

## Phase 5: Evaluate and extract the model

### 5.1 Run evaluation on the GH200

```bash
chant-omr evaluate \
  checkpoints/chant-omr-epoch=XX-val_loss=X.XXXX.ckpt \
  --benchmark-dir data/rendered/
```

### 5.2 Identify the best checkpoint

List saved checkpoints sorted by val_loss:

```bash
ls -1 checkpoints/ | sort -t= -k4 -n
```

Training saves the **top 3** by val_loss (`save_top_k: 3`).  The file with the
lowest `val_loss` number in its name is the best.

### 5.3 Copy the best checkpoint to your dev laptop

Copy **only the 1 best checkpoint** (~100-120 MB) to your laptop:

```bash
# From the GH200 machine
scp checkpoints/chant-omr-epoch=XX-val_loss=X.XXXX.ckpt \
  you@laptop:~/dev/lpacleaner/chant-omr/checkpoints/
```

Keep all 3 on the GH200 until you have verified the best one works on your
laptop (loads, runs inference, exports to OpenVINO).  If it turns out the best
one overfits or has issues, you have the 2nd and 3rd-best as fallbacks.  Once
verified, the GH200 copies can be deleted.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvidia-smi` not found | Driver not installed.  Re-run Phase 1.3 |
| `torch.cuda.is_available()` = False | Driver/CUDA mismatch.  Check `nvidia-smi` CUDA version vs `torch.version.cuda` |
| `texlive-gregoriotex` not in RHEL 9 | Use `texlive-scheme-full` or install from EPEL |
| `pip install torch` pulls CPU wheel | PyTorch < 2.11.  Force: `pip install torch>=2.11` |
| OOM during training | Reduce `--batch-size` or use `--precision bf16-mixed` |
| Slow rendering (gregorio) | Increase `--workers`.  Check TeX is properly installed: `lualatex --version` |
| SSH disconnects kill training | Use `tmux` or `screen`: `tmux new -s train` before starting |

---

## Quick reference: full sequence of commands (Option B)

```bash
# Phase 1: NVIDIA (as root or sudo)
sudo dnf install kernel-headers-$(uname -r | sed 's/+64k//g') kernel-devel-matched-$(uname -r | sed 's/+64k//g') -y
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm -y
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo
sudo dnf clean expire-cache
sudo dnf install cuda-toolkit-13-0 -y
sudo dnf module install nvidia-driver:580-open -y
sudo systemctl enable nvidia-persistenced && sudo reboot

# Phase 2: Python + project (as user, after reboot)
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-aarch64-rpms
sudo dnf install python3.13 python3.13-pip python3.13-devel git -y
git clone https://github.com/pgarciaq/chant-omr.git && cd chant-omr
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"

# Phase 3: Dataset (transfer from laptop — much faster than re-downloading)
# Run this on your LAPTOP, not the GH200:
rsync -avz --progress data/gregobase/ data/rendered/ data/tokenizer/ you@gh200:~/chant-omr/data/

# Phase 4: Train (on the GH200)
tmux new -s train
source .venv/bin/activate
python scripts/train.py --accelerator cuda --precision bf16-mixed --batch-size 8 --epochs 50
```

## Quick reference: full sequence of commands (Option A — QuickPod)

```bash
# On the pod (SSH in first — replace INSTANCE_IP from QuickPod dashboard)
ssh root@INSTANCE_IP -p 34200
git clone https://github.com/pgarciaq/chant-omr.git && cd chant-omr

# Install Python 3.13 (QuickPod ships 3.10 which is too old)
apt update && apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y && apt update
apt install python3.13 python3.13-venv python3.13-dev -y

# Create venv and install deps
python3.13 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"

# On your LAPTOP (transfer dataset):
rsync -avz --progress -e 'ssh -p 34200' \
  data/gregobase/ data/rendered/ data/tokenizer/ root@INSTANCE_IP:~/chant-omr/data/

# On the pod (set num_workers: 0 if not already, then train):
sed -i 's/num_workers: [0-9]*/num_workers: 0/' configs/default.yaml
tmux new -s train
source .venv/bin/activate
python scripts/train.py --accelerator cuda --precision bf16-mixed --epochs 50

# On your LAPTOP (copy best checkpoint back, then TERMINATE the pod):
scp -P 34200 root@INSTANCE_IP:~/chant-omr/checkpoints/chant-omr-epoch=XX-val_loss=X.XXXX.ckpt \
  ~/dev/lpacleaner/chant-omr/checkpoints/
```
