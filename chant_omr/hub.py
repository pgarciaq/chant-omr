"""HuggingFace Hub upload and download for ChantOMR models (#39)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from chant_omr import __version__

# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

_MODEL_CARD_TEMPLATE = """\
---
library_name: chant-omr
tags:
  - omr
  - gregorian-chant
  - gabc
  - openvino
  - onnx
  - square-notation
license: mit
pipeline_tag: image-to-text
---

# ChantOMR

End-to-end Optical Music Recognition for Gregorian chant square notation.
Converts photographs of historical chant manuscripts into
[GABC](https://gregorio-project.github.io/gabc/) notation.

Based on [Transcoda](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1)'s
ConvNeXt-V2 + Transformer architecture (~59M params), retrained from scratch
on ~20,000 [GregoBase](https://gregobase.selapa.net/) scores with domain
augmentation for square notation.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ConvNeXt-V2 Tiny encoder + 8-layer Transformer decoder |
| Parameters | ~59M |
| Input | Score image (width 1050, variable height) |
| Output | GABC token sequence |
| Vocabulary | ~2048 BPE tokens |
| Training data | GregoBase (~20k scores, synthetic renders + augmentation) |
| Framework | PyTorch / Lightning |

## Evaluation

{eval_section}

## Formats

This repository contains the model in multiple formats:

| Format | Files | Use case |
|--------|-------|----------|
| **Safetensors** | `model.safetensors` | PyTorch fine-tuning or inference |
| **OpenVINO IR** | `openvino/*.xml` + `.bin` | Production inference on Intel hardware |
| **ONNX** | `onnx/*.onnx` | Portable inference on any hardware |

## Quick Start

### CLI (easiest)

```bash
pip install chant-omr
chant-omr predict score.png --model {repo_id} --device openvino
```

### Python (OpenVINO)

```python
from chant_omr.hub import download_from_hub
from chant_omr.inference.ov_decode import load_openvino_models, ov_predict_gabc

model_dir = download_from_hub("{repo_id}")
gabc = ov_predict_gabc(
    "score.png",
    model_dir / "openvino",
    beam_width=3,
)
print(gabc)
```

### Python (PyTorch)

```python
from chant_omr.hub import download_from_hub
from chant_omr.inference.checkpoint import load_model_from_safetensors
from chant_omr.inference.predict import predict_gabc_from_hub

model_dir = download_from_hub("{repo_id}")
gabc = predict_gabc_from_hub("score.png", model_dir)
print(gabc)
```

## Limitations

- Trained on synthetic Gregorio renders; real manuscript accuracy depends on
  scan quality and augmentation coverage.
- Square notation only (no modern staff notation, no NABC/adiastematic neumes).
- Single-system images work best; multi-system page layout analysis is handled
  by [ghh](https://github.com/pgarciaq/ghh), not this model.

## Links

- **Code:** [github.com/pgarciaq/chant-omr](https://github.com/pgarciaq/chant-omr)
- **Pipeline:** [github.com/pgarciaq/ghh](https://github.com/pgarciaq/ghh)
- **Training data:** [GregoBase](https://gregobase.selapa.net/)

---
*Uploaded with chant-omr v{version}*
"""


def generate_model_card(
    repo_id: str,
    *,
    eval_results: dict[str, Any] | None = None,
) -> str:
    """Generate a HuggingFace model card README.md string."""
    if eval_results:
        lines = []
        for key, value in eval_results.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        eval_section = "| Metric | Value |\n|--------|-------|\n" + "\n".join(lines)
    else:
        eval_section = "No evaluation results available yet."

    return _MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        version=__version__,
        eval_section=eval_section,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_to_hub(
    checkpoint_path: Path,
    repo_id: str,
    *,
    config_path: Path | None = None,
    eval_results: dict[str, Any] | None = None,
    private: bool = False,
) -> str:
    """Export all formats and upload to HuggingFace Hub.

    Returns the URL of the uploaded repository.
    """
    from huggingface_hub import HfApi

    from chant_omr.inference.export import (
        export_decoder_init_openvino,
        export_decoder_openvino,
        export_decoder_step_openvino,
        export_onnx,
        export_openvino,
        export_safetensors,
    )

    checkpoint_path = Path(checkpoint_path)
    cfg_path = Path(config_path or "configs/default.yaml")

    with tempfile.TemporaryDirectory(prefix="chant-omr-hub-") as tmp:
        staging = Path(tmp)

        # --- Safetensors ---
        export_safetensors(checkpoint_path, staging, config_path=cfg_path)

        # --- OpenVINO IR ---
        ov_dir = staging / "openvino"
        ov_dir.mkdir()
        export_openvino(checkpoint_path, ov_dir, config_path=cfg_path)
        export_decoder_openvino(checkpoint_path, ov_dir, config_path=cfg_path)
        export_decoder_init_openvino(checkpoint_path, ov_dir, config_path=cfg_path)
        export_decoder_step_openvino(checkpoint_path, ov_dir, config_path=cfg_path)

        # --- ONNX ---
        onnx_dir = staging / "onnx"
        export_onnx(checkpoint_path, onnx_dir, config_path=cfg_path)

        # --- Config ---
        shutil.copy2(str(cfg_path), str(staging / "config.yaml"))

        # --- Tokenizer (copy from staging root if safetensors export put it there) ---
        tok_in_root = staging / "tokenizer.json"
        if not tok_in_root.exists():
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            tok_dir = Path(cfg.get("data", {}).get("tokenizer_dir", "data/tokenizer"))
            tok_src = tok_dir / "tokenizer.json"
            if tok_src.exists():
                shutil.copy2(str(tok_src), str(tok_in_root))

        # --- Model card ---
        card = generate_model_card(repo_id, eval_results=eval_results)
        (staging / "README.md").write_text(card, encoding="utf-8")

        # --- Upload ---
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"chant-omr v{__version__}: upload model artifacts",
        )

    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
) -> Path:
    """Download a ChantOMR model from HuggingFace Hub.

    Uses ``huggingface_hub.snapshot_download`` with the standard HF cache
    (``~/.cache/huggingface/hub/``).  Subsequent calls with the same
    *repo_id* and *revision* are instant cache hits.

    Returns:
        Local directory path containing the downloaded model files.
    """
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {}
    if revision is not None:
        kwargs["revision"] = revision

    return Path(snapshot_download(repo_id, **kwargs))
