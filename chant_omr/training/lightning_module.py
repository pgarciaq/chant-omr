"""PyTorch Lightning module for training the ChantOMR model.

Handles:
    - Training/validation step logic
    - Learning rate scheduling (cosine warmup)
    - Metric logging (loss, GABC edit distance, token accuracy)
    - Checkpoint management
"""

from __future__ import annotations


class ChantOMRLightningModule:
    """Lightning training module.

    Training recipe (following Transcoda):
        - Optimizer: AdamW, lr=1e-4, weight_decay=0.05
        - Scheduler: cosine with linear warmup (5% of total steps)
        - Gradient clipping: max_norm=1.0
        - Mixed precision: bf16 on A100/H100, fp16 on older GPUs
        - Batch size: 8-16 per GPU (adjust for VRAM)
        - Epochs: 50-100 (monitor validation loss plateau)
    """

    pass
