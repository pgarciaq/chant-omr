#!/usr/bin/env python3
"""Train the ChantOMR model.

Usage:
    # Local (single GPU)
    python scripts/train.py --config configs/default.yaml

    # Cloud (RunPod/Lambda, A100)
    python scripts/train.py --config configs/default.yaml --gpus 1 --precision bf16-mixed

    # Resume from checkpoint
    python scripts/train.py --config configs/default.yaml --resume checkpoints/last.ckpt
"""

import click


@click.command()
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--gpus", type=int, default=1)
@click.option("--precision", type=str, default=None, help="Override config precision")
@click.option("--batch-size", type=int, default=None, help="Override config batch size")
@click.option("--epochs", type=int, default=None, help="Override config epochs")
def main(config, resume, gpus, precision, batch_size, epochs):
    """Train the ChantOMR model."""
    click.echo(f"Training with config: {config}")
    click.echo("Not yet implemented -- see chant_omr/training/lightning_module.py")


if __name__ == "__main__":
    main()
