#!/usr/bin/env python3
"""Train the ChantOMR model."""

from __future__ import annotations

from pathlib import Path

import click

from chant_omr.training.lightning_module import run_training


@click.command()
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--gpus", type=int, default=1)
@click.option("--precision", type=str, default=None, help="Override config precision")
@click.option("--batch-size", type=int, default=None, help="Override config batch size")
@click.option("--epochs", type=int, default=None, help="Override config epochs")
@click.option(
    "--overfit-n",
    type=int,
    default=None,
    help="Train on the first N train samples only (smoke test)",
)
@click.option(
    "--encoder-pretrained/--no-encoder-pretrained",
    default=None,
    help="Override config encoder_pretrained",
)
def main(
    config: str,
    resume: str | None,
    gpus: int,
    precision: str | None,
    batch_size: int | None,
    epochs: int | None,
    overfit_n: int | None,
    encoder_pretrained: bool | None,
) -> None:
    """Train the ChantOMR model."""
    run_training(
        Path(config),
        resume=Path(resume) if resume else None,
        gpus=gpus,
        precision=precision,
        batch_size=batch_size,
        epochs=epochs,
        overfit_n=overfit_n,
        encoder_pretrained=encoder_pretrained,
    )


if __name__ == "__main__":
    main()
