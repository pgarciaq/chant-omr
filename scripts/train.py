#!/usr/bin/env python3
"""Train the ChantOMR model."""

from __future__ import annotations

from pathlib import Path

import click

from chant_omr.training.lightning_module import run_training


@click.command()
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--gpus", type=int, default=1, help="GPU count (CUDA) or enable GPU (auto/XPU)")
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
    help="Training device: auto prefers CUDA, then Intel XPU, then CPU",
)
@click.option("--xpu-index", type=int, default=0, show_default=True, help="Intel XPU device index")
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
@click.option(
    "--finetune",
    is_flag=True,
    default=False,
    help="Load --resume weights only (fresh optimizer/epochs; use for LR changes)",
)
def main(
    config: str,
    resume: str | None,
    gpus: int,
    accelerator: str,
    xpu_index: int,
    precision: str | None,
    batch_size: int | None,
    epochs: int | None,
    overfit_n: int | None,
    encoder_pretrained: bool | None,
    finetune: bool,
) -> None:
    """Train the ChantOMR model."""
    run_training(
        Path(config),
        resume=Path(resume) if resume else None,
        gpus=gpus,
        accelerator=accelerator,
        xpu_index=xpu_index,
        precision=precision,
        batch_size=batch_size,
        epochs=epochs,
        overfit_n=overfit_n,
        encoder_pretrained=encoder_pretrained,
        finetune=finetune,
    )


if __name__ == "__main__":
    main()
