#!/usr/bin/env python3
"""Train a BPE tokenizer on plain GABC bodies.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --gabc-dir data/gregobase/ --vocab-size 2048
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from chant_omr.model.tokenizer import DEFAULT_OUTPUT_DIR, DEFAULT_VOCAB_SIZE, train_tokenizer


def _load_vocab_size(config_path: Path | None) -> int:
    if config_path is None or not config_path.is_file():
        return DEFAULT_VOCAB_SIZE
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return int(config.get("model", {}).get("vocab_size", DEFAULT_VOCAB_SIZE))


@click.command()
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
@click.option("--vocab-size", type=int, default=None, help="Override config model.vocab_size")
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option(
    "--min-body-len",
    type=int,
    default=20,
    show_default=True,
    help="Skip GABC bodies shorter than this many characters",
)
@click.option(
    "--no-manifest",
    is_flag=True,
    help="Train on all plain .gabc files, ignoring manifest.json status",
)
def main(gabc_dir, output_dir, vocab_size, config, min_body_len, no_manifest):
    """Train and save a BPE tokenizer for GABC bodies."""
    target_vocab = vocab_size if vocab_size is not None else _load_vocab_size(Path(config))
    tokenizer = train_tokenizer(
        Path(gabc_dir),
        vocab_size=target_vocab,
        output_dir=Path(output_dir),
        min_body_len=min_body_len,
        use_manifest=not no_manifest,
    )
    click.echo(
        f"Trained tokenizer: vocab={tokenizer.vocab_size} | "
        f"pad={tokenizer.pad_id} bos={tokenizer.bos_id} "
        f"eos={tokenizer.eos_id} unk={tokenizer.unk_id}"
    )
    click.echo(f"Artifacts: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
