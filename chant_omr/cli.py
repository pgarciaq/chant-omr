"""CLI entry point for chant-omr training and inference."""

import click

from chant_omr import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Chant OMR -- train and run Gregorian chant recognition models."""


@main.command()
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from checkpoint")
@click.option("--gpus", type=int, default=1)
@click.option("--epochs", type=int, default=None, help="Override config epochs")
def train(config, resume, gpus, epochs):
    """Train the OMR model."""
    click.echo(f"Training with config: {config}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model", type=str, default="pgarciaq/chant-omr", help="Model path or HuggingFace ID")
@click.option("--device", type=str, default="auto")
@click.option("--output", type=click.Path(), default=None, help="Output GABC file path")
def predict(image_path, model, device, output):
    """Run OMR on a single image and output GABC."""
    click.echo(f"Predicting: {image_path}")


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["openvino", "onnx", "safetensors"]), default="openvino")
@click.option("--output-dir", type=click.Path(), default="models/")
def export(checkpoint, fmt, output_dir):
    """Export a trained model for inference."""
    click.echo(f"Exporting {checkpoint} to {fmt}")


@main.command()
@click.option("--output-dir", type=click.Path(), default="data/gregobase/")
@click.option("--limit", type=int, default=None, help="Max incomplete catalog IDs to process")
@click.option("--sync", is_flag=True, help="Also refresh IDs from updates.php")
@click.option("--days", type=int, default=None, help="Days window for updates.php")
@click.option(
    "--sync-limit",
    type=int,
    default=None,
    help="Max sync IDs to refresh (--sync only; does not cap --limit pending batch)",
)
@click.option(
    "--rate-limit",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between download.php requests",
)
@click.option("--no-progress", is_flag=True, help="Disable the download progress bar")
@click.option(
    "--progress",
    is_flag=True,
    help="Force the progress bar even when stderr is not a TTY",
)
def download(output_dir, limit, sync, days, sync_limit, rate_limit, no_progress, progress):
    """Download GABC files from GregoBase."""
    import sys
    from pathlib import Path

    from chant_omr.data.gregobase import download_corpus

    if no_progress and progress:
        raise click.ClickException("Use only one of --progress or --no-progress.")
    if progress:
        show_progress = True
    elif no_progress:
        show_progress = False
    else:
        show_progress = sys.stderr.isatty()

    if sync and limit is None:
        click.echo(
            "Warning: no --limit set — after sync IDs, ALL remaining catalog IDs "
            "will be downloaded (~20k at 1 req/s ≈ 5.5 h). "
            "Use --limit 500 for phased runs or --sync-limit to cap sync only.",
            err=True,
        )

    stats = download_corpus(
        Path(output_dir),
        limit=limit,
        sync=sync,
        sync_days=days,
        sync_limit=sync_limit,
        rate_limit=rate_limit,
        show_progress=show_progress,
    )
    click.echo(
        f"Catalog: {stats.catalog_count} | Attempted: {stats.attempted_ids} | "
        f"Downloaded: {stats.downloaded_files} | Skipped: {stats.skipped_files} | "
        f"Failed IDs: {stats.failed_ids}"
    )


@main.command()
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output-dir", type=click.Path(), default="data/rendered/")
@click.option("--limit", type=int, default=None, help="Max pending manifest entries to render")
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option(
    "--workers",
    type=int,
    default=0,
    show_default="auto",
    help="Parallel LuaLaTeX workers (0 = min(cpu, cap); cap via CHANT_OMR_RENDER_WORKERS_MAX)",
)
@click.option("--force", is_flag=True, help="Re-render even when PNG already exists")
@click.option("--no-progress", is_flag=True, help="Disable the render progress bar")
@click.option(
    "--progress",
    is_flag=True,
    help="Force the progress bar even when stderr is not a TTY",
)
def render(gabc_dir, output_dir, limit, dpi, workers, force, no_progress, progress):
    """Render GABC files into score images using Gregorio."""
    import sys
    from pathlib import Path

    from chant_omr.data.renderer import render_corpus, resolve_render_workers, toolchain_available

    if no_progress and progress:
        raise click.ClickException("Use only one of --progress or --no-progress.")
    if progress:
        show_progress = True
    elif no_progress:
        show_progress = False
    else:
        show_progress = sys.stderr.isatty()

    if not toolchain_available():
        raise click.ClickException(
            "Gregorio toolchain not found. Install gregorio, lualatex, and pdftoppm."
        )

    stats = render_corpus(
        Path(gabc_dir),
        Path(output_dir),
        limit=limit,
        dpi=dpi,
        workers=workers,
        force=force,
        show_progress=show_progress,
    )
    worker_count = resolve_render_workers(workers)
    click.echo(
        f"Manifest ok: {stats.manifest_ok} | Attempted: {stats.attempted} | "
        f"Rendered: {stats.rendered} | Skipped: {stats.skipped} | Failed: {stats.failed} | "
        f"Workers: {worker_count}"
    )


@main.command("train-tokenizer")
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output-dir", type=click.Path(), default="data/tokenizer/")
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
def train_tokenizer_cmd(gabc_dir, output_dir, vocab_size, config, min_body_len, no_manifest):
    """Train a BPE tokenizer on plain GABC bodies."""
    from pathlib import Path

    import yaml

    from chant_omr.model.tokenizer import DEFAULT_VOCAB_SIZE, train_tokenizer

    config_path = Path(config)
    target_vocab = vocab_size
    if target_vocab is None and config_path.is_file():
        with config_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        target_vocab = int(cfg.get("model", {}).get("vocab_size", DEFAULT_VOCAB_SIZE))
    if target_vocab is None:
        target_vocab = DEFAULT_VOCAB_SIZE

    tokenizer = train_tokenizer(
        Path(gabc_dir),
        vocab_size=target_vocab,
        output_dir=Path(output_dir),
        min_body_len=min_body_len,
        use_manifest=not no_manifest,
    )
    click.echo(
        f"Corpus trained | vocab={tokenizer.vocab_size} | "
        f"pad={tokenizer.pad_id} bos={tokenizer.bos_id} "
        f"eos={tokenizer.eos_id} unk={tokenizer.unk_id}"
    )
    click.echo(f"Artifacts: {Path(output_dir).resolve()}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--test-dir", type=click.Path(exists=True), default="benchmarks/")
@click.option("--beam-width", type=int, default=3)
def evaluate(model_path, test_dir, beam_width):
    """Evaluate model on benchmark data."""
    click.echo(f"Evaluating {model_path} on {test_dir}")
