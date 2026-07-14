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
@click.option("--gpus", type=int, default=1, help="GPU count (CUDA) or enable GPU (auto/XPU)")
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
    help="Training device: auto prefers CUDA, then Intel XPU, then CPU",
)
@click.option("--xpu-index", type=int, default=0, show_default=True, help="Intel XPU device index")
@click.option("--epochs", type=int, default=None, help="Override config epochs")
@click.option("--batch-size", type=int, default=None, help="Override config batch size")
@click.option("--precision", type=str, default=None, help="Override config precision")
@click.option("--overfit-n", type=int, default=None, help="Train on first N samples (smoke test)")
def train(config, resume, gpus, accelerator, xpu_index, epochs, batch_size, precision, overfit_n):
    """Train the OMR model."""
    from pathlib import Path

    from chant_omr.training.lightning_module import run_training

    run_training(
        Path(config),
        resume=Path(resume) if resume else None,
        gpus=gpus,
        accelerator=accelerator,
        xpu_index=xpu_index,
        epochs=epochs,
        batch_size=batch_size,
        precision=precision,
        overfit_n=overfit_n,
    )


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True),
    required=True,
    help="Lightning .ckpt checkpoint path",
)
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--xpu-index", type=int, default=0, show_default=True)
@click.option("--beam-width", type=int, default=None, help="Override config inference.beam_width")
@click.option("--max-length", type=int, default=None, help="Override config inference.max_length")
@click.option(
    "--repetition-penalty",
    type=float,
    default=None,
    help="Override config inference.repetition_penalty",
)
@click.option("--name", type=str, default=None, help="GABC name: header (default: OMR output)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output GABC file path")
@click.option(
    "--dump-metrics",
    is_flag=True,
    default=False,
    help="Print teacher-forcing vs greedy diagnostics (uses sidecar .gabc when present)",
)
def predict(
    image_path,
    checkpoint_path,
    config,
    device,
    xpu_index,
    beam_width,
    max_length,
    repetition_penalty,
    name,
    output,
    dump_metrics,
):
    """Run OMR on a single image and output GABC."""
    from pathlib import Path

    import yaml

    from chant_omr.inference.predict import predict_gabc

    cfg_path = Path(config)
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    infer_cfg = cfg.get("inference", {})

    gabc = predict_gabc(
        Path(image_path),
        Path(checkpoint_path),
        config_path=cfg_path,
        device=device,
        xpu_index=xpu_index,
        beam_width=int(beam_width if beam_width is not None else infer_cfg.get("beam_width", 3)),
        max_length=int(max_length if max_length is not None else infer_cfg.get("max_length", 2048)),
        repetition_penalty=float(
            repetition_penalty
            if repetition_penalty is not None
            else infer_cfg.get("repetition_penalty", 1.1)
        ),
        name=name,
        dump_metrics=dump_metrics,
    )
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(gabc, encoding="utf-8")
        click.echo(f"Wrote {out_path}")
    else:
        click.echo(gabc)


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["openvino", "safetensors"]),
    default="openvino",
)
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--output-dir", type=click.Path(), default="models/")
@click.option("--verify", is_flag=True, help="Run parity check after OpenVINO export")
def export(checkpoint, fmt, config, output_dir, verify):
    """Export encoder + decoder (OpenVINO IR) or full weights (safetensors)."""
    from pathlib import Path

    from chant_omr.inference.export import (
        export_decoder_openvino,
        export_openvino,
        export_safetensors,
        verify_decoder_openvino_parity,
        verify_openvino_parity,
    )

    ckpt = Path(checkpoint)
    cfg = Path(config)
    out = Path(output_dir)

    if fmt == "openvino":
        enc_xml = export_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Encoder IR: {enc_xml}")
        dec_xml = export_decoder_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Decoder IR: {dec_xml}")
        click.echo(f"OpenVINO IR written to {out}/")
        if verify:
            enc_diff = verify_openvino_parity(ckpt, enc_xml, config_path=cfg)
            click.echo(f"Encoder parity passed (max abs diff: {enc_diff:.6e})")
            dec_diff = verify_decoder_openvino_parity(
                ckpt, dec_xml, config_path=cfg,
            )
            click.echo(f"Decoder parity passed (max abs diff: {dec_diff:.6e})")
    elif fmt == "safetensors":
        st = export_safetensors(ckpt, out, config_path=cfg)
        click.echo(f"Safetensors written to {st}")


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
    pct = f" ({stats.success_rate:.0%})" if stats.attempted else ""
    click.echo(
        f"Manifest ok: {stats.manifest_ok} | Attempted: {stats.attempted} | "
        f"Rendered: {stats.rendered}{pct} | Skipped: {stats.skipped} | "
        f"Failed: {stats.failed} | Workers: {worker_count}"
    )
    if stats.skipped_nabc or stats.failed_missing or stats.failed_compile:
        click.echo(
            f"  NABC skipped: {stats.skipped_nabc} | "
            f"Missing GABC: {stats.failed_missing} | "
            f"Compile errors: {stats.failed_compile}"
        )


@main.command()
@click.option("--rendered-dir", type=click.Path(exists=True), default="data/rendered/")
@click.option("--dry-run/--no-dry-run", default=True, show_default=True,
              help="List orphans without deleting (default). Use --no-dry-run to delete.")
def cleanup(rendered_dir, dry_run):
    """Remove orphan files from the rendered directory."""
    from pathlib import Path

    from chant_omr.data.renderer import cleanup_rendered_dir

    stats = cleanup_rendered_dir(Path(rendered_dir), dry_run=dry_run)
    mode = "DRY RUN" if dry_run else "DELETED"
    click.echo(f"Orphan .gabc (no matching .png): {stats.orphan_gabc_deleted} [{mode}]")
    if stats.png_only_orphans:
        click.echo(
            f"PNG-only orphans (no .gabc sidecar): {stats.png_only_orphans} "
            f"[use --force re-render to backfill]"
        )
    if dry_run and stats.orphan_gabc_deleted:
        click.echo("Re-run with --no-dry-run to delete orphan .gabc files.")


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


@main.command("audit-tokens")
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--rendered-dir", type=click.Path(exists=True), default=None)
@click.option("--top-n", type=int, default=10, show_default=True, help="Show N longest samples")
def audit_tokens(config, rendered_dir, top_n):
    """Report token-length distribution over the rendered corpus (#33)."""
    from pathlib import Path

    import yaml

    from chant_omr.data.token_audit import audit_token_lengths, format_token_audit
    from chant_omr.model.tokenizer import TOKENIZER_FILENAME, GABCTokenizer

    cfg_path = Path(config)
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    tok_dir = Path(data_cfg.get("tokenizer_dir", "data/tokenizer/"))
    tokenizer = GABCTokenizer.load(tok_dir / TOKENIZER_FILENAME)
    rdir = Path(rendered_dir or data_cfg.get("rendered_dir", "data/rendered/"))
    max_seq_len = int(model_cfg.get("max_seq_len", 2048))

    report = audit_token_lengths(
        rdir,
        tokenizer,
        max_seq_len=max_seq_len,
        top_n=top_n,
    )
    click.echo(format_token_audit(report))


@main.group()
def manifest():
    """Manifest management commands."""


@manifest.command()
@click.option("--output-dir", type=click.Path(exists=True), default="data/gregobase/",
              help="Directory with .gabc files and manifest.json")
def rebuild(output_dir):
    """Rebuild manifest.json from existing .gabc files on disk (#16)."""
    from pathlib import Path

    from chant_omr.data.gregobase import (
        fetch_catalog,
        make_session,
        rebuild_manifest,
    )

    out = Path(output_dir)
    click.echo("Fetching catalog from csv.php ...")
    session = make_session()
    catalog, catalog_date = fetch_catalog(session)
    click.echo(f"Catalog: {len(catalog)} chants"
               + (f" (snapshot {catalog_date})" if catalog_date else ""))

    mf, stats = rebuild_manifest(out, catalog, catalog_date=catalog_date)
    click.echo(
        f"Matched: {stats.matched} | Skipped: {stats.skipped} | "
        f"Ambiguous: {stats.ambiguous} | No match: {stats.no_match} | "
        f"Invalid: {stats.invalid} | Total files: {stats.total_files}"
    )
    if stats.skipped:
        click.echo(f"Unmatched files logged to {out / 'rebuild-unmatched.txt'}")
    click.echo(f"Manifest written with {len(mf.entries)} entries.")


main.add_command(manifest)


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--benchmark-dir",
    type=click.Path(exists=True),
    default="benchmarks/",
    help="Directory with (image, gabc) pairs — benchmarks/ or rendered test split",
)
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--beam-width", type=int, default=3, show_default=True)
@click.option("--max-length", type=int, default=2048, show_default=True)
@click.option(
    "--repetition-penalty", type=float, default=1.1, show_default=True,
)
@click.option("--limit", type=int, default=None, help="Evaluate only first N pairs")
@click.option(
    "--test-split-only",
    is_flag=True,
    default=False,
    help="Only evaluate test-split samples (catalog_id %% 20 == 0)",
)
def evaluate(checkpoint, benchmark_dir, config, device, beam_width, max_length,
             repetition_penalty, limit, test_split_only):
    """Evaluate model on benchmark (image, GABC) pairs (#14)."""
    from pathlib import Path

    from chant_omr.evaluation.evaluate import (
        evaluate_checkpoint,
        format_eval_report,
    )

    report = evaluate_checkpoint(
        Path(checkpoint),
        Path(benchmark_dir),
        config_path=Path(config),
        device=device,
        beam_width=beam_width,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        limit=limit,
        test_split_only=test_split_only,
    )
    click.echo(format_eval_report(report))
