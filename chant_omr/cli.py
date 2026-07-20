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
    type=click.Choice(["auto", "cuda", "xpu", "cpu", "onnx", "openvino"]),
    default="auto",
    show_default=True,
    help="Inference device: auto/cuda/xpu/cpu use PyTorch; onnx/openvino use exported models",
)
@click.option("--xpu-index", type=int, default=0, show_default=True)
@click.option(
    "--model-dir",
    type=click.Path(),
    default="models/",
    show_default=True,
    help="Exported model directory (used with --device onnx or openvino)",
)
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
@click.option(
    "--grammar-constrained/--no-grammar-constrained",
    default=None,
    help="Override inference.grammar_constrained (balanced-paren mask)",
)
@click.option(
    "--grammar-penalty",
    type=float,
    default=None,
    help="Override inference.grammar_penalty (-inf = hard mask, e.g. -10.0 = soft)",
)
def predict(
    image_path,
    checkpoint_path,
    config,
    device,
    xpu_index,
    model_dir,
    beam_width,
    max_length,
    repetition_penalty,
    name,
    output,
    dump_metrics,
    grammar_constrained,
    grammar_penalty,
):
    """Run OMR on a single image and output GABC."""
    from pathlib import Path

    import yaml

    cfg_path = Path(config)
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    infer_cfg = cfg.get("inference", {})

    gc = grammar_constrained
    if gc is None:
        gc = bool(infer_cfg.get("grammar_constrained", False))

    gp_raw = infer_cfg.get("grammar_penalty", float("-inf"))
    gp = grammar_penalty if grammar_penalty is not None else float(gp_raw)

    bw = int(beam_width if beam_width is not None else infer_cfg.get("beam_width", 3))
    ml = int(max_length if max_length is not None else infer_cfg.get("max_length", 8192))
    rp = float(
        repetition_penalty
        if repetition_penalty is not None
        else infer_cfg.get("repetition_penalty", 1.1)
    )

    if device == "onnx":
        from chant_omr.inference.onnx_decode import onnx_predict_gabc

        gabc = onnx_predict_gabc(
            Path(image_path),
            Path(model_dir),
            beam_width=bw,
            max_length=ml,
            repetition_penalty=rp,
            grammar_constrained=gc,
            grammar_penalty=gp,
            name=name,
        )
    elif device == "openvino":
        from chant_omr.inference.ov_decode import ov_predict_gabc

        gabc = ov_predict_gabc(
            Path(image_path),
            Path(model_dir),
            beam_width=bw,
            max_length=ml,
            repetition_penalty=rp,
            grammar_constrained=gc,
            grammar_penalty=gp,
            name=name,
        )
    else:
        from chant_omr.inference.predict import predict_gabc

        gabc = predict_gabc(
            Path(image_path),
            Path(checkpoint_path),
            config_path=cfg_path,
            device=device,
            xpu_index=xpu_index,
            beam_width=bw,
            max_length=ml,
            repetition_penalty=rp,
            grammar_constrained=gc,
            grammar_penalty=gp,
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
    type=click.Choice(["onnx", "openvino", "safetensors"]),
    default="onnx",
)
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option("--output-dir", type=click.Path(), default="models/")
@click.option("--verify", is_flag=True, help="Run parity check after export")
def export(checkpoint, fmt, config, output_dir, verify):
    """Export model to ONNX (with KV cache), OpenVINO IR, or safetensors."""
    from pathlib import Path

    ckpt = Path(checkpoint)
    cfg = Path(config)
    out = Path(output_dir)

    if fmt == "onnx":
        from chant_omr.inference.export import (
            export_onnx,
            verify_onnx_decoder_init_parity,
            verify_onnx_decoder_parity,
            verify_onnx_decoder_step_parity,
            verify_onnx_encoder_parity,
        )

        result_dir = export_onnx(ckpt, out, config_path=cfg)
        click.echo(f"ONNX models written to {result_dir}/")
        click.echo("  encoder.onnx")
        click.echo("  decoder.onnx (non-cached, beam search)")
        click.echo("  decoder_init.onnx (cached, greedy)")
        click.echo("  decoder_step.onnx (cached, greedy)")
        if verify:
            enc_diff = verify_onnx_encoder_parity(
                ckpt, result_dir / "encoder.onnx", config_path=cfg,
            )
            click.echo(f"Encoder parity passed (max abs diff: {enc_diff:.6e})")
            dec_diff = verify_onnx_decoder_parity(
                ckpt, result_dir / "decoder.onnx", config_path=cfg,
            )
            click.echo(f"Decoder parity passed (max abs diff: {dec_diff:.6e})")
            init_diff = verify_onnx_decoder_init_parity(
                ckpt, result_dir / "decoder_init.onnx", config_path=cfg,
            )
            click.echo(f"Decoder init parity passed (max abs diff: {init_diff:.6e})")
            step_diff = verify_onnx_decoder_step_parity(
                ckpt, result_dir / "decoder_step.onnx", config_path=cfg,
            )
            click.echo(f"Decoder step parity passed (max abs diff: {step_diff:.6e})")
    elif fmt == "openvino":
        from chant_omr.inference.export import (
            export_decoder_init_openvino,
            export_decoder_openvino,
            export_decoder_step_openvino,
            export_openvino,
            verify_decoder_init_openvino_parity,
            verify_decoder_openvino_parity,
            verify_decoder_step_openvino_parity,
            verify_openvino_parity,
        )

        enc_xml = export_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Encoder IR: {enc_xml}")
        dec_xml = export_decoder_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Decoder IR (non-cached): {dec_xml}")
        init_xml = export_decoder_init_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Decoder init IR (cached): {init_xml}")
        step_xml = export_decoder_step_openvino(ckpt, out, config_path=cfg)
        click.echo(f"Decoder step IR (cached): {step_xml}")
        click.echo(f"OpenVINO IR written to {out}/")
        if verify:
            enc_diff = verify_openvino_parity(ckpt, enc_xml, config_path=cfg)
            click.echo(f"Encoder parity passed (max abs diff: {enc_diff:.6e})")
            dec_diff = verify_decoder_openvino_parity(
                ckpt, dec_xml, config_path=cfg,
            )
            click.echo(f"Decoder parity passed (max abs diff: {dec_diff:.6e})")
            init_diff = verify_decoder_init_openvino_parity(
                ckpt, init_xml, config_path=cfg,
            )
            click.echo(f"Decoder init parity passed (max abs diff: {init_diff:.6e})")
            step_diff = verify_decoder_step_openvino_parity(
                ckpt, step_xml, config_path=cfg,
            )
            click.echo(f"Decoder step parity passed (max abs diff: {step_diff:.6e})")
    elif fmt == "safetensors":
        from chant_omr.inference.export import export_safetensors

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
@click.option(
    "--prefetch-plain-twins",
    is_flag=True,
    help="After download, auto-fetch plain GABC for NABC entries that have catalog twins",
)
def download(output_dir, limit, sync, days, sync_limit, rate_limit, no_progress, progress,
             prefetch_plain_twins):
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

    out = Path(output_dir)
    stats = download_corpus(
        out,
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

    if prefetch_plain_twins:
        from chant_omr.data.gregobase import (
            MANIFEST_FILENAME,
            Manifest,
            RateLimiter,
            fetch_catalog,
            make_session,
            prefetch_plain_twins as _prefetch,
            scan_nabc_ids,
        )

        manifest = Manifest.load(out / MANIFEST_FILENAME)
        click.echo("Scanning for NABC entries...", err=True)
        nabc_ids = scan_nabc_ids(out, manifest)
        click.echo(f"Found {len(nabc_ids)} NABC entries", err=True)
        if nabc_ids:
            session = make_session()
            catalog, _ = fetch_catalog(session)
            limiter = RateLimiter(rate_limit)
            pf_stats = _prefetch(session, out, manifest, catalog, limiter, nabc_ids)
            click.echo(
                f"Prefetch: {pf_stats.twins_found} twins found | "
                f"{pf_stats.downloaded} downloaded | "
                f"{pf_stats.already_present} already present | "
                f"{pf_stats.no_twin} no twin"
            )


@main.command("collapse-nabc")
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output-dir", type=click.Path(), default="data/nabc-derived/")
@click.option("--only-if-plain-missing/--all", default=True,
              help="Skip NABC files that already have a plain twin in the corpus")
def collapse_nabc(gabc_dir, output_dir, only_if_plain_missing):
    """Collapse NABC GABC files to plain GABC by stripping pipe annotations."""
    from pathlib import Path

    from chant_omr.data.gregobase import (
        MANIFEST_FILENAME,
        Manifest,
        fetch_catalog,
        make_session,
    )
    from chant_omr.data.nabc import collapse_nabc_corpus

    gdir = Path(gabc_dir)
    manifest = Manifest.load(gdir / MANIFEST_FILENAME)
    session = make_session()
    catalog, _ = fetch_catalog(session)

    stats = collapse_nabc_corpus(
        gdir, Path(output_dir), manifest, catalog,
        only_if_plain_missing=only_if_plain_missing,
    )
    click.echo(
        f"Collapsed: {stats.collapsed} | "
        f"Skipped (has twin): {stats.skipped_has_twin} | "
        f"Skipped (other): {stats.skipped_other}"
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
    click.echo(f"Orphan .png (no matching .gabc): {stats.orphan_png_deleted} [{mode}]")
    if dry_run and (stats.orphan_gabc_deleted or stats.orphan_png_deleted):
        click.echo("Re-run with --no-dry-run to delete orphan files.")


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
    max_seq_len = int(model_cfg.get("max_seq_len", 8192))

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
    default=None,
    help="Directory with (image, gabc) pairs [default: benchmarks/ or data/rendered/]",
)
@click.option("--config", type=click.Path(exists=True), default="configs/default.yaml")
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "xpu", "cpu"]),
    default="auto",
    show_default=True,
)
@click.option("--beam-width", type=int, default=3, show_default=True)
@click.option("--max-length", type=int, default=8192, show_default=True)
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
@click.option(
    "--grammar-constrained/--no-grammar-constrained",
    default=False,
    show_default=True,
    help="Enable balanced-paren grammar mask during decoding (#37)",
)
@click.option(
    "--grammar-penalty",
    type=float,
    default=None,
    help="Grammar penalty (-inf = hard mask, e.g. -10.0 = soft). Default from config.",
)
@click.option(
    "--gregorio-check",
    is_flag=True,
    default=False,
    help="Run gregorio compilation on each prediction for structural validity (#46)",
)
def evaluate(checkpoint, benchmark_dir, config, device, beam_width, max_length,
             repetition_penalty, limit, test_split_only, grammar_constrained,
             grammar_penalty, gregorio_check):
    """Evaluate model on benchmark (image, GABC) pairs (#14)."""
    from pathlib import Path

    from chant_omr.evaluation.evaluate import (
        evaluate_checkpoint,
        format_eval_report,
    )

    if benchmark_dir is None:
        for candidate in ("benchmarks/", "data/rendered/"):
            p = Path(candidate)
            if p.is_dir() and any(p.rglob("*.png")):
                benchmark_dir = candidate
                break
        if benchmark_dir is None:
            raise click.UsageError("No benchmark directory with .png files found. Use --benchmark-dir.")
    click.echo(f"Benchmark dir: {benchmark_dir}", err=True)

    eval_start = __import__("time").monotonic()

    def _progress(done, total, img_path, elapsed_s):
        wall = __import__("time").monotonic() - eval_start
        avg = wall / done
        eta = avg * (total - done)
        eta_min, eta_sec = divmod(int(eta), 60)
        click.echo(
            f"  [{done}/{total}] {img_path.name} — {elapsed_s:.1f}s "
            f"(ETA {eta_min}m{eta_sec:02d}s)",
            err=True,
        )

    if grammar_penalty is None:
        import yaml
        with Path(config).open(encoding="utf-8") as fh:
            _cfg = yaml.safe_load(fh) or {}
        grammar_penalty = float(_cfg.get("inference", {}).get("grammar_penalty", float("-inf")))

    if gregorio_check:
        from chant_omr.evaluation.gregorio_roundtrip import gregorio_available
        if not gregorio_available():
            raise click.UsageError(
                "--gregorio-check requires the gregorio binary (texlive-binaries)."
            )
        click.echo("Gregorio compilation check enabled", err=True)

    report = evaluate_checkpoint(
        Path(checkpoint),
        Path(benchmark_dir),
        config_path=Path(config),
        device=device,
        beam_width=beam_width,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_constrained=grammar_constrained,
        grammar_penalty=grammar_penalty,
        gregorio_check=gregorio_check,
        limit=limit,
        test_split_only=test_split_only,
        progress_callback=_progress,
    )
    click.echo(format_eval_report(report))
