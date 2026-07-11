#!/usr/bin/env python3
"""Render GABC files into score images using Gregorio + LuaLaTeX.

Prerequisites:
    sudo dnf install texlive-gregoriotex texlive-luatex texlive-libertinus-fonts \
      texlive-metapost poppler-utils

Usage:
    python scripts/render_dataset.py --gabc-dir data/gregobase/ --output data/rendered/
    python scripts/render_dataset.py --limit 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from chant_omr.data.renderer import render_corpus, toolchain_available


@click.command()
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output", type=click.Path(), default="data/rendered/")
@click.option("--limit", type=int, default=None, help="Max pending manifest entries to render")
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option(
    "--workers",
    type=int,
    default=0,
    show_default="auto",
    help="Parallel LuaLaTeX workers (0 = min(cpu, cap))",
)
@click.option("--force", is_flag=True, help="Re-render even when PNG already exists")
@click.option("--no-progress", is_flag=True, help="Disable the render progress bar")
@click.option(
    "--progress",
    is_flag=True,
    help="Force the progress bar even when stderr is not a TTY",
)
def main(gabc_dir, output, limit, dpi, workers, force, no_progress, progress):
    """Render manifest GABC entries to training images."""
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
        Path(output),
        limit=limit,
        dpi=dpi,
        workers=workers,
        force=force,
        show_progress=show_progress,
    )
    click.echo(
        f"Manifest ok: {stats.manifest_ok} | Attempted: {stats.attempted} | "
        f"Rendered: {stats.rendered} | Skipped: {stats.skipped} | Failed: {stats.failed}"
    )


if __name__ == "__main__":
    main()
