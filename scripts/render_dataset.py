#!/usr/bin/env python3
"""Render GABC files into score images using Gregorio + LuaLaTeX.

Prerequisites:
    sudo dnf install texlive-gregoriotex texlive-luatex poppler-utils

Usage:
    python scripts/render_dataset.py --gabc-dir data/gregobase/ --output data/rendered/
"""

import click


@click.command()
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output", type=click.Path(), default="data/rendered/")
@click.option("--dpi", type=int, default=300)
@click.option("--workers", type=int, default=4)
def main(gabc_dir, output, dpi, workers):
    """Render GABC files to training images."""
    click.echo(f"Rendering {gabc_dir} → {output} at {dpi} DPI with {workers} workers")
    click.echo("Not yet implemented -- see chant_omr/data/renderer.py")


if __name__ == "__main__":
    main()
