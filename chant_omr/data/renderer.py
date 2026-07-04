"""Render GABC files into score images using Gregorio + LuaLaTeX.

Gregorio is a TeX package that typesets Gregorian chant from GABC notation.
This module wraps the rendering pipeline:

    GABC file → Gregorio → .gtex → LuaLaTeX → PDF → image (PNG)

Requirements:
    - texlive with gregorio package (texlive-music on Fedora)
    - lualatex
    - poppler-utils (pdftoppm for PDF→PNG conversion)
"""

from __future__ import annotations

from pathlib import Path

LATEX_TEMPLATE = r"""
\documentclass[12pt]{{article}}
\usepackage{{fullpage}}
\usepackage[autocompile]{{gregoriotex}}

\pagestyle{{empty}}

\begin{{document}}
\gregorioscore{{{gtex_path}}}
\end{{document}}
"""


def render_gabc_to_image(
    gabc_path: Path,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Render a GABC file to a PNG image via Gregorio + LuaLaTeX.

    Args:
        gabc_path: Path to .gabc input file.
        output_path: Path for the output PNG image.
        dpi: Resolution for PDF-to-image conversion.

    Returns:
        Path to the rendered PNG image.
    """
    raise NotImplementedError("Gregorio rendering not yet implemented")


def render_batch(
    gabc_dir: Path,
    output_dir: Path,
    dpi: int = 300,
    workers: int = 4,
) -> list[Path]:
    """Render all GABC files in a directory to images.

    Args:
        gabc_dir: Directory containing .gabc files.
        output_dir: Directory for output PNG images.
        dpi: Resolution for rendering.
        workers: Number of parallel rendering processes.

    Returns:
        List of paths to rendered images.
    """
    raise NotImplementedError("Batch rendering not yet implemented")
