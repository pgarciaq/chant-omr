"""Download and manage GABC files from GregoBase.

GregoBase (https://gregobase.selapa.net/) is a community-maintained database of
~10,000 Gregorian chant transcriptions in GABC format. This module handles
downloading and organizing the corpus for training data generation.
"""

from __future__ import annotations

from pathlib import Path

GREGOBASE_API = "https://gregobase.selapa.net/api"
GREGOBASE_GABC_URL = "https://gregobase.selapa.net/chant.php?id={chant_id}"


def download_corpus(output_dir: Path, limit: int | None = None) -> list[Path]:
    """Download GABC files from GregoBase.

    Args:
        output_dir: Directory to save GABC files.
        limit: Maximum number of chants to download. None for all.

    Returns:
        List of paths to downloaded GABC files.
    """
    raise NotImplementedError("GregoBase download not yet implemented")
