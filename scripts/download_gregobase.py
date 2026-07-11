#!/usr/bin/env python3
"""Download GABC files from GregoBase for training data generation.

Usage:
    python scripts/download_gregobase.py --output data/gregobase/ [--limit 50]
    python scripts/download_gregobase.py --sync --days 7
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chant_omr.data.gregobase import download_corpus


@click.command()
@click.option("--output", type=click.Path(), default="data/gregobase/", help="Output directory")
@click.option("--limit", type=int, default=None, help="Max catalog IDs without success to process")
@click.option("--sync", is_flag=True, help="Also refresh IDs from updates.php")
@click.option("--days", type=int, default=None, help="Days window for updates.php")
@click.option(
    "--rate-limit",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between download.php requests",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(output, limit, sync, days, rate_limit, verbose):
    """Download GABC corpus from GregoBase."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    stats = download_corpus(
        Path(output),
        limit=limit,
        sync=sync,
        sync_days=days,
        rate_limit=rate_limit,
    )
    click.echo(
        f"Catalog: {stats.catalog_count} | Attempted: {stats.attempted_ids} | "
        f"Downloaded: {stats.downloaded_files} | Skipped: {stats.skipped_files} | "
        f"Failed IDs: {stats.failed_ids}"
    )


if __name__ == "__main__":
    main()
