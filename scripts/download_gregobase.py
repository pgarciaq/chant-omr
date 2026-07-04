#!/usr/bin/env python3
"""Download GABC files from GregoBase for training data generation.

Usage:
    python scripts/download_gregobase.py --output data/gregobase/ [--limit 1000]
"""

import click


@click.command()
@click.option("--output", type=click.Path(), default="data/gregobase/", help="Output directory")
@click.option("--limit", type=int, default=None, help="Max chants to download")
def main(output, limit):
    """Download GABC corpus from GregoBase."""
    click.echo(f"Downloading GregoBase GABC files to {output}")
    click.echo("Not yet implemented -- see chant_omr/data/gregobase.py")


if __name__ == "__main__":
    main()
