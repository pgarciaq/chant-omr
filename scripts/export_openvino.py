#!/usr/bin/env python3
"""Export trained model to OpenVINO IR for deployment in ghh.

Usage:
    python scripts/export_openvino.py --checkpoint checkpoints/best.ckpt --output models/
"""

import click


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), default="models/")
def main(checkpoint, output):
    """Export model to OpenVINO IR format."""
    click.echo(f"Exporting {checkpoint} → {output}")
    click.echo("Not yet implemented -- see chant_omr/inference/export.py")


if __name__ == "__main__":
    main()
