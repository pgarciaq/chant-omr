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
@click.option("--limit", type=int, default=None, help="Max number of chants to download")
def download(output_dir, limit):
    """Download GABC files from GregoBase."""
    click.echo(f"Downloading GregoBase to {output_dir}")


@main.command()
@click.option("--gabc-dir", type=click.Path(exists=True), default="data/gregobase/")
@click.option("--output-dir", type=click.Path(), default="data/rendered/")
@click.option("--workers", type=int, default=4)
def render(gabc_dir, output_dir, workers):
    """Render GABC files into score images using Gregorio."""
    click.echo(f"Rendering {gabc_dir} -> {output_dir}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--test-dir", type=click.Path(exists=True), default="benchmarks/")
@click.option("--beam-width", type=int, default=3)
def evaluate(model_path, test_dir, beam_width):
    """Evaluate model on benchmark data."""
    click.echo(f"Evaluating {model_path} on {test_dir}")
