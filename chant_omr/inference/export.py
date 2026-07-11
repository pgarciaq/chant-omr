"""Export trained models for deployment.

Export targets:
    - OpenVINO IR: for ghh on Intel Arc GPU/NPU
    - ONNX: portable inference
    - Safetensors: for HuggingFace distribution
"""

from __future__ import annotations

from pathlib import Path


def export_openvino(
    checkpoint_path: Path,
    output_dir: Path,
    input_width: int = 1050,
    input_height: int = 1600,
) -> Path:
    """Export model to OpenVINO IR format.

    The exported model can be loaded by ghh's Stage 13 for
    inference on Intel Arc GPU or NPU.

    Args:
        checkpoint_path: Path to trained .ckpt or .safetensors.
        output_dir: Directory for OpenVINO .xml and .bin files.
        input_width: Reference input width (height may be dynamic until #13).
        input_height: Reference max input height for export benchmarking.

    Returns:
        Path to the exported .xml model file.
    """
    raise NotImplementedError("OpenVINO export not yet implemented")


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_width: int = 1050,
    input_height: int = 1600,
) -> Path:
    """Export model to ONNX format."""
    raise NotImplementedError("ONNX export not yet implemented")
