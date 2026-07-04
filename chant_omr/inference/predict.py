"""Run OMR inference on score images.

Supports multiple backends:
    - PyTorch (default, for GPU)
    - OpenVINO (optimized for Intel Arc GPU / NPU)
    - ONNX Runtime (portable)
"""

from __future__ import annotations

from pathlib import Path


def predict_gabc(
    image_path: Path,
    model_path: str = "pgarciaq/chant-omr",
    device: str = "auto",
    beam_width: int = 3,
    max_length: int = 2048,
) -> str:
    """Run OMR on a single image and return GABC notation.

    Args:
        image_path: Path to the score image.
        model_path: Local path or HuggingFace model ID.
        device: Device for inference ("auto", "cuda", "cpu", "xpu", "openvino").
        beam_width: Beam search width (1 = greedy).
        max_length: Maximum output sequence length.

    Returns:
        GABC notation string.
    """
    raise NotImplementedError("Inference not yet implemented")
