"""Export trained models for deployment.

Export targets:
    - OpenVINO IR encoder (#13b) and decoder step (#41): for ghh on Intel Arc GPU/NPU
    - Safetensors: full model weights for portable distribution
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from chant_omr.inference.checkpoint import load_config, load_model_from_checkpoint
from chant_omr.model.chant_omr_model import ChantOMR, ChantOMRConfig

EXPORT_CANVAS_HEIGHT = 1600
EXPORT_CANVAS_WIDTH = 1050
ENCODER_OUTPUT_STRIDE = 32


@dataclass(frozen=True)
class ExportManifest:
    """Metadata written alongside export artifacts."""

    format: str
    checkpoint_path: str
    config: dict[str, Any]
    canvas_height: int
    canvas_width: int
    encoder_stride: int
    encoder_patches: int
    d_model: int


class EncoderForExport(nn.Module):
    """Self-contained encoder path: backbone → pos-enc → flatten → projector.

    Wraps the three submodules so ``torch.onnx.export`` traces a single
    graph with ``pixel_values`` as the only input.  The fixed canvas size
    (``EXPORT_CANVAS_HEIGHT × EXPORT_CANVAS_WIDTH``) makes every operator
    static-shape, which OpenVINO can optimise aggressively.
    """

    def __init__(self, model: ChantOMR) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.positional_encoding = model.positional_encoding
        self.projector = model.projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(pixel_values)
        positioned = self.positional_encoding(encoder_output.feature_map)
        memory = positioned.flatten(2).transpose(1, 2).contiguous()
        return self.projector(memory)


class DecoderStepForExport(nn.Module):
    """Single decoder forward step with encoder attention mask.

    Inputs:
        input_ids        ``(1, T)``       int64
        encoder_memory   ``(1, N, D)``    float32
        encoder_mask     ``(1, N)``       float32   (1 = real, 0 = padding)

    Output:
        next_logits      ``(1, 1, vocab_size)`` float32

    All three spatial dimensions (``T``, ``N``) are dynamic axes so the same
    IR works across generation steps and image sizes.
    """

    def __init__(self, model: ChantOMR) -> None:
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.decoder(
            input_ids,
            encoder_memory,
            encoder_attention_mask=encoder_mask,
        )
        return logits[:, -1:, :]


def _build_dummy_input(
    height: int = EXPORT_CANVAS_HEIGHT,
    width: int = EXPORT_CANVAS_WIDTH,
) -> torch.Tensor:
    return torch.randn(1, 3, height, width)


def export_openvino(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    input_width: int = EXPORT_CANVAS_WIDTH,
    input_height: int = EXPORT_CANVAS_HEIGHT,
) -> Path:
    """Export the encoder path to OpenVINO IR format.

    Produces ``encoder.xml``, ``encoder.bin``, and ``manifest.json`` inside
    *output_dir*.  The exported graph covers:

        pixel_values (1, 3, H, W) → encoder_memory (1, N, d_model)

    where ``N = (H // 32) × (W // 32)`` and ``d_model`` comes from the
    model config.  At inference time the caller builds an
    ``encoder_attention_mask`` from the original image height to mark
    padded patches (see ``build_encoder_attention_mask`` in
    ``chant_omr.data.dataset``).

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination for ``.xml`` / ``.bin`` / manifest.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        input_width: Canvas width in pixels.
        input_height: Canvas height in pixels.

    Returns:
        Path to the exported ``.xml`` model file.
    """
    import openvino as ov

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    enc_module = EncoderForExport(model)
    enc_module.eval()
    dummy = _build_dummy_input(height=input_height, width=input_width)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "encoder.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                enc_module,
                dummy,
                str(onnx_path),
                input_names=["pixel_values"],
                output_names=["encoder_memory"],
                opset_version=18,
                dynamic_axes=None,
            )

        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))

    xml_path = output_dir / "encoder.xml"
    ov.save_model(ov_model, str(xml_path))

    num_patches = (input_height // ENCODER_OUTPUT_STRIDE) * (
        input_width // ENCODER_OUTPUT_STRIDE
    )
    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    manifest = ExportManifest(
        format="openvino",
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        config=asdict(chant_config),
        canvas_height=input_height,
        canvas_width=input_width,
        encoder_stride=ENCODER_OUTPUT_STRIDE,
        encoder_patches=num_patches,
        d_model=chant_config.d_model,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )

    return xml_path


def export_safetensors(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
) -> Path:
    """Export full model weights as safetensors.

    Writes ``model.safetensors`` and ``manifest.json`` to *output_dir*.

    Returns:
        Path to the ``.safetensors`` file.
    """
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    st_path = output_dir / "model.safetensors"
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(st_path))

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    manifest = ExportManifest(
        format="safetensors",
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        config=asdict(chant_config),
        canvas_height=EXPORT_CANVAS_HEIGHT,
        canvas_width=EXPORT_CANVAS_WIDTH,
        encoder_stride=ENCODER_OUTPUT_STRIDE,
        encoder_patches=(EXPORT_CANVAS_HEIGHT // ENCODER_OUTPUT_STRIDE)
        * (EXPORT_CANVAS_WIDTH // ENCODER_OUTPUT_STRIDE),
        d_model=chant_config.d_model,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )

    return st_path


def export_decoder_openvino(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    trace_seq_len: int = 8,
    trace_num_patches: int = 128,
) -> Path:
    """Export the decoder single-step graph to OpenVINO IR.

    Produces ``decoder.xml`` and ``decoder.bin`` inside *output_dir*.
    The graph has dynamic axes for both sequence length (``T``) and
    encoder patch count (``N``):

        input_ids      (1, T)           int64
        encoder_memory (1, N, d_model)  float32
        → next_logits  (1, 1, vocab_size) float32

    The Python beam loop calls this IR once per generation step.

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination for ``.xml`` / ``.bin``.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        trace_seq_len: Token count used for the tracing dummy input.
        trace_num_patches: Encoder patch count for the tracing dummy input.

    Returns:
        Path to the exported ``.xml`` model file.
    """
    import openvino as ov

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    dec_module = DecoderStepForExport(model)
    dec_module.eval()

    dummy_ids = torch.ones(1, trace_seq_len, dtype=torch.long)
    dummy_memory = torch.randn(1, trace_num_patches, chant_config.d_model)
    dummy_mask = torch.ones(1, trace_num_patches)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "decoder.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                dec_module,
                (dummy_ids, dummy_memory, dummy_mask),
                str(onnx_path),
                input_names=["input_ids", "encoder_memory", "encoder_mask"],
                output_names=["next_logits"],
                opset_version=18,
                dynamic_axes={
                    "input_ids": {1: "seq_len"},
                    "encoder_memory": {1: "num_patches"},
                    "encoder_mask": {1: "num_patches"},
                },
            )

        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))

    xml_path = output_dir / "decoder.xml"
    ov.save_model(ov_model, str(xml_path))
    return xml_path


def verify_openvino_parity(
    checkpoint_path: Path,
    xml_path: Path,
    *,
    config_path: Path | None = None,
    input_height: int = EXPORT_CANVAS_HEIGHT,
    input_width: int = EXPORT_CANVAS_WIDTH,
    atol: float = 2e-3,
) -> float:
    """Compare PyTorch encoder output with OpenVINO IR and return max abs diff.

    The default tolerance is 2e-3 — deep ConvNeXt-V2 graphs accumulate ~1e-3
    rounding when traced through ONNX → OpenVINO.

    Raises ``AssertionError`` if the difference exceeds *atol*.
    """
    import numpy as np
    import openvino as ov

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    enc_module = EncoderForExport(model)
    enc_module.eval()
    dummy = _build_dummy_input(height=input_height, width=input_width)

    with torch.inference_mode():
        pt_out = enc_module(dummy)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    ov_result = compiled(dummy.numpy())
    ov_out = torch.from_numpy(np.array(ov_result[0]))

    max_diff = float((pt_out - ov_out).abs().max())
    if max_diff > atol:
        raise AssertionError(
            f"OpenVINO parity check failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff


def verify_decoder_openvino_parity(
    checkpoint_path: Path,
    xml_path: Path,
    *,
    config_path: Path | None = None,
    seq_len: int = 8,
    num_patches: int = 128,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch decoder step with OpenVINO IR and return max abs diff.

    The 8-layer Transformer decoder accumulates ~2e-3 rounding through
    self-attention + cross-attention + FFN per layer.

    Raises ``AssertionError`` if the difference exceeds *atol*.
    """
    import numpy as np
    import openvino as ov

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device="cpu",
    )

    dec_module = DecoderStepForExport(model)
    dec_module.eval()

    dummy_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_memory = torch.randn(1, num_patches, model.config.d_model)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_out = dec_module(dummy_ids, dummy_memory, dummy_mask)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    ov_result = compiled({
        "input_ids": dummy_ids.numpy(),
        "encoder_memory": dummy_memory.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })
    ov_out = torch.from_numpy(np.array(ov_result[0]))

    max_diff = float((pt_out - ov_out).abs().max())
    if max_diff > atol:
        raise AssertionError(
            f"Decoder OpenVINO parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff
