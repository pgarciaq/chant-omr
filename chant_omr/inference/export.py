"""Export trained models for deployment.

Export targets:
    - ONNX encoder + decoder with KV cache (#50): portable, any hardware
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
from chant_omr.model.decoder import LayerCache

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
        logits, _ = self.decoder(
            input_ids,
            encoder_memory,
            encoder_attention_mask=encoder_mask,
        )
        return logits[:, -1:, :]


class CachedDecoderInitForExport(nn.Module):
    """First decoder step with KV cache: computes cross-attention K/V.

    Inputs:
        input_ids        ``(1, 1)``       int64   (BOS token)
        encoder_memory   ``(1, N, D)``    float32
        encoder_mask     ``(1, N)``       float32 (1 = real, 0 = padding)

    Outputs:
        logits           ``(1, 1, V)``    float32
        self_k           ``(L, 1, H, 1, head_dim)``  float32
        self_v           ``(L, 1, H, 1, head_dim)``  float32
        cross_k          ``(L, 1, H, N, head_dim)``  float32
        cross_v          ``(L, 1, H, N, head_dim)``  float32
    """

    def __init__(self, model: ChantOMR) -> None:
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, caches = self.decoder(
            input_ids,
            encoder_memory,
            encoder_attention_mask=encoder_mask,
            use_cache=True,
        )
        assert caches is not None
        self_k = torch.stack([c.self_k for c in caches])
        self_v = torch.stack([c.self_v for c in caches])
        cross_k = torch.stack([c.cross_k for c in caches])
        cross_v = torch.stack([c.cross_v for c in caches])
        return logits[:, -1:, :], self_k, self_v, cross_k, cross_v


class CachedDecoderStepForExport(nn.Module):
    """Subsequent decoder steps with KV cache: reuses cached cross-attn K/V.

    Inputs:
        input_ids        ``(1, 1)``                   int64
        past_self_k      ``(L, 1, H, S, head_dim)``   float32
        past_self_v      ``(L, 1, H, S, head_dim)``   float32
        past_cross_k     ``(L, 1, H, N, head_dim)``   float32
        past_cross_v     ``(L, 1, H, N, head_dim)``   float32
        encoder_mask     ``(1, N)``                    float32

    Outputs:
        logits           ``(1, 1, V)``                 float32
        self_k           ``(L, 1, H, S+1, head_dim)``  float32
        self_v           ``(L, 1, H, S+1, head_dim)``  float32
        cross_k          ``(L, 1, H, N, head_dim)``    float32  (pass-through)
        cross_v          ``(L, 1, H, N, head_dim)``    float32  (pass-through)
    """

    def __init__(self, model: ChantOMR) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.d_model = model.config.d_model

    def forward(
        self,
        input_ids: torch.Tensor,
        past_self_k: torch.Tensor,
        past_self_v: torch.Tensor,
        past_cross_k: torch.Tensor,
        past_cross_v: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_layers = past_self_k.shape[0]
        past_key_values = [
            LayerCache(
                self_k=past_self_k[i],
                self_v=past_self_v[i],
                cross_k=past_cross_k[i],
                cross_v=past_cross_v[i],
            )
            for i in range(n_layers)
        ]

        dummy_memory = past_self_k.new_zeros(1, 1, self.d_model)

        logits, new_caches = self.decoder(
            input_ids,
            dummy_memory,
            encoder_attention_mask=encoder_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        assert new_caches is not None
        new_self_k = torch.stack([c.self_k for c in new_caches])
        new_self_v = torch.stack([c.self_v for c in new_caches])
        new_cross_k = torch.stack([c.cross_k for c in new_caches])
        new_cross_v = torch.stack([c.cross_v for c in new_caches])
        return logits[:, -1:, :], new_self_k, new_self_v, new_cross_k, new_cross_v


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

    Produces ``encoder.xml``, ``encoder.bin``, ``tokenizer.json``, and
    ``manifest.json`` inside *output_dir*.  The exported graph covers:

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
    import shutil

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

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    tok_dir = Path(cfg.get("data", {}).get("tokenizer_dir", "data/tokenizer"))
    tok_src = tok_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(str(tok_src), str(output_dir / "tokenizer.json"))

    num_patches = (input_height // ENCODER_OUTPUT_STRIDE) * (
        input_width // ENCODER_OUTPUT_STRIDE
    )
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


def export_onnx(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    input_width: int = EXPORT_CANVAS_WIDTH,
    trace_height: int = EXPORT_CANVAS_HEIGHT,
    trace_num_patches: int = 128,
) -> Path:
    """Export encoder + decoder (cached + non-cached) to ONNX format.

    Produces four ONNX models:

        encoder.onnx        pixel_values → encoder_memory
        decoder.onnx         non-cached: full input_ids → logits (for beam search)
        decoder_init.onnx    cached first step: input_ids + memory → logits + KV
        decoder_step.onnx    cached subsequent: input_ids + KV → logits + KV

    Also copies ``tokenizer.json`` and writes ``manifest.json``.

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination directory for ONNX files.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        input_width: Canvas width in pixels.
        trace_height: Height used for tracing the encoder dummy input.
        trace_num_patches: Encoder patch count for decoder dummy inputs.

    Returns:
        Path to the output directory.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    n_layers = chant_config.n_layers
    n_heads = chant_config.n_heads
    head_dim = chant_config.d_model // chant_config.n_heads

    # --- Encoder ---
    enc_module = EncoderForExport(model)
    enc_module.eval()
    dummy_img = _build_dummy_input(height=trace_height, width=input_width)

    encoder_path = output_dir / "encoder.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            enc_module,
            dummy_img,
            str(encoder_path),
            input_names=["pixel_values"],
            output_names=["encoder_memory"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={
                "pixel_values": {2: "height"},
                "encoder_memory": {1: "num_patches"},
            },
        )

    # --- Decoder init (first step) ---
    init_module = CachedDecoderInitForExport(model)
    init_module.eval()

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_memory = torch.randn(1, trace_num_patches, chant_config.d_model)
    dummy_mask = torch.ones(1, trace_num_patches)

    decoder_init_path = output_dir / "decoder_init.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            init_module,
            (dummy_ids, dummy_memory, dummy_mask),
            str(decoder_init_path),
            input_names=["input_ids", "encoder_memory", "encoder_mask"],
            output_names=["logits", "self_k", "self_v", "cross_k", "cross_v"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={
                "encoder_memory": {1: "num_patches"},
                "encoder_mask": {1: "num_patches"},
                "cross_k": {3: "num_patches"},
                "cross_v": {3: "num_patches"},
            },
        )

    # --- Decoder step (subsequent steps) ---
    step_module = CachedDecoderStepForExport(model)
    step_module.eval()

    dummy_self_k = torch.randn(n_layers, 1, n_heads, 1, head_dim)
    dummy_self_v = torch.randn(n_layers, 1, n_heads, 1, head_dim)
    dummy_cross_k = torch.randn(n_layers, 1, n_heads, trace_num_patches, head_dim)
    dummy_cross_v = torch.randn(n_layers, 1, n_heads, trace_num_patches, head_dim)

    decoder_step_path = output_dir / "decoder_step.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            step_module,
            (dummy_ids, dummy_self_k, dummy_self_v,
             dummy_cross_k, dummy_cross_v, dummy_mask),
            str(decoder_step_path),
            input_names=[
                "input_ids",
                "past_self_k", "past_self_v",
                "past_cross_k", "past_cross_v",
                "encoder_mask",
            ],
            output_names=["logits", "self_k", "self_v", "cross_k", "cross_v"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={
                "past_self_k": {3: "past_seq_len"},
                "past_self_v": {3: "past_seq_len"},
                "past_cross_k": {3: "num_patches"},
                "past_cross_v": {3: "num_patches"},
                "encoder_mask": {1: "num_patches"},
                "self_k": {3: "seq_len"},
                "self_v": {3: "seq_len"},
                "cross_k": {3: "num_patches"},
                "cross_v": {3: "num_patches"},
            },
        )

    # --- Decoder (non-cached, for beam search) ---
    # Traced last to avoid polluting the shared model's RoPE cache
    # before the cached decoder traces (which use seq_len=1).
    dec_module = DecoderStepForExport(model)
    dec_module.eval()

    dummy_dec_ids = torch.ones(1, 8, dtype=torch.long)
    dummy_dec_memory = torch.randn(1, trace_num_patches, chant_config.d_model)
    dummy_dec_mask = torch.ones(1, trace_num_patches)

    decoder_path = output_dir / "decoder.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            dec_module,
            (dummy_dec_ids, dummy_dec_memory, dummy_dec_mask),
            str(decoder_path),
            input_names=["input_ids", "encoder_memory", "encoder_mask"],
            output_names=["next_logits"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={
                "input_ids": {1: "seq_len"},
                "encoder_memory": {1: "num_patches"},
                "encoder_mask": {1: "num_patches"},
            },
        )

    # --- Tokenizer ---
    tok_dir = Path(cfg.get("data", {}).get("tokenizer_dir", "data/tokenizer"))
    tok_src = tok_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(str(tok_src), str(output_dir / "tokenizer.json"))

    # --- Manifest ---
    num_patches = (trace_height // ENCODER_OUTPUT_STRIDE) * (
        input_width // ENCODER_OUTPUT_STRIDE
    )
    manifest = ExportManifest(
        format="onnx",
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        config=asdict(chant_config),
        canvas_height=trace_height,
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

    return output_dir


def verify_onnx_encoder_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    config_path: Path | None = None,
    input_height: int = EXPORT_CANVAS_HEIGHT,
    input_width: int = EXPORT_CANVAS_WIDTH,
    atol: float = 2e-3,
) -> float:
    """Compare PyTorch encoder with ONNX Runtime and return max abs diff."""
    import numpy as np
    import onnxruntime as ort

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    enc_module = EncoderForExport(model)
    enc_module.eval()
    dummy = _build_dummy_input(height=input_height, width=input_width)

    with torch.inference_mode():
        pt_out = enc_module(dummy)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_result = session.run(None, {"pixel_values": dummy.numpy()})
    onnx_out = torch.from_numpy(np.array(ort_result[0]))

    max_diff = float((pt_out - onnx_out).abs().max())
    if max_diff > atol:
        raise AssertionError(
            f"ONNX encoder parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff


def verify_onnx_decoder_init_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    config_path: Path | None = None,
    num_patches: int = 128,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch decoder init step with ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    init_module = CachedDecoderInitForExport(model)
    init_module.eval()

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_memory = torch.randn(1, num_patches, model.config.d_model)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_outputs = init_module(dummy_ids, dummy_memory, dummy_mask)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {
        "input_ids": dummy_ids.numpy(),
        "encoder_memory": dummy_memory.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })

    max_diff = 0.0
    names = ["logits", "self_k", "self_v", "cross_k", "cross_v"]
    for i, name in enumerate(names):
        pt_t = pt_outputs[i]
        ort_t = torch.from_numpy(np.array(ort_outputs[i]))
        diff = float((pt_t - ort_t).abs().max())
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        raise AssertionError(
            f"ONNX decoder_init parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff


def verify_onnx_decoder_step_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    config_path: Path | None = None,
    num_patches: int = 128,
    past_seq_len: int = 3,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch decoder step with ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    step_module = CachedDecoderStepForExport(model)
    step_module.eval()

    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    head_dim = model.config.d_model // model.config.n_heads

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_self_k = torch.randn(n_layers, 1, n_heads, past_seq_len, head_dim)
    dummy_self_v = torch.randn(n_layers, 1, n_heads, past_seq_len, head_dim)
    dummy_cross_k = torch.randn(n_layers, 1, n_heads, num_patches, head_dim)
    dummy_cross_v = torch.randn(n_layers, 1, n_heads, num_patches, head_dim)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_outputs = step_module(
            dummy_ids, dummy_self_k, dummy_self_v,
            dummy_cross_k, dummy_cross_v, dummy_mask,
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {
        "input_ids": dummy_ids.numpy(),
        "past_self_k": dummy_self_k.numpy(),
        "past_self_v": dummy_self_v.numpy(),
        "past_cross_k": dummy_cross_k.numpy(),
        "past_cross_v": dummy_cross_v.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })

    max_diff = 0.0
    names = ["logits", "self_k", "self_v", "cross_k", "cross_v"]
    for i, name in enumerate(names):
        pt_t = pt_outputs[i]
        ort_t = torch.from_numpy(np.array(ort_outputs[i]))
        diff = float((pt_t - ort_t).abs().max())
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        raise AssertionError(
            f"ONNX decoder_step parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff


def export_decoder_onnx(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    trace_seq_len: int = 8,
    trace_num_patches: int = 128,
) -> Path:
    """Export the non-cached decoder to ONNX format.

    Produces ``decoder.onnx`` inside *output_dir*.  This graph processes
    the full token sequence each step (O(n^2) but supports beam search):

        input_ids (1,S) + encoder_memory (1,N,D) + encoder_mask (1,N)
        → next_logits (1,S,V)

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination for ``.onnx`` file.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        trace_seq_len: Token count used for the tracing dummy input.
        trace_num_patches: Encoder patch count for the tracing dummy input.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    dec_module = DecoderStepForExport(model)
    dec_module.eval()

    dummy_ids = torch.ones(1, trace_seq_len, dtype=torch.long)
    dummy_memory = torch.randn(1, trace_num_patches, chant_config.d_model)
    dummy_mask = torch.ones(1, trace_num_patches)

    decoder_path = output_dir / "decoder.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            dec_module,
            (dummy_ids, dummy_memory, dummy_mask),
            str(decoder_path),
            input_names=["input_ids", "encoder_memory", "encoder_mask"],
            output_names=["next_logits"],
            opset_version=18,
            dynamo=False,
            dynamic_axes={
                "input_ids": {1: "seq_len"},
                "encoder_memory": {1: "num_patches"},
                "encoder_mask": {1: "num_patches"},
            },
        )

    return decoder_path


def verify_onnx_decoder_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    config_path: Path | None = None,
    seq_len: int = 8,
    num_patches: int = 128,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch non-cached decoder with ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    dec_module = DecoderStepForExport(model)
    dec_module.eval()

    dummy_ids = torch.ones(1, seq_len, dtype=torch.long)
    dummy_memory = torch.randn(1, num_patches, model.config.d_model)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_out = dec_module(dummy_ids, dummy_memory, dummy_mask)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_result = session.run(None, {
        "input_ids": dummy_ids.numpy(),
        "encoder_memory": dummy_memory.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })
    onnx_out = torch.from_numpy(np.array(ort_result[0]))

    max_diff = float((pt_out - onnx_out).abs().max())
    if max_diff > atol:
        raise AssertionError(
            f"ONNX decoder parity failed: max abs diff {max_diff:.6f} > atol {atol}"
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


# ---------------------------------------------------------------------------
# KV-cached decoder OpenVINO export (#36)
# ---------------------------------------------------------------------------


def export_decoder_init_openvino(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    trace_num_patches: int = 128,
) -> Path:
    """Export the cached decoder init step to OpenVINO IR.

    Produces ``decoder_init.xml`` and ``decoder_init.bin`` inside *output_dir*.
    This graph computes cross-attention K/V from encoder memory on the first
    decoding step:

        input_ids (1,1) + encoder_memory (1,N,D) + encoder_mask (1,N)
        → logits (1,1,V) + self_k/v (L,1,H,1,hd) + cross_k/v (L,1,H,N,hd)

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination for ``.xml`` / ``.bin``.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        trace_num_patches: Encoder patch count for tracing dummy inputs.

    Returns:
        Path to the exported ``.xml`` model file.
    """
    import openvino as ov

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    init_module = CachedDecoderInitForExport(model)
    init_module.eval()

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_memory = torch.randn(1, trace_num_patches, chant_config.d_model)
    dummy_mask = torch.ones(1, trace_num_patches)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "decoder_init.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                init_module,
                (dummy_ids, dummy_memory, dummy_mask),
                str(onnx_path),
                input_names=["input_ids", "encoder_memory", "encoder_mask"],
                output_names=[
                    "logits", "self_k", "self_v", "cross_k", "cross_v",
                ],
                opset_version=18,
                dynamo=False,
                dynamic_axes={
                    "encoder_memory": {1: "num_patches"},
                    "encoder_mask": {1: "num_patches"},
                    "cross_k": {3: "num_patches"},
                    "cross_v": {3: "num_patches"},
                },
            )

        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))

    xml_path = output_dir / "decoder_init.xml"
    ov.save_model(ov_model, str(xml_path))
    return xml_path


def export_decoder_step_openvino(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    config_path: Path | None = None,
    trace_num_patches: int = 128,
) -> Path:
    """Export the cached decoder step to OpenVINO IR.

    Produces ``decoder_step.xml`` and ``decoder_step.bin`` inside *output_dir*.
    This graph processes one new token using KV cache from previous steps:

        input_ids (1,1) + past_self_k/v (L,1,H,S,hd) + past_cross_k/v
        + encoder_mask (1,N)
        → logits (1,1,V) + updated self_k/v (L,1,H,S+1,hd)
        + pass-through cross_k/v

    Args:
        checkpoint_path: Lightning ``.ckpt`` checkpoint.
        output_dir: Destination for ``.xml`` / ``.bin``.
        config_path: YAML config (defaults to ``configs/default.yaml``).
        trace_num_patches: Encoder patch count for tracing dummy inputs.

    Returns:
        Path to the exported ``.xml`` model file.
    """
    import openvino as ov

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    cfg = load_config(Path(config_path or "configs/default.yaml"))
    model_cfg = cfg.get("model", {})
    chant_config = ChantOMRConfig.from_mapping(model_cfg)

    n_layers = chant_config.n_layers
    n_heads = chant_config.n_heads
    head_dim = chant_config.d_model // chant_config.n_heads

    step_module = CachedDecoderStepForExport(model)
    step_module.eval()

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_self_k = torch.randn(n_layers, 1, n_heads, 1, head_dim)
    dummy_self_v = torch.randn(n_layers, 1, n_heads, 1, head_dim)
    dummy_cross_k = torch.randn(n_layers, 1, n_heads, trace_num_patches, head_dim)
    dummy_cross_v = torch.randn(n_layers, 1, n_heads, trace_num_patches, head_dim)
    dummy_mask = torch.ones(1, trace_num_patches)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "decoder_step.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                step_module,
                (dummy_ids, dummy_self_k, dummy_self_v,
                 dummy_cross_k, dummy_cross_v, dummy_mask),
                str(onnx_path),
                input_names=[
                    "input_ids",
                    "past_self_k", "past_self_v",
                    "past_cross_k", "past_cross_v",
                    "encoder_mask",
                ],
                output_names=[
                    "logits", "self_k", "self_v", "cross_k", "cross_v",
                ],
                opset_version=18,
                dynamo=False,
                dynamic_axes={
                    "past_self_k": {3: "past_seq_len"},
                    "past_self_v": {3: "past_seq_len"},
                    "past_cross_k": {3: "num_patches"},
                    "past_cross_v": {3: "num_patches"},
                    "encoder_mask": {1: "num_patches"},
                    "self_k": {3: "seq_len"},
                    "self_v": {3: "seq_len"},
                    "cross_k": {3: "num_patches"},
                    "cross_v": {3: "num_patches"},
                },
            )

        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))

    xml_path = output_dir / "decoder_step.xml"
    ov.save_model(ov_model, str(xml_path))
    return xml_path


def verify_decoder_init_openvino_parity(
    checkpoint_path: Path,
    xml_path: Path,
    *,
    config_path: Path | None = None,
    num_patches: int = 128,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch cached decoder init with OpenVINO IR."""
    import numpy as np
    import openvino as ov

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    init_module = CachedDecoderInitForExport(model)
    init_module.eval()

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_memory = torch.randn(1, num_patches, model.config.d_model)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_outputs = init_module(dummy_ids, dummy_memory, dummy_mask)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    ov_results = compiled({
        "input_ids": dummy_ids.numpy(),
        "encoder_memory": dummy_memory.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })

    max_diff = 0.0
    names = ["logits", "self_k", "self_v", "cross_k", "cross_v"]
    for i, name in enumerate(names):
        pt_t = pt_outputs[i]
        ov_t = torch.from_numpy(np.array(ov_results[i]))
        diff = float((pt_t - ov_t).abs().max())
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        raise AssertionError(
            f"Decoder init OV parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff


def verify_decoder_step_openvino_parity(
    checkpoint_path: Path,
    xml_path: Path,
    *,
    config_path: Path | None = None,
    num_patches: int = 128,
    past_seq_len: int = 3,
    atol: float = 5e-3,
) -> float:
    """Compare PyTorch cached decoder step with OpenVINO IR."""
    import numpy as np
    import openvino as ov

    model, _tok, _meta = load_model_from_checkpoint(
        checkpoint_path, config_path=config_path, device="cpu",
    )

    step_module = CachedDecoderStepForExport(model)
    step_module.eval()

    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    head_dim = model.config.d_model // model.config.n_heads

    dummy_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_self_k = torch.randn(n_layers, 1, n_heads, past_seq_len, head_dim)
    dummy_self_v = torch.randn(n_layers, 1, n_heads, past_seq_len, head_dim)
    dummy_cross_k = torch.randn(n_layers, 1, n_heads, num_patches, head_dim)
    dummy_cross_v = torch.randn(n_layers, 1, n_heads, num_patches, head_dim)
    dummy_mask = torch.ones(1, num_patches)

    with torch.inference_mode():
        pt_outputs = step_module(
            dummy_ids, dummy_self_k, dummy_self_v,
            dummy_cross_k, dummy_cross_v, dummy_mask,
        )

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    ov_results = compiled({
        "input_ids": dummy_ids.numpy(),
        "past_self_k": dummy_self_k.numpy(),
        "past_self_v": dummy_self_v.numpy(),
        "past_cross_k": dummy_cross_k.numpy(),
        "past_cross_v": dummy_cross_v.numpy(),
        "encoder_mask": dummy_mask.numpy(),
    })

    max_diff = 0.0
    names = ["logits", "self_k", "self_v", "cross_k", "cross_v"]
    for i, name in enumerate(names):
        pt_t = pt_outputs[i]
        ov_t = torch.from_numpy(np.array(ov_results[i]))
        diff = float((pt_t - ov_t).abs().max())
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        raise AssertionError(
            f"Decoder step OV parity failed: max abs diff {max_diff:.6f} > atol {atol}"
        )
    return max_diff
