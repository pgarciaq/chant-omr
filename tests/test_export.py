"""Tests for ONNX/OpenVINO export, safetensors, and OV decode (#13b, #36, #41, #50)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from chant_omr.inference.beam_search import (
    greedy_decode_generic,
    pytorch_logits_func,
)
from chant_omr.inference.export import (
    ENCODER_OUTPUT_STRIDE,
    CachedDecoderInitForExport,
    CachedDecoderStepForExport,
    DecoderStepForExport,
    EncoderForExport,
    export_decoder_init_openvino,
    export_decoder_openvino,
    export_decoder_step_openvino,
    export_onnx,
    export_openvino,
    export_safetensors,
    verify_decoder_init_openvino_parity,
    verify_decoder_openvino_parity,
    verify_decoder_step_openvino_parity,
    verify_onnx_decoder_init_parity,
    verify_onnx_decoder_step_parity,
    verify_onnx_encoder_parity,
    verify_openvino_parity,
)
from chant_omr.model.chant_omr_model import ChantOMRConfig, build_model
from chant_omr.model.tokenizer import train_tokenizer
from chant_omr.training.lightning_module import ChantOMRLightningModule

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"


@pytest.fixture
def tokenizer(tmp_path: Path):
    return train_tokenizer(
        GABC_FIXTURES,
        vocab_size=256,
        output_dir=tmp_path / "tokenizer",
        min_body_len=10,
        use_manifest=False,
    )


@pytest.fixture
def config_and_ckpt(tmp_path: Path, tokenizer) -> tuple[Path, Path]:
    """Create a minimal config and checkpoint for export tests."""
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    cfg = {
        "data": {"tokenizer_dir": str(tok_dir)},
        "model": {
            "encoder_pretrained": False,
            "vocab_size": 256,
            "max_seq_len": 128,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    model = build_model(ChantOMRConfig.from_mapping(cfg["model"]), encoder_pretrained=False)
    module = ChantOMRLightningModule(model, pad_token_id=tokenizer.pad_id)
    ckpt = {
        "state_dict": {f"model.{k}": v for k, v in module.model.state_dict().items()},
        "hyper_parameters": module.hparams,
    }
    ckpt_path = tmp_path / "test.ckpt"
    torch.save(ckpt, ckpt_path)
    return cfg_path, ckpt_path


# ---------------------------------------------------------------------------
# Encoder export (#13b)
# ---------------------------------------------------------------------------


class TestEncoderForExport:
    def test_forward_shape(self):
        model = build_model(encoder_pretrained=False)
        model.eval()
        enc = EncoderForExport(model)
        enc.eval()
        dummy = torch.randn(1, 3, 128, 1050)
        with torch.inference_mode():
            out = enc(dummy)
        expected_patches = (128 // ENCODER_OUTPUT_STRIDE) * (1050 // ENCODER_OUTPUT_STRIDE)
        assert out.shape == (1, expected_patches, model.config.d_model)

    def test_matches_model_encode(self):
        model = build_model(encoder_pretrained=False)
        model.eval()
        enc = EncoderForExport(model)
        enc.eval()
        dummy = torch.randn(1, 3, 128, 1050)
        with torch.inference_mode():
            from_wrapper = enc(dummy)
            from_model = model.encode(dummy)
        assert torch.allclose(from_wrapper, from_model, atol=1e-6)


# ---------------------------------------------------------------------------
# ONNX wrapper modules (#50)
# ---------------------------------------------------------------------------


class TestCachedDecoderInitForExport:
    def test_output_shapes(self):
        cfg = ChantOMRConfig(encoder_pretrained=False, vocab_size=256, max_seq_len=128)
        model = build_model(cfg, encoder_pretrained=False)
        model.eval()
        init = CachedDecoderInitForExport(model)
        init.eval()
        dummy_ids = torch.ones(1, 1, dtype=torch.long)
        dummy_memory = torch.randn(1, 64, cfg.d_model)
        dummy_mask = torch.ones(1, 64)
        with torch.inference_mode():
            logits, sk, sv, ck, cv = init(dummy_ids, dummy_memory, dummy_mask)
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert sk.shape == (cfg.n_layers, 1, cfg.n_heads, 1, cfg.d_model // cfg.n_heads)
        assert sv.shape == sk.shape
        assert ck.shape == (cfg.n_layers, 1, cfg.n_heads, 64, cfg.d_model // cfg.n_heads)
        assert cv.shape == ck.shape

    def test_logits_match_decoder(self):
        cfg = ChantOMRConfig(encoder_pretrained=False, vocab_size=256, max_seq_len=128)
        model = build_model(cfg, encoder_pretrained=False)
        model.eval()
        init = CachedDecoderInitForExport(model)
        init.eval()
        dummy_ids = torch.ones(1, 1, dtype=torch.long)
        dummy_memory = torch.randn(1, 64, cfg.d_model)
        dummy_mask = torch.ones(1, 64)
        with torch.inference_mode():
            init_logits, *_ = init(dummy_ids, dummy_memory, dummy_mask)
            full_logits, _ = model.decoder(
                dummy_ids, dummy_memory, encoder_attention_mask=dummy_mask,
            )
        assert torch.allclose(init_logits[0, 0], full_logits[0, -1], atol=1e-6)


class TestCachedDecoderStepForExport:
    def test_output_shapes(self):
        cfg = ChantOMRConfig(encoder_pretrained=False, vocab_size=256, max_seq_len=128)
        model = build_model(cfg, encoder_pretrained=False)
        model.eval()
        step = CachedDecoderStepForExport(model)
        step.eval()
        head_dim = cfg.d_model // cfg.n_heads
        dummy_ids = torch.ones(1, 1, dtype=torch.long)
        past_self_k = torch.randn(cfg.n_layers, 1, cfg.n_heads, 3, head_dim)
        past_self_v = torch.randn(cfg.n_layers, 1, cfg.n_heads, 3, head_dim)
        past_cross_k = torch.randn(cfg.n_layers, 1, cfg.n_heads, 64, head_dim)
        past_cross_v = torch.randn(cfg.n_layers, 1, cfg.n_heads, 64, head_dim)
        dummy_mask = torch.ones(1, 64)
        with torch.inference_mode():
            logits, sk, sv, ck, cv = step(
                dummy_ids, past_self_k, past_self_v, past_cross_k, past_cross_v, dummy_mask,
            )
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert sk.shape == (cfg.n_layers, 1, cfg.n_heads, 4, head_dim)  # past + 1
        assert sv.shape == sk.shape
        assert ck.shape == past_cross_k.shape  # unchanged
        assert cv.shape == past_cross_v.shape

    def test_init_then_step_consistency(self):
        """Run init → step and verify the step produces valid logits."""
        cfg = ChantOMRConfig(encoder_pretrained=False, vocab_size=256, max_seq_len=128)
        model = build_model(cfg, encoder_pretrained=False)
        model.eval()
        init = CachedDecoderInitForExport(model)
        step = CachedDecoderStepForExport(model)
        init.eval()
        step.eval()

        dummy_memory = torch.randn(1, 64, cfg.d_model)
        dummy_mask = torch.ones(1, 64)
        bos = torch.ones(1, 1, dtype=torch.long)

        with torch.inference_mode():
            _logits0, sk, sv, ck, cv = init(bos, dummy_memory, dummy_mask)
            next_token = torch.tensor([[10]], dtype=torch.long)
            logits1, sk2, sv2, ck2, cv2 = step(
                next_token, sk, sv, ck, cv, dummy_mask,
            )

        assert logits1.shape == (1, 1, cfg.vocab_size)
        head_dim = cfg.d_model // cfg.n_heads
        assert sk2.shape == (cfg.n_layers, 1, cfg.n_heads, 2, head_dim)


class TestExportOnnx:
    def test_produces_onnx_files(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "onnx_export"
        export_onnx(
            ckpt_path, out_dir, config_path=cfg_path,
            trace_height=128,
        )
        assert (out_dir / "encoder.onnx").exists()
        assert (out_dir / "decoder_init.onnx").exists()
        assert (out_dir / "decoder_step.onnx").exists()
        assert (out_dir / "manifest.json").exists()

    def test_manifest_format(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "onnx_manifest"
        export_onnx(
            ckpt_path, out_dir, config_path=cfg_path,
            trace_height=128,
        )
        data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert data["format"] == "onnx"
        assert data["d_model"] == 512
        assert data["canvas_height"] == 128

    def test_encoder_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "onnx_enc_parity"
        export_onnx(
            ckpt_path, out_dir, config_path=cfg_path,
            trace_height=128,
        )
        diff = verify_onnx_encoder_parity(
            ckpt_path, out_dir / "encoder.onnx",
            config_path=cfg_path, input_height=128,
        )
        assert diff < 2e-3

    def test_decoder_init_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "onnx_init_parity"
        export_onnx(
            ckpt_path, out_dir, config_path=cfg_path,
            trace_height=128,
        )
        diff = verify_onnx_decoder_init_parity(
            ckpt_path, out_dir / "decoder_init.onnx",
            config_path=cfg_path,
        )
        assert diff < 5e-3

    def test_decoder_step_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "onnx_step_parity"
        export_onnx(
            ckpt_path, out_dir, config_path=cfg_path,
            trace_height=128,
        )
        diff = verify_onnx_decoder_step_parity(
            ckpt_path, out_dir / "decoder_step.onnx",
            config_path=cfg_path,
        )
        assert diff < 5e-3


# ---------------------------------------------------------------------------
# OpenVINO export (#13b)
# ---------------------------------------------------------------------------


class TestExportOpenVINO:
    def test_produces_ir_files(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_export"
        xml_path = export_openvino(
            ckpt_path,
            out_dir,
            config_path=cfg_path,
            input_height=128,
            input_width=1050,
        )
        assert xml_path.exists()
        assert xml_path.suffix == ".xml"
        bin_path = xml_path.with_suffix(".bin")
        assert bin_path.exists()

    def test_manifest_written(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_manifest"
        export_openvino(
            ckpt_path,
            out_dir,
            config_path=cfg_path,
            input_height=128,
            input_width=1050,
        )
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["format"] == "openvino"
        assert data["canvas_height"] == 128
        assert data["canvas_width"] == 1050
        expected_patches = (128 // ENCODER_OUTPUT_STRIDE) * (1050 // ENCODER_OUTPUT_STRIDE)
        assert data["encoder_patches"] == expected_patches

    def test_parity_check_passes(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_parity"
        xml_path = export_openvino(
            ckpt_path,
            out_dir,
            config_path=cfg_path,
            input_height=128,
            input_width=1050,
        )
        diff = verify_openvino_parity(
            ckpt_path,
            xml_path,
            config_path=cfg_path,
            input_height=128,
            input_width=1050,
            atol=1e-3,
        )
        assert diff < 1e-3


# ---------------------------------------------------------------------------
# Decoder export (#41)
# ---------------------------------------------------------------------------


class TestDecoderStepForExport:
    def test_output_shape(self):
        model = build_model(encoder_pretrained=False)
        model.eval()
        dec = DecoderStepForExport(model)
        dec.eval()
        dummy_ids = torch.tensor([[1, 10, 11]], dtype=torch.long)
        dummy_memory = torch.randn(1, 64, model.config.d_model)
        dummy_mask = torch.ones(1, 64)
        with torch.inference_mode():
            out = dec(dummy_ids, dummy_memory, dummy_mask)
        assert out.shape == (1, 1, model.config.vocab_size)

    def test_last_position_matches_full_forward(self):
        model = build_model(encoder_pretrained=False)
        model.eval()
        dec = DecoderStepForExport(model)
        dec.eval()
        dummy_ids = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
        dummy_memory = torch.randn(1, 64, model.config.d_model)
        dummy_mask = torch.ones(1, 64)
        with torch.inference_mode():
            step_out = dec(dummy_ids, dummy_memory, dummy_mask)
            full_out, _ = model.decoder(
                dummy_ids, dummy_memory,
                encoder_attention_mask=dummy_mask,
            )
        assert torch.allclose(step_out[0, 0], full_out[0, -1], atol=1e-6)


class TestExportDecoderOpenVINO:
    def test_produces_ir_files(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_export"
        xml_path = export_decoder_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        assert xml_path.exists()
        assert xml_path.name == "decoder.xml"
        assert xml_path.with_suffix(".bin").exists()

    def test_decoder_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_parity"
        xml_path = export_decoder_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        diff = verify_decoder_openvino_parity(
            ckpt_path, xml_path, config_path=cfg_path,
        )
        assert diff < 5e-3


class TestExportDecoderInitOpenVINO:
    """Tests for KV-cached decoder_init OpenVINO export (#36)."""

    def test_produces_ir_files(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_init_export"
        xml_path = export_decoder_init_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        assert xml_path.exists()
        assert xml_path.name == "decoder_init.xml"
        assert xml_path.with_suffix(".bin").exists()

    def test_decoder_init_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_init_parity"
        xml_path = export_decoder_init_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        diff = verify_decoder_init_openvino_parity(
            ckpt_path, xml_path, config_path=cfg_path,
        )
        assert diff < 5e-3


class TestExportDecoderStepOpenVINO:
    """Tests for KV-cached decoder_step OpenVINO export (#36)."""

    def test_produces_ir_files(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_step_export"
        xml_path = export_decoder_step_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        assert xml_path.exists()
        assert xml_path.name == "decoder_step.xml"
        assert xml_path.with_suffix(".bin").exists()

    def test_decoder_step_parity(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "dec_step_parity"
        xml_path = export_decoder_step_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )
        diff = verify_decoder_step_openvino_parity(
            ckpt_path, xml_path, config_path=cfg_path,
        )
        assert diff < 5e-3


def _export_all_ov(ckpt_path, cfg_path, out_dir):
    """Helper: export all 4 OpenVINO IRs + tokenizer into out_dir."""
    export_openvino(
        ckpt_path, out_dir, config_path=cfg_path,
        input_height=128, input_width=1050,
    )
    export_decoder_openvino(ckpt_path, out_dir, config_path=cfg_path)
    export_decoder_init_openvino(ckpt_path, out_dir, config_path=cfg_path)
    export_decoder_step_openvino(ckpt_path, out_dir, config_path=cfg_path)


class TestOVDecodeIntegration:
    """End-to-end: all 4 OpenVINO IRs → token IDs (no PyTorch model)."""

    def test_ov_cached_greedy_produces_tokens(self, config_and_ckpt, tokenizer, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_greedy"
        _export_all_ov(ckpt_path, cfg_path, out_dir)

        import numpy as np

        from chant_omr.inference.ov_decode import (
            load_openvino_models,
            ov_encoder_infer,
            ov_logits_func_cached,
        )

        enc, _dec, init, step, manifest, tok = load_openvino_models(out_dir)
        dummy_pixels = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory = ov_encoder_infer(enc, dummy_pixels)
        memory_t = torch.from_numpy(memory)
        mask_np = np.ones((1, memory.shape[1]), dtype=np.float32)

        logits_fn = ov_logits_func_cached(init, step, mask_np)
        token_ids = greedy_decode_generic(
            logits_fn, memory_t,
            bos_token_id=tok.bos_id, eos_token_id=tok.eos_id, max_length=8,
        )
        assert token_ids[0] == tok.bos_id
        assert len(token_ids) >= 2

    def test_ov_noncached_beam_produces_tokens(self, config_and_ckpt, tokenizer, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_beam"
        _export_all_ov(ckpt_path, cfg_path, out_dir)

        import numpy as np

        from chant_omr.inference.beam_search import beam_search_decode_generic
        from chant_omr.inference.ov_decode import (
            load_openvino_models,
            ov_decoder_logits_func,
            ov_encoder_infer,
        )

        enc, dec, _init, _step, manifest, tok = load_openvino_models(out_dir)
        dummy_pixels = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory = ov_encoder_infer(enc, dummy_pixels)
        memory_t = torch.from_numpy(memory)
        mask_np = np.ones((1, memory.shape[1]), dtype=np.float32)

        logits_fn = ov_decoder_logits_func(dec, mask_np)
        token_ids = beam_search_decode_generic(
            logits_fn, memory_t,
            bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
            max_length=8, beam_width=3,
        )
        assert token_ids[0] == tok.bos_id
        assert len(token_ids) >= 2

    def test_ov_cached_greedy_decoded_is_string(self, config_and_ckpt, tokenizer, tmp_path):
        """Greedy cached path: token IDs decode to a string."""
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_decode_str"
        _export_all_ov(ckpt_path, cfg_path, out_dir)

        import numpy as np

        from chant_omr.inference.ov_decode import (
            load_openvino_models,
            ov_encoder_infer,
            ov_logits_func_cached,
        )

        enc, _dec, init, step, _manifest, tok = load_openvino_models(out_dir)
        dummy_pixels = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory = ov_encoder_infer(enc, dummy_pixels)
        memory_t = torch.from_numpy(memory)
        mask_np = np.ones((1, memory.shape[1]), dtype=np.float32)

        logits_fn = ov_logits_func_cached(init, step, mask_np)
        token_ids = greedy_decode_generic(
            logits_fn, memory_t,
            bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
            max_length=32,
        )
        decoded = tok.decode(token_ids, skip_special_tokens=True)
        assert isinstance(decoded, str)


# ---------------------------------------------------------------------------
# Beam search refactor — verify generic functions match originals
# ---------------------------------------------------------------------------


class TestGenericDecodeBackwardCompat:
    def test_generic_greedy_matches_pytorch(self):
        model = build_model(
            ChantOMRConfig(encoder_pretrained=False, vocab_size=256),
            encoder_pretrained=False,
        )
        model.eval()
        memory = torch.randn(1, 64, model.config.d_model)
        logits_fn = pytorch_logits_func(model)
        with torch.inference_mode():
            from chant_omr.inference.beam_search import greedy_decode

            original = greedy_decode(
                model, memory,
                bos_token_id=0, eos_token_id=2, max_length=12,
            )
            generic = greedy_decode_generic(
                logits_fn, memory,
                bos_token_id=0, eos_token_id=2, max_length=12,
            )
        assert original == generic


# ---------------------------------------------------------------------------
# Safetensors export (#13b)
# ---------------------------------------------------------------------------


class TestExportSafetensors:
    def test_produces_safetensors(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "st_export"
        st_path = export_safetensors(ckpt_path, out_dir, config_path=cfg_path)
        assert st_path.exists()
        assert st_path.suffix == ".safetensors"

    def test_safetensors_manifest(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "st_manifest"
        export_safetensors(ckpt_path, out_dir, config_path=cfg_path)
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format"] == "safetensors"
        assert "d_model" in manifest

    def test_roundtrip_weights(self, config_and_ckpt, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "st_roundtrip"
        st_path = export_safetensors(ckpt_path, out_dir, config_path=cfg_path)

        from safetensors.torch import load_file

        loaded = load_file(str(st_path))
        from chant_omr.inference.checkpoint import load_model_from_checkpoint

        model, _, _ = load_model_from_checkpoint(ckpt_path, config_path=cfg_path, device="cpu")
        for key in model.state_dict():
            assert key in loaded, f"missing key: {key}"
            assert torch.equal(model.state_dict()[key], loaded[key]), f"mismatch: {key}"
