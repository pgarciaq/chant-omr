"""Tests for OpenVINO encoder/decoder export, safetensors, and OV decode (#13b, #41)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from chant_omr.inference.beam_search import (
    DecodeConfig,
    greedy_decode_generic,
    pytorch_logits_func,
)
from chant_omr.inference.export import (
    ENCODER_OUTPUT_STRIDE,
    DecoderStepForExport,
    EncoderForExport,
    export_decoder_openvino,
    export_openvino,
    export_safetensors,
    verify_decoder_openvino_parity,
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
        with torch.inference_mode():
            out = dec(dummy_ids, dummy_memory)
        assert out.shape == (1, 1, model.config.vocab_size)

    def test_last_position_matches_full_forward(self):
        model = build_model(encoder_pretrained=False)
        model.eval()
        dec = DecoderStepForExport(model)
        dec.eval()
        dummy_ids = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
        dummy_memory = torch.randn(1, 64, model.config.d_model)
        with torch.inference_mode():
            step_out = dec(dummy_ids, dummy_memory)
            full_out = model.decoder(dummy_ids, dummy_memory)
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


class TestOVDecodeIntegration:
    """End-to-end: encoder IR + decoder IR → token IDs (no PyTorch model)."""

    def test_ov_greedy_decode_produces_tokens(self, config_and_ckpt, tokenizer, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_e2e"
        export_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
            input_height=128, input_width=1050,
        )
        export_decoder_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )

        import numpy as np

        from chant_omr.inference.ov_decode import load_openvino_models, ov_decode_token_ids

        enc_compiled, dec_compiled, manifest = load_openvino_models(out_dir)
        dummy_pixels = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        token_ids = ov_decode_token_ids(
            enc_compiled,
            dec_compiled,
            dummy_pixels,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            config=DecodeConfig(beam_width=1, max_length=8, repetition_penalty=1.0),
        )
        assert token_ids[0] == tokenizer.bos_id
        assert len(token_ids) >= 2

    def test_ov_beam_decode_produces_tokens(self, config_and_ckpt, tokenizer, tmp_path):
        cfg_path, ckpt_path = config_and_ckpt
        out_dir = tmp_path / "ov_beam"
        export_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
            input_height=128, input_width=1050,
        )
        export_decoder_openvino(
            ckpt_path, out_dir, config_path=cfg_path,
        )

        import numpy as np

        from chant_omr.inference.ov_decode import load_openvino_models, ov_decode_token_ids

        enc_compiled, dec_compiled, _ = load_openvino_models(out_dir)
        dummy_pixels = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        token_ids = ov_decode_token_ids(
            enc_compiled,
            dec_compiled,
            dummy_pixels,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            config=DecodeConfig(beam_width=3, max_length=8, repetition_penalty=1.0),
        )
        assert token_ids[0] == tokenizer.bos_id
        assert len(token_ids) >= 2


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
