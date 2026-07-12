"""Tests for inference / predict (#13a)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from chant_omr.data.gabc_parser import parse_gabc, plain_gabc_reject_reason
from chant_omr.inference.beam_search import DecodeConfig, decode_token_ids, greedy_decode
from chant_omr.inference.checkpoint import load_model_from_checkpoint
from chant_omr.inference.gabc_output import assemble_gabc
from chant_omr.inference.predict import predict_gabc, resolve_inference_device
from chant_omr.model.chant_omr_model import build_model
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
def config_paths(tmp_path: Path, tokenizer) -> tuple[Path, Path]:
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    cfg = {
        "data": {"tokenizer_dir": str(tok_dir)},
        "model": {
            "encoder_pretrained": False,
            "vocab_size": 256,
            "max_seq_len": 128,
        },
        "inference": {"beam_width": 1, "repetition_penalty": 1.0, "max_length": 32},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_path, tok_dir


@pytest.fixture
def checkpoint_path(tmp_path: Path, tokenizer, config_paths) -> Path:
    cfg_path, _ = config_paths
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    from chant_omr.model.chant_omr_model import ChantOMRConfig

    model = build_model(ChantOMRConfig.from_mapping(cfg["model"]), encoder_pretrained=False)
    module = ChantOMRLightningModule(model, pad_token_id=tokenizer.pad_id)
    ckpt = {
        "state_dict": {f"model.{k}": v for k, v in module.model.state_dict().items()},
        "hyper_parameters": module.hparams,
    }
    path = tmp_path / "test.ckpt"
    torch.save(ckpt, path)
    return path


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    path = tmp_path / "score.png"
    Image.new("RGB", (420, 120), color=(255, 255, 255)).save(path)
    return path


class TestGabcOutput:
    def test_assemble_gabc_parses(self):
        body = "(c4) Ky(f)ri(gf)e(h) *() e(ixhi)lé(h)i(g)son.(f)"
        text = assemble_gabc(body, name="Test chant")
        score = parse_gabc(text)
        assert score.headers["name"] == "Test chant"
        assert score.body == body
        assert plain_gabc_reject_reason(text.encode(), min_body_len=10) is None


class TestCheckpoint:
    def test_load_model_from_checkpoint(self, checkpoint_path, config_paths, tokenizer):
        cfg_path, tok_dir = config_paths
        model, loaded_tok, meta = load_model_from_checkpoint(
            checkpoint_path,
            config_path=cfg_path,
            tokenizer_dir=tok_dir,
        )
        assert loaded_tok.vocab_size == tokenizer.vocab_size
        assert meta["checkpoint_path"].endswith("test.ckpt")
        assert sum(p.numel() for p in model.parameters()) > 0


class TestBeamSearch:
    def test_greedy_decode_smoke(self, tokenizer):
        model = build_model(encoder_pretrained=False)
        model.eval()
        memory = torch.randn(1, 64, model.config.d_model)
        tokens = greedy_decode(
            model,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=16,
        )
        assert tokens[0] == tokenizer.bos_id
        assert len(tokens) >= 2

    def test_decode_token_ids_smoke(self, tokenizer):
        model = build_model(encoder_pretrained=False)
        model.eval()
        pixels = torch.randn(1, 3, 128, 1050)
        ids = decode_token_ids(
            model,
            pixels,
            tokenizer,
            DecodeConfig(beam_width=1, max_length=12, repetition_penalty=1.0),
        )
        assert ids[0] == tokenizer.bos_id


class TestPredictGabc:
    def test_predict_output_valid_gabc(
        self,
        checkpoint_path,
        config_paths,
        tiny_png,
        monkeypatch: pytest.MonkeyPatch,
    ):
        cfg_path, _ = config_paths
        monkeypatch.setattr(
            "chant_omr.inference.predict.format_training_device_message",
            lambda **kwargs: "",
        )
        text = predict_gabc(
            tiny_png,
            checkpoint_path,
            config_path=cfg_path,
            device="cpu",
            beam_width=1,
            max_length=16,
            repetition_penalty=1.0,
        )
        assert "%%" in text
        parse_gabc(text)

    def test_resolve_inference_device_cpu(self):
        assert resolve_inference_device("cpu").type == "cpu"

    def test_resolve_inference_device_auto_cpu(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("chant_omr.inference.predict.torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("chant_omr.inference.predict.xpu_is_available", lambda: False)
        assert resolve_inference_device("auto").type == "cpu"
