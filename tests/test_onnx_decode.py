"""Tests for ONNX Runtime predict backend (#51, #60)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from chant_omr.inference.beam_search import (
    greedy_decode_generic,
    pytorch_logits_func_cached,
)
from chant_omr.inference.export import export_onnx
from chant_omr.inference.preprocess import prepare_inference_numpy, prepare_inference_tensor
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


@pytest.fixture
def onnx_model_dir(config_and_ckpt, tmp_path) -> Path:
    """Export ONNX models to a temp directory."""
    cfg_path, ckpt_path = config_and_ckpt
    out_dir = tmp_path / "onnx_models"
    export_onnx(ckpt_path, out_dir, config_path=cfg_path, trace_height=128)
    return out_dir


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    path = tmp_path / "score.png"
    Image.new("RGB", (420, 120), color=(255, 255, 255)).save(path)
    return path


class TestPrepareInferenceNumpy:
    def test_output_shape_and_dtype(self, tiny_png):
        arr = prepare_inference_numpy(tiny_png)
        assert arr.dtype == np.float32
        assert arr.ndim == 4
        assert arr.shape[0] == 1
        assert arr.shape[1] == 3

    def test_matches_tensor_version(self, tiny_png):
        np_arr = prepare_inference_numpy(tiny_png)
        t_arr = prepare_inference_tensor(tiny_png).numpy()
        np.testing.assert_allclose(np_arr, t_arr, atol=1e-6)


class TestLoadOnnxModels:
    def test_loads_sessions(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import load_onnx_models

        enc, dec, init, step, manifest, tok = load_onnx_models(onnx_model_dir)
        assert enc is not None
        assert dec is not None
        assert init is not None
        assert step is not None
        assert manifest["format"] == "onnx"
        assert tok.bos_id >= 0

    def test_missing_manifest_raises(self, tmp_path):
        from chant_omr.inference.onnx_decode import load_onnx_models

        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_onnx_models(tmp_path)

    def test_missing_onnx_file_raises(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import load_onnx_models

        (onnx_model_dir / "encoder.onnx").unlink()
        with pytest.raises(FileNotFoundError, match="encoder.onnx"):
            load_onnx_models(onnx_model_dir)


class TestOnnxEncoderInfer:
    def test_output_shape(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import load_onnx_models, onnx_encoder_infer

        enc, _dec, _init, _step, _manifest, _tok = load_onnx_models(onnx_model_dir)
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory = onnx_encoder_infer(enc, dummy)
        assert memory.ndim == 3
        assert memory.shape[0] == 1
        assert memory.shape[1] == (128 // 32) * (1050 // 32)


class TestOnnxLogitsFuncCached:
    def test_first_call_uses_init(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_encoder_infer,
            onnx_logits_func_cached,
        )

        enc, _dec, init, step, manifest, tok = load_onnx_models(onnx_model_dir)
        model_vocab = manifest["config"]["vocab_size"]
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory_np = onnx_encoder_infer(enc, dummy)
        memory_t = torch.from_numpy(memory_np)
        mask = np.ones((1, memory_np.shape[1]), dtype=np.float32)
        logits_fn = onnx_logits_func_cached(init, step, mask)

        bos_ids = torch.tensor([[tok.bos_id]], dtype=torch.long)
        log_probs = logits_fn(bos_ids, memory_t)
        assert log_probs.ndim == 1
        assert log_probs.shape[0] == model_vocab

    def test_subsequent_calls_grow_cache(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_encoder_infer,
            onnx_logits_func_cached,
        )

        enc, _dec, init, step, manifest, tok = load_onnx_models(onnx_model_dir)
        model_vocab = manifest["config"]["vocab_size"]
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory_np = onnx_encoder_infer(enc, dummy)
        memory_t = torch.from_numpy(memory_np)
        mask = np.ones((1, memory_np.shape[1]), dtype=np.float32)
        logits_fn = onnx_logits_func_cached(init, step, mask)

        ids = torch.tensor([[tok.bos_id]], dtype=torch.long)
        lp1 = logits_fn(ids, memory_t)
        assert lp1.shape[0] == model_vocab

        next_id = int(torch.argmax(lp1).item())
        ids = torch.tensor([[tok.bos_id, next_id]], dtype=torch.long)
        lp2 = logits_fn(ids, memory_t)
        assert lp2.shape[0] == model_vocab

    def test_greedy_decode_produces_tokens(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_encoder_infer,
            onnx_logits_func_cached,
        )

        enc, _dec, init, step, _manifest, tok = load_onnx_models(onnx_model_dir)
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory_np = onnx_encoder_infer(enc, dummy)
        memory_t = torch.from_numpy(memory_np)
        mask = np.ones((1, memory_np.shape[1]), dtype=np.float32)
        logits_fn = onnx_logits_func_cached(init, step, mask)

        token_ids = greedy_decode_generic(
            logits_fn,
            memory_t,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
            max_length=16,
        )
        assert token_ids[0] == tok.bos_id
        assert len(token_ids) >= 2


class TestOnnxDecoderLogitsFunc:
    """Tests for the non-cached decoder logits func (beam search path)."""

    def test_noncached_produces_log_probs(self, onnx_model_dir):
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_decoder_logits_func,
            onnx_encoder_infer,
        )

        enc, dec, _init, _step, manifest, tok = load_onnx_models(onnx_model_dir)
        model_vocab = manifest["config"]["vocab_size"]
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory_np = onnx_encoder_infer(enc, dummy)
        memory_t = torch.from_numpy(memory_np)
        mask_np = np.ones((1, memory_np.shape[1]), dtype=np.float32)

        logits_fn = onnx_decoder_logits_func(dec, mask_np)
        ids = torch.tensor([[tok.bos_id]], dtype=torch.long)
        log_probs = logits_fn(ids, memory_t)
        assert log_probs.ndim == 1
        assert log_probs.shape[0] == model_vocab

    def test_beam_search_produces_tokens(self, onnx_model_dir):
        from chant_omr.inference.beam_search import beam_search_decode_generic
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_decoder_logits_func,
            onnx_encoder_infer,
        )

        enc, dec, _init, _step, _manifest, tok = load_onnx_models(onnx_model_dir)
        dummy = np.random.randn(1, 3, 128, 1050).astype(np.float32)
        memory_np = onnx_encoder_infer(enc, dummy)
        memory_t = torch.from_numpy(memory_np)
        mask_np = np.ones((1, memory_np.shape[1]), dtype=np.float32)

        logits_fn = onnx_decoder_logits_func(dec, mask_np)
        token_ids = beam_search_decode_generic(
            logits_fn, memory_t,
            bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
            max_length=8, beam_width=3,
        )
        assert token_ids[0] == tok.bos_id
        assert len(token_ids) >= 2


class TestOnnxPredictGabc:
    def test_end_to_end(self, onnx_model_dir, tiny_png):
        """Full pipeline: load ONNX models → preprocess → decode → tokens.

        An untrained model may produce only special tokens on a blank image,
        so we test the pipeline up to token generation rather than requiring
        valid GABC text.
        """
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_encoder_infer,
            onnx_logits_func_cached,
        )

        enc, _dec, init, step, _manifest, tok = load_onnx_models(onnx_model_dir)
        pixel_np = prepare_inference_numpy(tiny_png)
        memory_np = onnx_encoder_infer(enc, pixel_np)
        memory_t = torch.from_numpy(memory_np)
        mask_np = np.ones((1, memory_np.shape[1]), dtype=np.float32)
        logits_fn = onnx_logits_func_cached(init, step, mask_np)

        token_ids = greedy_decode_generic(
            logits_fn,
            memory_t,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
            max_length=32,
            repetition_penalty=1.0,
        )
        assert token_ids[0] == tok.bos_id
        assert len(token_ids) >= 2
        decoded = tok.decode(token_ids, skip_special_tokens=True)
        assert isinstance(decoded, str)


class TestOnnxPredictMatchesPyTorch:
    """Same checkpoint, same image: ONNX greedy == PyTorch greedy."""

    def test_identical_token_sequence(self, config_and_ckpt, onnx_model_dir, tiny_png):
        from chant_omr.inference.checkpoint import load_model_from_checkpoint
        from chant_omr.inference.onnx_decode import (
            load_onnx_models,
            onnx_encoder_infer,
            onnx_logits_func_cached,
        )
        from chant_omr.inference.preprocess import prepare_inference_tensor

        cfg_path, ckpt_path = config_and_ckpt

        # --- PyTorch greedy ---
        model, tokenizer, _meta = load_model_from_checkpoint(
            ckpt_path, config_path=cfg_path, device=torch.device("cpu"),
        )
        pixel_values = prepare_inference_tensor(tiny_png)

        with torch.inference_mode():
            memory_pt = model.encode(pixel_values)
            enc_mask = torch.ones(1, memory_pt.shape[1])
            logits_fn_pt = pytorch_logits_func_cached(model, enc_mask)
            pt_tokens = greedy_decode_generic(
                logits_fn_pt,
                memory_pt,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
                max_length=32,
                repetition_penalty=1.0,
            )

        # --- ONNX greedy ---
        enc, _dec, init, step, _manifest, tok = load_onnx_models(onnx_model_dir)
        pixel_np = prepare_inference_numpy(tiny_png)
        memory_np = onnx_encoder_infer(enc, pixel_np)
        memory_t = torch.from_numpy(memory_np)
        mask_np = np.ones((1, memory_np.shape[1]), dtype=np.float32)
        logits_fn_onnx = onnx_logits_func_cached(init, step, mask_np)

        onnx_tokens = greedy_decode_generic(
            logits_fn_onnx,
            memory_t,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
            max_length=32,
            repetition_penalty=1.0,
        )

        assert pt_tokens == onnx_tokens, (
            f"Token mismatch:\n  PyTorch: {pt_tokens}\n  ONNX:    {onnx_tokens}"
        )
