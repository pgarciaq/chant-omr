"""Tests for HuggingFace Hub upload, download, and model card (#39)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml
from click.testing import CliRunner

from chant_omr.cli import main
from chant_omr.hub import download_from_hub, generate_model_card, upload_to_hub

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"


@pytest.fixture
def tokenizer(tmp_path: Path):
    from chant_omr.model.tokenizer import train_tokenizer

    return train_tokenizer(
        GABC_FIXTURES,
        vocab_size=256,
        output_dir=tmp_path / "tokenizer",
        min_body_len=10,
        use_manifest=False,
    )


@pytest.fixture
def config_and_ckpt(tmp_path: Path, tokenizer) -> tuple[Path, Path]:
    """Produce a tiny config + Lightning checkpoint for testing."""
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

    from chant_omr.model.chant_omr_model import ChantOMRConfig, build_model
    from chant_omr.training.lightning_module import ChantOMRLightningModule

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
def hub_dir(tmp_path: Path, config_and_ckpt, tokenizer) -> Path:
    """Simulate a HuggingFace download directory with safetensors + config + tokenizer."""
    cfg_path, ckpt_path = config_and_ckpt
    d = tmp_path / "hub_cache"
    d.mkdir()

    from safetensors.torch import save_file

    from chant_omr.inference.checkpoint import load_model_from_checkpoint

    model, _tok, _meta = load_model_from_checkpoint(
        ckpt_path, config_path=cfg_path, device="cpu",
    )
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(d / "model.safetensors"))

    import shutil
    shutil.copy2(str(cfg_path), str(d / "config.yaml"))

    tok_dir = tmp_path / "tok"
    tok_src = tok_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(str(tok_src), str(d / "tokenizer.json"))

    return d


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_generate_model_card_basic(self):
        card = generate_model_card("pgquiles/chant-omr")
        assert "pgquiles/chant-omr" in card
        assert "library_name: chant-omr" in card
        assert "license: mit" in card
        assert "Transcoda" in card
        assert "GABC" in card
        assert "No evaluation results" in card

    def test_generate_model_card_with_eval(self):
        card = generate_model_card(
            "pgquiles/chant-omr",
            eval_results={"neume_accuracy": 0.971, "gregorio_compile": 0.99},
        )
        assert "0.9710" in card
        assert "0.9900" in card
        assert "neume_accuracy" in card

    def test_model_card_valid_yaml_frontmatter(self):
        card = generate_model_card("pgquiles/chant-omr")
        assert card.startswith("---\n")
        end = card.index("---\n", 4) + 4
        frontmatter = card[4:end - 4]
        parsed = yaml.safe_load(frontmatter)
        assert parsed["library_name"] == "chant-omr"
        assert "omr" in parsed["tags"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class TestDownloadFromHub:
    def test_download_calls_snapshot(self):
        with patch("huggingface_hub.snapshot_download", return_value="/tmp/hub/model") as mock_sd:
            result = download_from_hub("pgquiles/chant-omr")
            mock_sd.assert_called_once_with("pgquiles/chant-omr")
            assert result == Path("/tmp/hub/model")

    def test_download_passes_revision(self):
        with patch("huggingface_hub.snapshot_download", return_value="/tmp/hub/model") as mock_sd:
            download_from_hub("pgquiles/chant-omr", revision="v1.0")
            mock_sd.assert_called_once_with("pgquiles/chant-omr", revision="v1.0")


# ---------------------------------------------------------------------------
# Upload (mocked)
# ---------------------------------------------------------------------------


class TestUploadToHub:
    def test_upload_calls_hf_api(self, config_and_ckpt):
        cfg_path, ckpt_path = config_and_ckpt

        mock_api = MagicMock()
        with (
            patch("huggingface_hub.HfApi", return_value=mock_api),
            patch(
                "chant_omr.inference.export.export_safetensors",
                side_effect=lambda *a, **kw: None,
            ),
            patch(
                "chant_omr.inference.export.export_openvino",
                side_effect=lambda *a, **kw: None,
            ),
            patch(
                "chant_omr.inference.export.export_decoder_openvino",
                side_effect=lambda *a, **kw: None,
            ),
            patch(
                "chant_omr.inference.export.export_decoder_init_openvino",
                side_effect=lambda *a, **kw: None,
            ),
            patch(
                "chant_omr.inference.export.export_decoder_step_openvino",
                side_effect=lambda *a, **kw: None,
            ),
            patch(
                "chant_omr.inference.export.export_onnx",
                side_effect=lambda *a, **kw: None,
            ),
        ):
            url = upload_to_hub(ckpt_path, "pgquiles/chant-omr", config_path=cfg_path)

        assert url == "https://huggingface.co/pgquiles/chant-omr"
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()


# ---------------------------------------------------------------------------
# load_model_from_safetensors
# ---------------------------------------------------------------------------


class TestLoadModelFromSafetensors:
    def test_roundtrip(self, hub_dir):
        from chant_omr.inference.checkpoint import load_model_from_safetensors

        model, tokenizer, meta = load_model_from_safetensors(hub_dir)
        assert sum(p.numel() for p in model.parameters()) > 0
        assert tokenizer.vocab_size > 0
        assert "model_dir" in meta

    def test_missing_safetensors_raises(self, tmp_path):
        from chant_omr.inference.checkpoint import load_model_from_safetensors

        (tmp_path / "config.yaml").write_text(
            yaml.dump({"model": {"encoder_pretrained": False, "vocab_size": 256}}),
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            load_model_from_safetensors(tmp_path)

    def test_missing_config_raises(self, tmp_path):
        from chant_omr.inference.checkpoint import load_model_from_safetensors

        with pytest.raises(FileNotFoundError, match="config.yaml"):
            load_model_from_safetensors(tmp_path)


# ---------------------------------------------------------------------------
# CLI --model flag
# ---------------------------------------------------------------------------


class TestPredictModelFlag:
    def test_both_checkpoint_and_model_errors(self, tmp_path):
        runner = CliRunner()
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (100, 100)).save(img)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.touch()

        result = runner.invoke(
            main,
            ["predict", str(img), "--checkpoint", str(ckpt), "--model", "pgquiles/chant-omr"],
        )
        assert result.exit_code != 0
        exc_str = str(result.exception) if result.exception else ""
        assert "not both" in result.output or "not both" in exc_str

    def test_neither_checkpoint_nor_model_errors(self, tmp_path):
        runner = CliRunner()
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (100, 100)).save(img)

        result = runner.invoke(main, ["predict", str(img)])
        assert result.exit_code != 0

    def test_model_flag_triggers_download(self, tmp_path, hub_dir):
        runner = CliRunner()
        img = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (420, 120), color=(255, 255, 255)).save(img)

        with patch("chant_omr.hub.download_from_hub", return_value=hub_dir) as mock_dl:
            runner.invoke(
                main,
                ["predict", str(img), "--model", "pgquiles/chant-omr", "--device", "cpu"],
                catch_exceptions=False,
            )
            mock_dl.assert_called_once_with("pgquiles/chant-omr")


# ---------------------------------------------------------------------------
# CLI upload command
# ---------------------------------------------------------------------------


class TestUploadCLI:
    def test_upload_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["upload", "--help"])
        assert result.exit_code == 0
        assert "--repo-id" in result.output
