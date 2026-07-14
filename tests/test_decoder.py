"""Tests for Transformer decoder."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chant_omr.model.decoder import (
    ChantDecoder,
    DecoderConfig,
    KVCache,
    build_decoder,
    count_parameters,
)


@pytest.fixture
def config() -> DecoderConfig:
    return DecoderConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=1024,
        dropout=0.0,
        max_seq_len=2048,
        vocab_size=2048,
    )


@pytest.fixture
def decoder(config: DecoderConfig) -> ChantDecoder:
    return build_decoder(config)


class TestDecoderConfig:
    def test_from_mapping_defaults(self):
        cfg = DecoderConfig.from_mapping({})
        assert cfg.d_model == 512
        assert cfg.n_layers == 8
        assert cfg.vocab_size == 2048

    def test_from_yaml_model_section(self, tmp_path: Path):
        config_path = tmp_path / "default.yaml"
        config_path.write_text(
            yaml.dump({"model": {"d_model": 256, "n_layers": 2, "n_heads": 4, "d_ff": 512}}),
            encoding="utf-8",
        )
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = DecoderConfig.from_mapping(data["model"])
        assert cfg.d_model == 256
        assert cfg.n_layers == 2

    def test_rejects_invalid_head_split(self):
        with pytest.raises(ValueError, match="divisible"):
            DecoderConfig(d_model=512, n_heads=7)


class TestBuildDecoder:
    def test_returns_chant_decoder(self, decoder: ChantDecoder):
        assert isinstance(decoder, ChantDecoder)
        assert len(decoder.layers) == 8


class TestDecoderForward:
    def test_decoder_forward_shape(self, decoder: ChantDecoder):
        batch, seq_len, enc_len = 2, 16, 64
        input_ids = torch.randint(0, 2048, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)
        logits, _ = decoder(input_ids, encoder_memory)
        assert logits.shape == (batch, seq_len, 2048)

    def test_rejects_encoder_dim_mismatch(self, decoder: ChantDecoder):
        input_ids = torch.randint(0, 2048, (1, 8))
        encoder_memory = torch.randn(1, 32, 768)
        with pytest.raises(ValueError, match="d_model"):
            decoder(input_ids, encoder_memory)

    def test_rejects_batch_mismatch(self, decoder: ChantDecoder):
        input_ids = torch.randint(0, 2048, (2, 8))
        encoder_memory = torch.randn(1, 32, 512)
        with pytest.raises(ValueError, match="batch size"):
            decoder(input_ids, encoder_memory)

    def test_attention_mask_allows_padding(self, decoder: ChantDecoder):
        batch, seq_len, enc_len = 1, 6, 32
        input_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        encoder_memory = torch.randn(batch, enc_len, 512)
        logits, _ = decoder(input_ids, encoder_memory, attention_mask=attention_mask)
        assert logits.shape == (batch, seq_len, 2048)
        assert torch.isfinite(logits).all()


class TestCausalMask:
    def test_causal_mask(self, decoder: ChantDecoder):
        torch.manual_seed(0)
        batch, seq_len, enc_len = 1, 10, 32
        base_ids = torch.randint(1, 100, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)

        decoder.eval()
        with torch.no_grad():
            base_logits, _ = decoder(base_ids, encoder_memory)

        position = 4
        modified_ids = base_ids.clone()
        modified_ids[:, position + 1 :] = torch.randint(100, 200, (batch, seq_len - position - 1))
        with torch.no_grad():
            modified_logits, _ = decoder(modified_ids, encoder_memory)

        assert torch.allclose(
            base_logits[:, : position + 1],
            modified_logits[:, : position + 1],
            atol=1e-5,
            rtol=1e-5,
        )
        assert not torch.allclose(
            base_logits[:, position + 1 :],
            modified_logits[:, position + 1 :],
        )


class TestEncoderMask:
    def test_encoder_mask_blocks_padded_patches(self, decoder: ChantDecoder):
        torch.manual_seed(1)
        batch, seq_len, enc_len = 1, 6, 8
        input_ids = torch.randint(1, 50, (batch, seq_len))
        encoder_memory = torch.randn(batch, enc_len, 512)

        all_valid = torch.ones(batch, enc_len, dtype=torch.long)
        mask_head = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

        modified_memory = encoder_memory.clone()
        modified_memory[:, 4:, :] = 999.0

        decoder.eval()
        with torch.no_grad():
            logits_clean, _ = decoder(input_ids, encoder_memory, encoder_attention_mask=mask_head)
            logits_corrupt, _ = decoder(
                input_ids,
                modified_memory,
                encoder_attention_mask=mask_head,
            )
            logits_all, _ = decoder(input_ids, modified_memory, encoder_attention_mask=all_valid)

        assert torch.allclose(logits_clean, logits_corrupt, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(logits_clean, logits_all)


class TestKVCache:
    """Verify that cached decoding produces identical outputs to full decoding."""

    def test_cached_matches_non_cached(self, decoder: ChantDecoder):
        """Step-by-step cached decode must match full-prefix non-cached decode."""
        torch.manual_seed(42)
        batch, enc_len = 1, 16
        encoder_memory = torch.randn(batch, enc_len, 512)
        prefix = [1, 10, 20, 30, 40]

        decoder.eval()
        with torch.no_grad():
            full_ids = torch.tensor([prefix], dtype=torch.long)
            full_logits, _ = decoder(full_ids, encoder_memory)

            cache: KVCache | None = None
            for step, tok in enumerate(prefix):
                ids = torch.tensor([[tok]], dtype=torch.long)
                step_logits, cache = decoder(
                    ids,
                    encoder_memory,
                    past_key_values=cache,
                    use_cache=True,
                )
                assert torch.allclose(
                    step_logits[0, 0],
                    full_logits[0, step],
                    atol=1e-5,
                    rtol=1e-5,
                ), f"mismatch at step {step}"

    def test_cached_with_encoder_mask(self, decoder: ChantDecoder):
        """KV cache works correctly with encoder_attention_mask."""
        torch.manual_seed(7)
        batch, enc_len = 1, 8
        encoder_memory = torch.randn(batch, enc_len, 512)
        enc_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
        prefix = [1, 5, 15]

        decoder.eval()
        with torch.no_grad():
            full_ids = torch.tensor([prefix], dtype=torch.long)
            full_logits, _ = decoder(
                full_ids, encoder_memory, encoder_attention_mask=enc_mask
            )

            cache: KVCache | None = None
            for step, tok in enumerate(prefix):
                ids = torch.tensor([[tok]], dtype=torch.long)
                step_logits, cache = decoder(
                    ids,
                    encoder_memory,
                    encoder_attention_mask=enc_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                assert torch.allclose(
                    step_logits[0, 0],
                    full_logits[0, step],
                    atol=1e-5,
                    rtol=1e-5,
                ), f"mismatch at step {step}"

    def test_cache_returns_none_without_use_cache(self, decoder: ChantDecoder):
        """When use_cache=False, second return value is None."""
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mem = torch.randn(1, 4, 512)
        _, cache = decoder(ids, mem)
        assert cache is None

    def test_cache_returns_list_with_use_cache(self, decoder: ChantDecoder):
        """When use_cache=True, second return value is a list of LayerCache."""
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mem = torch.randn(1, 4, 512)
        _, cache = decoder(ids, mem, use_cache=True)
        assert cache is not None
        assert len(cache) == 8
        for lc in cache:
            assert lc.self_k.shape == (1, 8, 3, 64)
            assert lc.cross_k.shape == (1, 8, 4, 64)

    def test_cross_attention_cached_once(self, decoder: ChantDecoder):
        """Cross-attention K/V should be computed once and reused."""
        torch.manual_seed(0)
        mem = torch.randn(1, 6, 512)
        ids1 = torch.tensor([[1]], dtype=torch.long)
        ids2 = torch.tensor([[2]], dtype=torch.long)

        decoder.eval()
        with torch.no_grad():
            _, cache1 = decoder(ids1, mem, use_cache=True)
            assert cache1 is not None
            _, cache2 = decoder(ids2, mem, past_key_values=cache1, use_cache=True)
            assert cache2 is not None

        for lc1, lc2 in zip(cache1, cache2):
            assert torch.equal(lc1.cross_k, lc2.cross_k)
            assert torch.equal(lc1.cross_v, lc2.cross_v)
            assert lc2.self_k.shape[2] == 2


class TestParameterCount:
    def test_decoder_parameter_budget(self, decoder: ChantDecoder):
        params = count_parameters(decoder)
        assert 20_000_000 <= params <= 35_000_000
