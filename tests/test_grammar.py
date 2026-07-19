"""Tests for grammar-constrained decoding (#37, #57)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from chant_omr.inference.grammar import (
    DEFAULT_GRAMMAR_PENALTY,
    GrammarMask,
    TokenParenInfo,
    _analyse_token_string,
    build_paren_table,
)
from chant_omr.model.tokenizer import train_tokenizer

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"
NEG_INF = float("-inf")


class TestAnalyseTokenString:
    def test_no_parens(self):
        info = _analyse_token_string("abc")
        assert info == TokenParenInfo(opens=0, closes=0, net_delta=0, min_running=0)

    def test_single_open(self):
        info = _analyse_token_string("(fg")
        assert info.opens == 1
        assert info.closes == 0
        assert info.net_delta == 1
        assert info.min_running == 0

    def test_single_close(self):
        info = _analyse_token_string("g)")
        assert info.opens == 0
        assert info.closes == 1
        assert info.net_delta == -1
        assert info.min_running == -1

    def test_balanced(self):
        info = _analyse_token_string("(fg)")
        assert info.opens == 1
        assert info.closes == 1
        assert info.net_delta == 0
        assert info.min_running == 0

    def test_close_then_open(self):
        info = _analyse_token_string(")(")
        assert info.opens == 1
        assert info.closes == 1
        assert info.net_delta == 0
        assert info.min_running == -1

    def test_double_open(self):
        info = _analyse_token_string("((")
        assert info.opens == 2
        assert info.closes == 0
        assert info.net_delta == 2
        assert info.min_running == 0

    def test_empty_string(self):
        info = _analyse_token_string("")
        assert info == TokenParenInfo(opens=0, closes=0, net_delta=0, min_running=0)


class TestGrammarMask:
    @pytest.fixture
    def simple_table(self):
        """4-token vocabulary: pad=0, bos=1, eos=2, '('=3, ')'=4, 'abc'=5."""
        return [
            TokenParenInfo(0, 0, 0, 0),   # 0: pad
            TokenParenInfo(0, 0, 0, 0),   # 1: bos
            TokenParenInfo(0, 0, 0, 0),   # 2: eos
            TokenParenInfo(1, 0, 1, 0),   # 3: "("
            TokenParenInfo(0, 1, -1, -1), # 4: ")"
            TokenParenInfo(0, 0, 0, 0),   # 5: "abc"
        ]

    def test_initial_depth_zero(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        assert gm.depth == 0

    def test_forbids_close_at_depth_zero(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        mask = gm.get_mask()
        assert mask[4].item() == NEG_INF  # ")" forbidden at depth 0

    def test_allows_open_at_depth_zero(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        mask = gm.get_mask()
        assert mask[3].item() == 0.0  # "(" allowed at depth 0

    def test_allows_eos_at_depth_zero(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        mask = gm.get_mask()
        assert mask[2].item() == 0.0  # eos allowed at depth 0

    def test_forbids_eos_at_depth_one(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        gm.update(3)  # opened "("
        assert gm.depth == 1
        mask = gm.get_mask()
        assert mask[2].item() == NEG_INF  # eos forbidden while paren open

    def test_allows_close_at_depth_one(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        gm.update(3)  # opened "("
        mask = gm.get_mask()
        assert mask[4].item() == 0.0  # ")" allowed at depth 1

    def test_forbids_nested_open(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        gm.update(3)  # opened "("
        mask = gm.get_mask()
        assert mask[3].item() == NEG_INF  # nested "(" forbidden

    def test_depth_returns_to_zero_after_close(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        gm.update(3)  # "("
        gm.update(4)  # ")"
        assert gm.depth == 0
        mask = gm.get_mask()
        assert mask[2].item() == 0.0  # eos allowed again

    def test_clone_is_independent(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        gm.update(3)  # depth 1
        clone = gm.clone()
        clone.update(4)  # depth 0 in clone only
        assert gm.depth == 1
        assert clone.depth == 0

    def test_text_tokens_always_allowed(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        mask = gm.get_mask()
        assert mask[5].item() == 0.0  # "abc" always allowed


class TestGrammarMaskWithBalancedTokens:
    """Test tokens that contain both open and close parens."""

    def test_balanced_token_allowed_at_depth_zero(self):
        table = [
            TokenParenInfo(0, 0, 0, 0),  # 0: pad
            TokenParenInfo(0, 0, 0, 0),  # 1: bos
            TokenParenInfo(0, 0, 0, 0),  # 2: eos
            TokenParenInfo(1, 1, 0, 0),  # 3: "(fg)" — balanced, min_running=0
        ]
        gm = GrammarMask(table, eos_token_id=2, vocab_size=4)
        mask = gm.get_mask()
        assert mask[3].item() == 0.0  # balanced token ok at depth 0

    def test_close_then_open_token_at_depth_one(self):
        table = [
            TokenParenInfo(0, 0, 0, 0),   # 0: pad
            TokenParenInfo(0, 0, 0, 0),   # 1: bos
            TokenParenInfo(0, 0, 0, 0),   # 2: eos
            TokenParenInfo(1, 0, 1, 0),   # 3: "(" — opens paren
            TokenParenInfo(1, 1, 0, -1),  # 4: ")(fg" — close then open, min_running=-1
        ]
        gm = GrammarMask(table, eos_token_id=2, vocab_size=5)
        gm.update(3)  # open paren, depth -> 1
        assert gm.depth == 1
        mask = gm.get_mask()
        # At depth 1, token ")(fg": min_running=-1, depth+min_running=0 >= 0, allowed
        # It closes the existing paren (depth 0), then re-opens (depth 1).
        assert mask[4].item() == 0.0


class TestBuildParenTable:
    def test_table_length_matches_vocab(self, tmp_path):
        tokenizer = train_tokenizer(
            GABC_FIXTURES,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )
        table = build_paren_table(tokenizer)
        assert len(table) == tokenizer.vocab_size

    def test_special_tokens_are_neutral(self, tmp_path):
        tokenizer = train_tokenizer(
            GABC_FIXTURES,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )
        table = build_paren_table(tokenizer)
        for sid in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id):
            assert table[sid].net_delta == 0
            assert table[sid].opens == 0
            assert table[sid].closes == 0


class TestGreedyDecodeWithGrammar:
    """Integration: grammar mask prevents invalid output in greedy decode."""

    def test_greedy_never_produces_negative_depth(self, tmp_path):
        from chant_omr.inference.beam_search import greedy_decode_generic

        tokenizer = train_tokenizer(
            GABC_FIXTURES,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )
        table = build_paren_table(tokenizer)
        gm = GrammarMask(table, tokenizer.eos_id, tokenizer.vocab_size)

        call_count = [0]

        def fake_logits_fn(input_ids, memory):
            call_count[0] += 1
            logits = torch.randn(tokenizer.vocab_size)
            if call_count[0] > 20:
                logits[tokenizer.eos_id] = 100.0
            return logits

        memory = torch.zeros(1, 10, 512)
        tokens = greedy_decode_generic(
            fake_logits_fn,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=50,
            grammar_mask=gm,
        )

        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        depth = 0
        for ch in decoded:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            assert depth >= 0, f"negative depth in: {decoded}"
        assert depth == 0 or tokens[-1] != tokenizer.eos_id


class TestBeamSearchDecodeWithGrammar:
    """Integration: grammar mask prevents invalid output in beam search."""

    def test_beam_search_produces_balanced_parens(self, tmp_path):
        from chant_omr.inference.beam_search import beam_search_decode_generic

        tokenizer = train_tokenizer(
            GABC_FIXTURES,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )
        table = build_paren_table(tokenizer)
        gm = GrammarMask(table, tokenizer.eos_id, tokenizer.vocab_size)

        call_count = [0]

        def fake_logits_fn(input_ids, memory):
            call_count[0] += 1
            logits = torch.randn(tokenizer.vocab_size)
            if call_count[0] > 30:
                logits[tokenizer.eos_id] = 100.0
            return logits

        memory = torch.zeros(1, 10, 512)
        tokens = beam_search_decode_generic(
            fake_logits_fn,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=50,
            beam_width=3,
            grammar_mask=gm,
        )

        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        depth = 0
        for ch in decoded:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            assert depth >= 0, f"negative depth in: {decoded}"


class TestSoftPenaltyMode:
    """Soft penalty (#57): finite negative penalty instead of -inf."""

    @pytest.fixture
    def simple_table(self):
        return [
            TokenParenInfo(0, 0, 0, 0),   # 0: pad
            TokenParenInfo(0, 0, 0, 0),   # 1: bos
            TokenParenInfo(0, 0, 0, 0),   # 2: eos
            TokenParenInfo(1, 0, 1, 0),   # 3: "("
            TokenParenInfo(0, 1, -1, -1), # 4: ")"
            TokenParenInfo(0, 0, 0, 0),   # 5: "abc"
        ]

    def test_default_penalty_is_neg_inf(self):
        assert math.isinf(DEFAULT_GRAMMAR_PENALTY) and DEFAULT_GRAMMAR_PENALTY < 0

    def test_hard_mask_uses_neg_inf(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6)
        mask = gm.get_mask()
        assert mask[4].item() == NEG_INF

    def test_soft_penalty_uses_finite_value(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-10.0)
        mask = gm.get_mask()
        assert mask[4].item() == -10.0
        assert mask[3].item() == 0.0

    def test_soft_penalty_on_eos_at_depth_one(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-5.0)
        gm.update(3)  # "(" -> depth 1
        mask = gm.get_mask()
        assert mask[2].item() == -5.0

    def test_soft_penalty_on_nested_open(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-20.0)
        gm.update(3)  # depth 1
        mask = gm.get_mask()
        assert mask[3].item() == -20.0

    def test_clone_preserves_penalty(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-7.5)
        clone = gm.clone()
        assert clone.penalty == -7.5

    def test_penalty_property(self, simple_table):
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-3.0)
        assert gm.penalty == -3.0

    def test_soft_penalty_allows_override_by_strong_logit(self, simple_table):
        """With soft penalty, a confident model can still pick the penalized token."""
        gm = GrammarMask(simple_table, eos_token_id=2, vocab_size=6, penalty=-5.0)
        logits = torch.zeros(6)
        logits[4] = 100.0
        penalized = logits + gm.get_mask()
        assert penalized[4].item() == 95.0
        assert torch.argmax(penalized).item() == 4


class TestDecodeConfigGrammarPenalty:
    """DecodeConfig carries grammar_penalty to GrammarMask."""

    def test_default_is_neg_inf(self):
        from chant_omr.inference.beam_search import DecodeConfig
        cfg = DecodeConfig()
        assert math.isinf(cfg.grammar_penalty) and cfg.grammar_penalty < 0

    def test_custom_penalty(self):
        from chant_omr.inference.beam_search import DecodeConfig
        cfg = DecodeConfig(grammar_constrained=True, grammar_penalty=-10.0)
        assert cfg.grammar_penalty == -10.0
