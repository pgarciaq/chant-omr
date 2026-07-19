"""Grammar-constrained decoding for GABC output (#37).

Enforces balanced parentheses during autoregressive generation by masking
invalid next tokens. Each BPE token is analysed at tokenizer-load time for
its parenthesis effect (opens, closes, net delta), and at decode time a
lightweight state machine forbids tokens that would:

1. Drive parenthesis depth negative (unmatched ``)``)
2. Open a nested ``(`` when already inside ``()``
3. Allow EOS while parentheses are still open

This is the v1 grammar — richer syntax rules are tracked in #56.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chant_omr.model.tokenizer import GABCTokenizer

NEG_INF = float("-inf")


@dataclass(frozen=True)
class TokenParenInfo:
    """Parenthesis effect of a single BPE token's decoded string."""

    opens: int
    closes: int
    net_delta: int
    min_running: int  # minimum running depth *within* the token's characters


def _analyse_token_string(s: str) -> TokenParenInfo:
    """Walk characters and compute paren statistics."""
    opens = 0
    closes = 0
    running = 0
    min_running = 0
    for ch in s:
        if ch == "(":
            opens += 1
            running += 1
        elif ch == ")":
            closes += 1
            running -= 1
            min_running = min(min_running, running)
    return TokenParenInfo(
        opens=opens,
        closes=closes,
        net_delta=opens - closes,
        min_running=min_running,
    )


def build_paren_table(tokenizer: GABCTokenizer) -> list[TokenParenInfo]:
    """Precompute parenthesis info for every token in the vocabulary.

    Special tokens (pad, bos, eos, unk) get zero-effect entries.
    """
    table: list[TokenParenInfo] = []
    neutral = TokenParenInfo(opens=0, closes=0, net_delta=0, min_running=0)
    special_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id}

    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            table.append(neutral)
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        table.append(_analyse_token_string(decoded))
    return table


class GrammarMask:
    """Tracks parenthesis depth and produces a validity mask over the vocabulary.

    Stateful: call :meth:`update` after each token is selected to advance
    the grammar state. Call :meth:`get_mask` before token selection to get
    a ``(vocab_size,)`` tensor of ``0.0`` (allowed) or ``-inf`` (forbidden).
    """

    def __init__(
        self,
        paren_table: list[TokenParenInfo],
        eos_token_id: int,
        vocab_size: int,
    ):
        self._table = paren_table
        self._eos_id = eos_token_id
        self._vocab_size = vocab_size
        self._depth = 0

    @property
    def depth(self) -> int:
        return self._depth

    def clone(self) -> GrammarMask:
        """Create an independent copy (for beam search branching)."""
        copy = GrammarMask(self._table, self._eos_id, self._vocab_size)
        copy._depth = self._depth
        return copy

    def update(self, token_id: int) -> None:
        """Advance state after a token has been selected."""
        if token_id < len(self._table):
            self._depth += self._table[token_id].net_delta
            self._depth = max(0, self._depth)

    def get_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """Return ``(vocab_size,)`` additive mask: 0.0 = allowed, -inf = forbidden."""
        mask = torch.zeros(self._vocab_size, device=device)
        for tid in range(self._vocab_size):
            if tid >= len(self._table):
                continue
            info = self._table[tid]

            # Rule 1: token's closes must not exceed current depth
            # (check min_running: the depth must never go negative mid-token)
            if self._depth + info.min_running < 0:
                mask[tid] = NEG_INF
                continue

            # Rule 2: no nested opens — if depth > 0 (inside parens) and
            # token opens more parens, forbid it. Exception: if the token
            # also fully closes before re-opening (e.g. ")("), allow it.
            if self._depth > 0 and info.opens > 0:
                # A token like ")(" is OK at depth 1: it closes, then opens.
                # But "((" at depth 1 is nested. Check if any open happens
                # while already at depth > 0. We approximate: if the token's
                # min_running never drops to 0 (meaning it never fully closed
                # before opening again), and it has opens, that's nesting.
                if self._depth + info.min_running > 0 and info.opens > 0:
                    mask[tid] = NEG_INF
                    continue

            # Rule 3: EOS only when depth == 0
            if tid == self._eos_id and self._depth + info.net_delta != 0:
                mask[tid] = NEG_INF

        # Rule 3 (also): forbid EOS directly if depth > 0 and token is EOS
        if self._depth > 0:
            mask[self._eos_id] = NEG_INF

        return mask
