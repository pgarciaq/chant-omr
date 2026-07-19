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

DEFAULT_GRAMMAR_PENALTY = float("-inf")


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
        penalty: float = DEFAULT_GRAMMAR_PENALTY,
    ):
        self._table = paren_table
        self._eos_id = eos_token_id
        self._vocab_size = vocab_size
        self._penalty = penalty
        self._depth = 0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def penalty(self) -> float:
        return self._penalty

    def clone(self) -> GrammarMask:
        """Create an independent copy (for beam search branching)."""
        copy = GrammarMask(self._table, self._eos_id, self._vocab_size, self._penalty)
        copy._depth = self._depth
        return copy

    def update(self, token_id: int) -> None:
        """Advance state after a token has been selected."""
        if token_id < len(self._table):
            self._depth += self._table[token_id].net_delta
            self._depth = max(0, self._depth)

    def get_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """Return ``(vocab_size,)`` additive logit mask.

        Allowed tokens get ``0.0``; forbidden tokens get ``self._penalty``
        (default ``-inf`` for hard masking, or a finite negative value like
        ``-10.0`` for soft penalty mode — see #57).
        """
        p = self._penalty
        mask = torch.zeros(self._vocab_size, device=device)
        for tid in range(self._vocab_size):
            if tid >= len(self._table):
                continue
            info = self._table[tid]

            # Rule 1: token's closes must not exceed current depth
            if self._depth + info.min_running < 0:
                mask[tid] = p
                continue

            # Rule 2: no nested opens (exception: balanced tokens like ")(")
            if self._depth > 0 and info.opens > 0:
                if self._depth + info.min_running > 0 and info.opens > 0:
                    mask[tid] = p
                    continue

            # Rule 3: EOS only when depth == 0
            if tid == self._eos_id and self._depth + info.net_delta != 0:
                mask[tid] = p

        if self._depth > 0:
            mask[self._eos_id] = p

        return mask
