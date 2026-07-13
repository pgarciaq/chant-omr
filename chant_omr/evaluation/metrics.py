"""Evaluation metrics for Gregorian chant OMR (#14).

Metrics:
    - GABC Edit Distance (GED): normalized character-level Levenshtein
    - Neume accuracy: accuracy on parenthesized neume groups
    - Structural validity: lightweight parse checks on GABC output
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NEUME_GROUP_RE = re.compile(r"\([^()]*\)")
CLEF_RE = re.compile(r"\(c[1-4]\)|$\(f[1-4]\)")


@dataclass(frozen=True)
class GEDResult:
    """GABC Edit Distance result for a single pair."""

    raw_distance: int
    ref_len: int
    pred_len: int
    normalized: float


@dataclass(frozen=True)
class NeumeAccuracyResult:
    """Neume group accuracy result for a single pair."""

    correct: int
    total: int
    accuracy: float
    ref_groups: list[str]
    pred_groups: list[str]


@dataclass(frozen=True)
class StructuralValidityResult:
    """Structural validity check for a single GABC prediction."""

    is_valid: bool
    errors: list[str]


def _levenshtein(s: str, t: str) -> int:
    """Compute character-level Levenshtein distance between *s* and *t*."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[m]


def gabc_edit_distance(pred_body: str, ref_body: str) -> GEDResult:
    """Compute normalized GABC Edit Distance between prediction and reference.

    Uses symmetric normalization: ``max(len(pred), len(ref))`` as denominator,
    capped to ``[0.0, 1.0]``.
    """
    pred = pred_body.strip()
    ref = ref_body.strip()
    raw = _levenshtein(pred, ref)
    denom = max(len(pred), len(ref))
    normalized = raw / denom if denom > 0 else 0.0
    return GEDResult(
        raw_distance=raw,
        ref_len=len(ref),
        pred_len=len(pred),
        normalized=min(normalized, 1.0),
    )


def extract_neume_groups(body: str) -> list[str]:
    """Extract all parenthesized neume groups from a GABC body.

    Returns groups including parentheses, e.g. ``["(c4)", "(fg)", "(h)"]``.
    """
    return NEUME_GROUP_RE.findall(body)


def _neume_group_lev(pred_groups: list[str], ref_groups: list[str]) -> int:
    """Levenshtein at the neume-group level (sequence of groups)."""
    n, m = len(pred_groups), len(ref_groups)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if pred_groups[i - 1] == ref_groups[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[m]


def neume_accuracy(pred_body: str, ref_body: str) -> NeumeAccuracyResult:
    """Compute neume group accuracy between prediction and reference.

    Extracts ``(...)`` groups from both, computes sequence-level edit distance,
    and returns accuracy as ``1 - (edit_distance / max_groups)``.
    """
    pred_groups = extract_neume_groups(pred_body)
    ref_groups = extract_neume_groups(ref_body)
    total = max(len(pred_groups), len(ref_groups))

    if total == 0:
        return NeumeAccuracyResult(
            correct=0, total=0, accuracy=1.0,
            ref_groups=ref_groups, pred_groups=pred_groups,
        )

    dist = _neume_group_lev(pred_groups, ref_groups)
    correct = total - dist
    accuracy = max(correct / total, 0.0)

    return NeumeAccuracyResult(
        correct=correct,
        total=total,
        accuracy=accuracy,
        ref_groups=ref_groups,
        pred_groups=pred_groups,
    )


def check_structural_validity(gabc_body: str) -> StructuralValidityResult:
    """Check lightweight structural validity of a GABC body.

    Checks:
        1. Non-empty body
        2. Balanced parentheses
        3. At least one clef declaration ``(c1)``..``(c4)`` or ``(f1)``..``(f4)``

    Does NOT run ``gregorio`` compilation (that would be a stretch goal).
    """
    errors: list[str] = []
    body = gabc_body.strip()

    if not body:
        errors.append("empty body")
        return StructuralValidityResult(is_valid=False, errors=errors)

    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            errors.append("unbalanced parentheses: unexpected ')'")
            break
    if depth > 0:
        errors.append("unbalanced parentheses: unclosed '('")

    groups = extract_neume_groups(body)
    clefs = [g for g in groups if re.match(r"\([cf][1-4]\)", g)]
    if not clefs:
        errors.append("no clef declaration found")

    return StructuralValidityResult(is_valid=len(errors) == 0, errors=errors)
