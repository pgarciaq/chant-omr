"""Evaluation metrics for Gregorian chant OMR (#14).

Metrics:
    - GABC Edit Distance (GED): normalized character-level Levenshtein
    - Neume accuracy: accuracy on parenthesized neume groups
    - Structural validity: lightweight parse checks on GABC output

Encoding-equivalence normalization (#47): GABC has multiple valid encodings
for the same visual neume.  ``normalize_gabc_group`` canonicalizes rendering-
only differences so that equivalent encodings compare as equal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NEUME_GROUP_RE = re.compile(r"\([^()]*\)")
CLEF_RE = re.compile(r"\(c[1-4]\)|$\(f[1-4]\)")

# Repeated-note shorthand: e.g. "hsss" → "hs/hs/hs" (tristropha).
# Matches a pitch letter (a-m) followed by a modifier (s or v) repeated 2-3x.
_REPEATED_NOTE_RE = re.compile(r"([a-m])([sv])(\2{1,2})")

# Oriscus orientation suffixes: o0, o1 → o
_ORISCUS_ORIENT_RE = re.compile(r"o[01]")

# Punctum inclinatum shape suffixes: G0, G1, G2 → G (for any A-M)
_INCLINATUM_SHAPE_RE = re.compile(r"([A-M])[012]")


@dataclass(frozen=True)
class GEDResult:
    """GABC Edit Distance result for a single pair."""

    raw_distance: int
    ref_len: int
    pred_len: int
    normalized: float

    # Equivalence-normalized (#47): may differ from raw when alternate
    # encodings of the same neume are present.
    norm_raw_distance: int | None = None
    norm_normalized: float | None = None


@dataclass(frozen=True)
class NeumeAccuracyResult:
    """Neume group accuracy result for a single pair."""

    correct: int
    total: int
    accuracy: float
    ref_groups: list[str]
    pred_groups: list[str]

    # Equivalence-normalized (#47)
    norm_correct: int | None = None
    norm_total: int | None = None
    norm_accuracy: float | None = None


@dataclass(frozen=True)
class StructuralValidityResult:
    """Structural validity check for a single GABC prediction."""

    is_valid: bool
    errors: list[str]


def _expand_repeated_note(m: re.Match[str]) -> str:
    """Expand repeated-note shorthand to canonical ``note/note/...`` form.

    ``hsss`` → ``hs/hs/hs``  (tristropha)
    ``hvv``  → ``hv/hv``     (bivirga)
    """
    pitch = m.group(1)
    mod = m.group(2)
    repeats = len(m.group(3)) + 1  # group(3) captures the *extra* copies
    return "/".join(f"{pitch}{mod}" for _ in range(repeats))


def normalize_gabc_group(raw: str) -> str:
    """Canonicalize a single neume-group string for equivalence comparison.

    Strips rendering-only differences so that ``(fg!h)`` and ``(fgh)`` compare
    as equal, ``(hsss)`` matches ``(hs/hs/hs)``, etc.

    The input should include parentheses, e.g. ``"(fgh)"``.  Returns the
    normalized form with parentheses.

    Rules applied (conservative v0 set):
        1. Strip glyph-break ``!`` and fusion ``@`` — they change visual glyph
           shape but not pitch content.
        2. Normalize space width: ``//`` → ``/``.
        3. Expand repeated-note shorthand: ``hsss`` → ``hs/hs/hs``.
        4. Normalize oriscus orientation: ``o0``, ``o1`` → ``o``.
        5. Normalize punctum inclinatum shape: ``G0``, ``G1``, ``G2`` → ``G``.
        6. Strip leading/trailing whitespace inside parens.
    """
    inner = raw
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]

    inner = inner.strip()

    inner = inner.replace("!", "").replace("@", "")
    inner = inner.replace("//", "/")
    inner = _REPEATED_NOTE_RE.sub(_expand_repeated_note, inner)
    inner = _ORISCUS_ORIENT_RE.sub("o", inner)
    inner = _INCLINATUM_SHAPE_RE.sub(r"\1", inner)

    return f"({inner})"


def normalize_gabc_body(body: str) -> str:
    """Normalize all neume groups in a GABC body string.

    Applies ``normalize_gabc_group`` to every ``(...)`` group.  Text between
    groups is left untouched.
    """
    return NEUME_GROUP_RE.sub(lambda m: normalize_gabc_group(m.group(0)), body)


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
    capped to ``[0.0, 1.0]``.  Also computes equivalence-normalized variants
    (#47) where rendering-only encoding differences are canonicalized first.
    """
    pred = pred_body.strip()
    ref = ref_body.strip()
    raw = _levenshtein(pred, ref)
    denom = max(len(pred), len(ref))
    normalized = raw / denom if denom > 0 else 0.0

    norm_pred = normalize_gabc_body(pred)
    norm_ref = normalize_gabc_body(ref)
    norm_raw = _levenshtein(norm_pred, norm_ref)
    norm_denom = max(len(norm_pred), len(norm_ref))
    norm_normalized = norm_raw / norm_denom if norm_denom > 0 else 0.0

    return GEDResult(
        raw_distance=raw,
        ref_len=len(ref),
        pred_len=len(pred),
        normalized=min(normalized, 1.0),
        norm_raw_distance=norm_raw,
        norm_normalized=min(norm_normalized, 1.0),
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

    Also computes equivalence-normalized variants (#47) where each group is
    canonicalized via ``normalize_gabc_group`` before comparison.
    """
    pred_groups = extract_neume_groups(pred_body)
    ref_groups = extract_neume_groups(ref_body)
    total = max(len(pred_groups), len(ref_groups))

    if total == 0:
        return NeumeAccuracyResult(
            correct=0, total=0, accuracy=1.0,
            ref_groups=ref_groups, pred_groups=pred_groups,
            norm_correct=0, norm_total=0, norm_accuracy=1.0,
        )

    dist = _neume_group_lev(pred_groups, ref_groups)
    correct = total - dist
    accuracy = max(correct / total, 0.0)

    norm_pred = [normalize_gabc_group(g) for g in pred_groups]
    norm_ref = [normalize_gabc_group(g) for g in ref_groups]
    norm_total = max(len(norm_pred), len(norm_ref))
    norm_dist = _neume_group_lev(norm_pred, norm_ref)
    norm_correct = norm_total - norm_dist
    norm_accuracy = max(norm_correct / norm_total, 0.0) if norm_total > 0 else 1.0

    return NeumeAccuracyResult(
        correct=correct,
        total=total,
        accuracy=accuracy,
        ref_groups=ref_groups,
        pred_groups=pred_groups,
        norm_correct=norm_correct,
        norm_total=norm_total,
        norm_accuracy=norm_accuracy,
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
