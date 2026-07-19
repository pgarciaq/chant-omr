"""Tests for encoding-equivalence normalization (#47)."""

from __future__ import annotations

import pytest

from chant_omr.evaluation.metrics import (
    gabc_edit_distance,
    neume_accuracy,
    normalize_gabc_body,
    normalize_gabc_group,
)

# ---------------------------------------------------------------------------
# normalize_gabc_group — individual rules
# ---------------------------------------------------------------------------


class TestNormalizeGlyphBreak:
    """Rule 1: Strip ``!`` (glyph break) and ``@`` (fusion)."""

    def test_glyph_break_removed(self):
        assert normalize_gabc_group("(fg!h)") == "(fgh)"

    def test_multiple_glyph_breaks(self):
        assert normalize_gabc_group("(f!g!h)") == "(fgh)"

    def test_fusion_removed(self):
        assert normalize_gabc_group("(@fgh)") == "(fgh)"

    def test_no_effect_on_plain(self):
        assert normalize_gabc_group("(fgh)") == "(fgh)"


class TestNormalizeSpaceWidth:
    """Rule 2: Collapse ``//`` → ``/``."""

    def test_double_slash_collapsed(self):
        assert normalize_gabc_group("(fg//h)") == "(fg/h)"

    def test_single_slash_unchanged(self):
        assert normalize_gabc_group("(fg/h)") == "(fg/h)"


class TestNormalizeRepeatedNotes:
    """Rule 3: Expand repeated-note shorthand."""

    def test_tristropha(self):
        assert normalize_gabc_group("(hsss)") == "(hs/hs/hs)"

    def test_bistropha(self):
        assert normalize_gabc_group("(hss)") == "(hs/hs)"

    def test_bivirga(self):
        assert normalize_gabc_group("(hvv)") == "(hv/hv)"

    def test_trivirga(self):
        assert normalize_gabc_group("(hvvv)") == "(hv/hv/hv)"

    def test_already_expanded_form(self):
        assert normalize_gabc_group("(hs/hs/hs)") == "(hs/hs/hs)"

    def test_single_stropha_unchanged(self):
        assert normalize_gabc_group("(hs)") == "(hs)"


class TestNormalizeOriscusOrientation:
    """Rule 4: ``o0``, ``o1`` → ``o``."""

    def test_o0(self):
        assert normalize_gabc_group("(go0)") == "(go)"

    def test_o1(self):
        assert normalize_gabc_group("(go1)") == "(go)"

    def test_plain_oriscus_unchanged(self):
        assert normalize_gabc_group("(go)") == "(go)"


class TestNormalizeInclinatumShape:
    """Rule 5: ``G0``, ``G1``, ``G2`` → ``G``."""

    def test_descending_shape(self):
        assert normalize_gabc_group("(G0)") == "(G)"

    def test_ascending_shape(self):
        assert normalize_gabc_group("(G1)") == "(G)"

    def test_unison_shape(self):
        assert normalize_gabc_group("(G2)") == "(G)"

    def test_plain_inclinatum_unchanged(self):
        assert normalize_gabc_group("(G)") == "(G)"

    def test_other_inclinatum_letters(self):
        assert normalize_gabc_group("(H1)") == "(H)"
        assert normalize_gabc_group("(F0)") == "(F)"


class TestNormalizeWhitespace:
    """Rule 6: Strip leading/trailing whitespace inside parens."""

    def test_leading_space(self):
        assert normalize_gabc_group("( fgh)") == "(fgh)"

    def test_trailing_space(self):
        assert normalize_gabc_group("(fgh )") == "(fgh)"

    def test_both(self):
        assert normalize_gabc_group("( fgh )") == "(fgh)"


# ---------------------------------------------------------------------------
# normalize_gabc_group — combined rules
# ---------------------------------------------------------------------------


class TestNormalizeGroupCombined:
    """Multiple rules interacting."""

    def test_glyph_break_and_shorthand(self):
        assert normalize_gabc_group("(f!g!hsss)") == "(fghs/hs/hs)"

    def test_fusion_and_oriscus(self):
        assert normalize_gabc_group("(@go1)") == "(go)"

    def test_all_rules(self):
        result = normalize_gabc_group("( @f!g//hss )")
        assert result == "(fg/hs/hs)"


# ---------------------------------------------------------------------------
# normalize_gabc_body
# ---------------------------------------------------------------------------


class TestNormalizeGABCBody:
    def test_normalizes_all_groups(self):
        body = "(c4) text(fg!h) more(hsss)"
        expected = "(c4) text(fgh) more(hs/hs/hs)"
        assert normalize_gabc_body(body) == expected

    def test_text_between_groups_preserved(self):
        body = "(c4) Ky(f)ri(gf)e(h)"
        assert normalize_gabc_body(body) == body

    def test_clef_preserved(self):
        body = "(c4) test(f)"
        assert normalize_gabc_body(body) == "(c4) test(f)"

    def test_empty_body(self):
        assert normalize_gabc_body("") == ""

    def test_bars_preserved(self):
        body = "(c4) a(f) (::) b(g)"
        assert normalize_gabc_body(body) == "(c4) a(f) (::) b(g)"


# ---------------------------------------------------------------------------
# Equivalence pairs — normalization makes them equal
# ---------------------------------------------------------------------------


class TestEquivalencePairs:
    """Pairs that should compare as equal after normalization."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(fg!h)", "(fgh)"),
            ("(f!g!h)", "(fgh)"),
            ("(hsss)", "(hs/hs/hs)"),
            ("(hss)", "(hs/hs)"),
            ("(hvv)", "(hv/hv)"),
            ("(go0)", "(go)"),
            ("(go1)", "(go)"),
            ("(G0)", "(G)"),
            ("( fgh )", "(fgh)"),
            ("(@fgh)", "(fgh)"),
            ("(fg//h)", "(fg/h)"),
        ],
    )
    def test_groups_equivalent(self, a: str, b: str):
        assert normalize_gabc_group(a) == normalize_gabc_group(b)


# ---------------------------------------------------------------------------
# Non-equivalence pairs — normalization must NOT conflate these
# ---------------------------------------------------------------------------


class TestNonEquivalencePairs:
    """Pairs that must remain different after normalization."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(fgh)", "(fgj)"),
            ("(f)", "(g)"),
            ("(fg)", "(gf)"),
            ("(hs)", "(hv)"),
            ("(fgh~)", "(fgh)"),
            ("(fgh<)", "(fgh>)"),
        ],
    )
    def test_groups_not_equivalent(self, a: str, b: str):
        assert normalize_gabc_group(a) != normalize_gabc_group(b)


# ---------------------------------------------------------------------------
# Integration: normalization wired into metrics
# ---------------------------------------------------------------------------


class TestNormalizedGED:
    def test_equivalent_encodings_lower_norm_ged(self):
        pred = "(c4) Ky(fg!h)ri(hsss)e(go0)"
        ref = "(c4) Ky(fgh)ri(hs/hs/hs)e(go)"
        result = gabc_edit_distance(pred, ref)
        assert result.raw_distance > 0
        assert result.norm_raw_distance == 0
        assert result.norm_normalized == 0.0

    def test_identical_strings_both_zero(self):
        s = "(c4) Ky(f)ri(gf)e(h)"
        result = gabc_edit_distance(s, s)
        assert result.normalized == 0.0
        assert result.norm_normalized == 0.0

    def test_real_difference_not_masked(self):
        pred = "(c4) Ky(f)ri(j)e(h)"
        ref = "(c4) Ky(f)ri(g)e(h)"
        result = gabc_edit_distance(pred, ref)
        assert result.norm_raw_distance is not None
        assert result.norm_raw_distance > 0


class TestNormalizedNeumeAccuracy:
    def test_equivalent_groups_boost_accuracy(self):
        pred = "(c4) Ky(fg!h)ri(hsss)e(h)"
        ref = "(c4) Ky(fgh)ri(hs/hs/hs)e(h)"
        result = neume_accuracy(pred, ref)
        assert result.accuracy < 1.0
        assert result.norm_accuracy == 1.0

    def test_identical_strings(self):
        s = "(c4) Ky(f)ri(gf)e(h)"
        result = neume_accuracy(s, s)
        assert result.accuracy == 1.0
        assert result.norm_accuracy == 1.0

    def test_real_difference_not_masked(self):
        pred = "(c4) Ky(f)ri(j)e(h)"
        ref = "(c4) Ky(f)ri(g)e(h)"
        result = neume_accuracy(pred, ref)
        assert result.norm_accuracy is not None
        assert result.norm_accuracy < 1.0
