"""Tests for encoding-equivalence normalization (#47, Option D).

Tests cover:
    - Conservative normalization table (only proven equivalences)
    - Non-equivalences verified via ``gregorio -S`` round-trip
    - Gregorio round-trip fallback
    - Integration with GED and neume accuracy metrics
"""

from __future__ import annotations

import pytest

from chant_omr.evaluation.metrics import (
    gabc_edit_distance,
    neume_accuracy,
    normalize_gabc_body,
    normalize_gabc_group,
)

# ---------------------------------------------------------------------------
# normalize_gabc_group — proven equivalences (verified with gregorio -S)
# ---------------------------------------------------------------------------


class TestNormalizeBareOriscus:
    """Bare ``go`` → ``go0`` (Gregorio defaults to descending)."""

    def test_bare_expanded(self):
        assert normalize_gabc_group("(go)") == "(go0)"

    def test_explicit_o0_unchanged(self):
        assert normalize_gabc_group("(go0)") == "(go0)"

    def test_o1_NOT_touched(self):
        assert normalize_gabc_group("(go1)") == "(go1)"

    def test_multiple_oriscus(self):
        assert normalize_gabc_group("(gofoh)") == "(go0fo0h)"

    def test_bare_with_pitch_prefix(self):
        assert normalize_gabc_group("(ho)") == "(ho0)"


class TestInclinatumNotNormalized:
    """Bare ``G`` is context-dependent in Gregorio — NOT safe to normalize.

    Alone: ``G`` → StansPunctumInclinatum (G2 equivalent).
    In descending series: ``G`` → DescendensPunctumInclinatum (G0 equivalent).
    """

    def test_bare_preserved(self):
        assert normalize_gabc_group("(G)") == "(G)"

    def test_G0_preserved(self):
        assert normalize_gabc_group("(G0)") == "(G0)"

    def test_G1_preserved(self):
        assert normalize_gabc_group("(G1)") == "(G1)"

    def test_G2_preserved(self):
        assert normalize_gabc_group("(G2)") == "(G2)"


class TestNormalizeWhitespace:
    """Strip leading/trailing whitespace inside parens."""

    def test_leading_space(self):
        assert normalize_gabc_group("( fgh)") == "(fgh)"

    def test_trailing_space(self):
        assert normalize_gabc_group("(fgh )") == "(fgh)"

    def test_both(self):
        assert normalize_gabc_group("( fgh )") == "(fgh)"


# ---------------------------------------------------------------------------
# normalize_gabc_group — intentionally NOT normalized (different .gtex)
# ---------------------------------------------------------------------------


class TestNotNormalized:
    """These modifiers change the Gregorio visual output and MUST stay."""

    def test_glyph_break_preserved(self):
        """``!`` changes glyph (scandicus vs pes+punctum)."""
        assert normalize_gabc_group("(fg!h)") != normalize_gabc_group("(fgh)")

    def test_fusion_preserved(self):
        """``@`` can change glyph connections."""
        assert "(fgh)" not in normalize_gabc_group("(@fgh)").replace("0", "")
        result = normalize_gabc_group("(@fgh)")
        assert "@" in result

    def test_double_slash_preserved(self):
        """``//`` is a different spacing than ``/``."""
        assert normalize_gabc_group("(fg//h)") != normalize_gabc_group("(fg/h)")

    def test_repeated_shorthand_preserved(self):
        """``hsss`` has different glyph structure than ``hs/hs/hs``."""
        assert normalize_gabc_group("(hsss)") != normalize_gabc_group("(hs/hs/hs)")

    def test_oriscus_o0_o1_different(self):
        """``o0`` (descending) ≠ ``o1`` (ascending)."""
        assert normalize_gabc_group("(go0)") != normalize_gabc_group("(go1)")

    def test_inclinatum_shapes_different(self):
        """G0, G1, G2 are visually different inclinatum shapes."""
        assert normalize_gabc_group("(G0)") != normalize_gabc_group("(G1)")
        assert normalize_gabc_group("(G0)") != normalize_gabc_group("(G2)")
        assert normalize_gabc_group("(G1)") != normalize_gabc_group("(G2)")


# ---------------------------------------------------------------------------
# normalize_gabc_group — proven equivalence pairs
# ---------------------------------------------------------------------------


class TestProvenEquivalences:
    """Pairs that ``gregorio -S`` confirms are identical output (table only)."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(go)", "(go0)"),
            ("( fgh )", "(fgh)"),
            ("( fgh)", "(fgh)"),
            ("(fgh )", "(fgh)"),
        ],
    )
    def test_groups_equivalent(self, a: str, b: str):
        assert normalize_gabc_group(a) == normalize_gabc_group(b)


class TestProvenNonEquivalences:
    """Pairs that ``gregorio -S`` confirms are different output."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(fg!h)", "(fgh)"),
            ("(fg//h)", "(fg/h)"),
            ("(hsss)", "(hs/hs/hs)"),
            ("(go0)", "(go1)"),
            ("(G0)", "(G1)"),
            ("(G0)", "(G2)"),
            ("(G)", "(G0)"),
            ("(fgh)", "(fgj)"),
            ("(f)", "(g)"),
            ("(fg)", "(gf)"),
        ],
    )
    def test_groups_not_equivalent(self, a: str, b: str):
        assert normalize_gabc_group(a) != normalize_gabc_group(b)


# ---------------------------------------------------------------------------
# normalize_gabc_body
# ---------------------------------------------------------------------------


class TestNormalizeGABCBody:
    def test_normalizes_all_groups(self):
        body = "(c4) text(go) more(hgf)"
        expected = "(c4) text(go0) more(hgf)"
        assert normalize_gabc_body(body) == expected

    def test_text_preserved(self):
        body = "(c4) Ky(f)ri(gf)e(h)"
        assert normalize_gabc_body(body) == "(c4) Ky(f)ri(gf)e(h)"

    def test_empty_body(self):
        assert normalize_gabc_body("") == ""

    def test_bars_preserved(self):
        body = "(c4) a(f) (::) b(g)"
        assert normalize_gabc_body(body) == "(c4) a(f) (::) b(g)"


# ---------------------------------------------------------------------------
# Gregorio round-trip fallback
# ---------------------------------------------------------------------------


class TestGregorioRoundtrip:
    """Tests that require ``gregorio`` to be installed."""

    @pytest.fixture(autouse=True)
    def _require_gregorio(self):
        from chant_omr.evaluation.gregorio_roundtrip import gregorio_available

        if not gregorio_available():
            pytest.skip("gregorio not installed")

    def test_identical_groups(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(fgh)", "(fgh)") is True

    def test_different_groups(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(fgh)", "(fgj)") is False

    def test_glyph_break_is_different(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(fg!h)", "(fgh)") is False

    def test_fusion_at_is_equivalent(self):
        """``@fgh`` and ``fgh`` produce identical .gtex for simple ascent."""
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(@fgh)", "(fgh)") is True

    def test_bare_oriscus_equivalent(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(go)", "(go0)") is True

    def test_oriscus_orientations_different(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(go0)", "(go1)") is False

    def test_bare_inclinatum_context_dependent(self):
        """Bare ``G`` alone != ``G0`` alone (stans vs descending)."""
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(G)", "(G0)") is False

    def test_inclinatum_shapes_different(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_groups_equivalent,
        )

        assert gregorio_groups_equivalent("(G0)", "(G1)") is False

    def test_bodies_equivalent(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_bodies_equivalent,
        )

        a = "(c4) Ky(f)ri(gf)e(h)"
        assert gregorio_bodies_equivalent(a, a) is True

    def test_bodies_different(self):
        from chant_omr.evaluation.gregorio_roundtrip import (
            gregorio_bodies_equivalent,
        )

        a = "(c4) Ky(f)ri(gf)e(h)"
        b = "(c4) Ky(f)ri(gj)e(h)"
        assert gregorio_bodies_equivalent(a, b) is False

    def test_compile_caches(self):
        """Verify repeated calls hit the LRU cache."""
        from chant_omr.evaluation.gregorio_roundtrip import _compile_gabc_body

        body = "(c4) test(fgh)"
        r1 = _compile_gabc_body(body)
        r2 = _compile_gabc_body(body)
        assert r1 is r2  # same object from cache


# ---------------------------------------------------------------------------
# Hybrid neume accuracy (table + round-trip)
# ---------------------------------------------------------------------------


class TestHybridNeumeAccuracy:
    """Integration: neume_accuracy uses 3-tier equivalence."""

    @pytest.fixture(autouse=True)
    def _require_gregorio(self):
        from chant_omr.evaluation.gregorio_roundtrip import gregorio_available

        if not gregorio_available():
            pytest.skip("gregorio not installed")

    def test_fusion_caught_by_roundtrip(self):
        """``@fgh`` vs ``fgh``: table says different, gregorio says same."""
        pred = "(c4) x(@fgh)"
        ref = "(c4) x(fgh)"
        result = neume_accuracy(pred, ref)
        assert result.accuracy < 1.0
        assert result.norm_accuracy == 1.0

    def test_real_error_not_masked(self):
        pred = "(c4) x(fgj)"
        ref = "(c4) x(fgh)"
        result = neume_accuracy(pred, ref)
        assert result.norm_accuracy < 1.0

    def test_glyph_break_stays_different(self):
        """``fg!h`` vs ``fgh`` should remain different even with gregorio."""
        pred = "(c4) x(fg!h)"
        ref = "(c4) x(fgh)"
        result = neume_accuracy(pred, ref)
        assert result.norm_accuracy < 1.0


# ---------------------------------------------------------------------------
# GED with corrected table normalization
# ---------------------------------------------------------------------------


class TestNormalizedGED:
    def test_bare_oriscus_lowers_ged(self):
        pred = "(c4) x(go)"
        ref = "(c4) x(go0)"
        result = gabc_edit_distance(pred, ref)
        assert result.raw_distance > 0
        assert result.norm_raw_distance == 0

    def test_identical_both_zero(self):
        s = "(c4) Ky(f)ri(gf)e(h)"
        result = gabc_edit_distance(s, s)
        assert result.normalized == 0.0
        assert result.norm_normalized == 0.0

    def test_real_diff_not_masked(self):
        pred = "(c4) Ky(f)ri(j)e(h)"
        ref = "(c4) Ky(f)ri(g)e(h)"
        result = gabc_edit_distance(pred, ref)
        assert result.norm_raw_distance is not None
        assert result.norm_raw_distance > 0

    def test_glyph_break_still_counted(self):
        """``!`` is NOT normalized away — GED should still see it."""
        pred = "(c4) x(fg!h)"
        ref = "(c4) x(fgh)"
        result = gabc_edit_distance(pred, ref)
        assert result.norm_raw_distance is not None
        assert result.norm_raw_distance > 0


# ---------------------------------------------------------------------------
# Gregorio compilation check (#46)
# ---------------------------------------------------------------------------


class TestGregorioCompilation:
    """Tests for check_gregorio_compilation()."""

    @pytest.fixture(autouse=True)
    def _require_gregorio(self):
        from chant_omr.evaluation.gregorio_roundtrip import gregorio_available

        if not gregorio_available():
            pytest.skip("gregorio not installed")

    def test_valid_gabc_compiles(self):
        from chant_omr.evaluation.gregorio_roundtrip import check_gregorio_compilation

        result = check_gregorio_compilation("(c4) Ky(f)ri(gf)e(h)")
        assert result.compiles is True
        assert result.errors == []

    def test_empty_body_fails(self):
        from chant_omr.evaluation.gregorio_roundtrip import check_gregorio_compilation

        result = check_gregorio_compilation("")
        assert result.compiles is False
        assert len(result.errors) > 0

    def test_unbalanced_parens_fails(self):
        from chant_omr.evaluation.gregorio_roundtrip import check_gregorio_compilation

        result = check_gregorio_compilation("(c4) Ky(f")
        assert result.compiles is False
        assert len(result.errors) > 0

    def test_valid_complex_body(self):
        from chant_omr.evaluation.gregorio_roundtrip import check_gregorio_compilation

        body = "(c4) Al(f)le(ghg)lú(h){ia}(hg) (::)"
        result = check_gregorio_compilation(body)
        assert result.compiles is True
        assert result.errors == []

    def test_errors_contain_message(self):
        from chant_omr.evaluation.gregorio_roundtrip import check_gregorio_compilation

        result = check_gregorio_compilation("(c4) x(ZZZZZ)")
        if not result.compiles:
            assert any(len(e) > 0 for e in result.errors)


class TestRunGregorio:
    """Tests for the _run_gregorio low-level function."""

    @pytest.fixture(autouse=True)
    def _require_gregorio(self):
        from chant_omr.evaluation.gregorio_roundtrip import gregorio_available

        if not gregorio_available():
            pytest.skip("gregorio not installed")

    def test_returns_gregorio_result(self):
        from chant_omr.evaluation.gregorio_roundtrip import GregorioResult, _run_gregorio

        result = _run_gregorio("(c4) x(fgh)")
        assert isinstance(result, GregorioResult)
        assert result.gtex is not None
        assert result.returncode == 0

    def test_compile_gabc_body_still_works(self):
        """Existing _compile_gabc_body wrapper returns same results as before."""
        from chant_omr.evaluation.gregorio_roundtrip import _compile_gabc_body

        gtex = _compile_gabc_body("(c4) x(fgh)")
        assert gtex is not None
        assert "Gre" in gtex

    def test_caching(self):
        from chant_omr.evaluation.gregorio_roundtrip import _run_gregorio

        _run_gregorio.cache_clear()
        body = "(c4) x(abc)"
        _run_gregorio(body)
        _run_gregorio(body)
        info = _run_gregorio.cache_info()
        assert info.hits >= 1
