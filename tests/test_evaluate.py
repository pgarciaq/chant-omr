"""Tests for evaluation metrics and harness (#14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chant_omr.evaluation.evaluate import (
    EvalReport,
    SampleResult,
    discover_benchmark_pairs,
    format_eval_report,
)
from chant_omr.evaluation.metrics import (
    _levenshtein,
    check_structural_validity,
    extract_neume_groups,
    gabc_edit_distance,
    neume_accuracy,
)

# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty_source(self):
        assert _levenshtein("", "abc") == 3

    def test_empty_target(self):
        assert _levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert _levenshtein("", "") == 0

    def test_substitution(self):
        assert _levenshtein("abc", "axc") == 1

    def test_insertion(self):
        assert _levenshtein("ac", "abc") == 1

    def test_deletion(self):
        assert _levenshtein("abc", "ac") == 1

    def test_longer(self):
        assert _levenshtein("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# GABC Edit Distance
# ---------------------------------------------------------------------------


class TestGABCEditDistance:
    def test_identical(self):
        result = gabc_edit_distance("(c4) Ky(f)ri(g)e(h)", "(c4) Ky(f)ri(g)e(h)")
        assert result.normalized == 0.0
        assert result.raw_distance == 0

    def test_completely_different(self):
        result = gabc_edit_distance("abc", "xyz")
        assert result.normalized == 1.0

    def test_partial_match(self):
        result = gabc_edit_distance("(c4) Ky(f)ri(g)e(h)", "(c4) Ky(f)ri(gx)e(h)")
        assert 0.0 < result.normalized < 1.0
        assert result.raw_distance == 1

    def test_symmetric_normalization(self):
        r1 = gabc_edit_distance("short", "a much longer string here")
        assert r1.normalized <= 1.0
        r2 = gabc_edit_distance("a much longer string here", "short")
        assert abs(r1.normalized - r2.normalized) < 1e-10

    def test_empty_pred_nonempty_ref(self):
        result = gabc_edit_distance("", "(c4) Ky(f)ri(g)e(h)")
        assert result.normalized == 1.0

    def test_both_empty(self):
        result = gabc_edit_distance("", "")
        assert result.normalized == 0.0

    def test_whitespace_stripping(self):
        result = gabc_edit_distance("  (c4) test  ", "(c4) test")
        assert result.normalized == 0.0


# ---------------------------------------------------------------------------
# Neume group extraction and accuracy
# ---------------------------------------------------------------------------


class TestExtractNeumeGroups:
    def test_simple(self):
        body = "(c4) Ky(f)ri(gf)e(h) *() e(ixhi)lé(h)i(g)son.(f)"
        groups = extract_neume_groups(body)
        assert groups == ["(c4)", "(f)", "(gf)", "(h)", "()", "(ixhi)", "(h)", "(g)", "(f)"]

    def test_empty(self):
        assert extract_neume_groups("no parens here") == []

    def test_double_bar(self):
        groups = extract_neume_groups("(::)")
        assert groups == ["(::)"]


class TestNeumeAccuracy:
    def test_identical(self):
        body = "(c4) Ky(f)ri(gf)e(h)"
        result = neume_accuracy(body, body)
        assert result.accuracy == 1.0
        assert result.correct == result.total

    def test_one_group_wrong(self):
        pred = "(c4) Ky(f)ri(gx)e(h)"
        ref = "(c4) Ky(f)ri(gf)e(h)"
        result = neume_accuracy(pred, ref)
        assert result.total == 4
        assert result.correct == 3
        assert result.accuracy == 0.75

    def test_extra_group_in_pred(self):
        pred = "(c4) Ky(f)ri(gf)e(h)(j)"
        ref = "(c4) Ky(f)ri(gf)e(h)"
        result = neume_accuracy(pred, ref)
        assert result.total == 5
        assert result.accuracy < 1.0

    def test_missing_group_in_pred(self):
        pred = "(c4) Ky(f)e(h)"
        ref = "(c4) Ky(f)ri(gf)e(h)"
        result = neume_accuracy(pred, ref)
        assert result.total == 4
        assert result.accuracy < 1.0

    def test_both_empty(self):
        result = neume_accuracy("no groups", "also no groups")
        assert result.accuracy == 1.0
        assert result.total == 0


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


class TestStructuralValidity:
    def test_valid(self):
        result = check_structural_validity("(c4) Ky(f)ri(gf)e(h)")
        assert result.is_valid is True
        assert result.errors == []

    def test_empty_body(self):
        result = check_structural_validity("")
        assert result.is_valid is False
        assert "empty body" in result.errors[0]

    def test_unbalanced_close(self):
        result = check_structural_validity("(c4) Ky(f)ri)e(h)")
        assert result.is_valid is False
        assert any("unexpected ')'" in e for e in result.errors)

    def test_unbalanced_open(self):
        result = check_structural_validity("(c4) Ky(f)ri(gfe(h)")
        assert result.is_valid is False
        assert any("unclosed '('" in e for e in result.errors)

    def test_no_clef(self):
        result = check_structural_validity("Ky(f)ri(gf)e(h)")
        assert result.is_valid is False
        assert any("no clef" in e for e in result.errors)

    def test_f_clef_valid(self):
        result = check_structural_validity("(f3) Ky(f)ri(gf)e(h)")
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Benchmark pair discovery
# ---------------------------------------------------------------------------


class TestDiscoverBenchmarkPairs:
    def test_discovers_matching_pairs(self, tmp_path):
        book_dir = tmp_path / "lpa1"
        book_dir.mkdir()
        (book_dir / "page_001.png").write_bytes(b"PNG")
        (book_dir / "page_001.gabc").write_text("name:t;\n%%\n(c4) test(f)", encoding="utf-8")
        (book_dir / "page_002.png").write_bytes(b"PNG")
        (book_dir / "page_002.gabc").write_text("name:t;\n%%\n(c4) more(g)", encoding="utf-8")

        pairs = discover_benchmark_pairs(tmp_path)
        assert len(pairs) == 2
        assert pairs[0][0].name == "page_001.png"
        assert pairs[0][1].name == "page_001.gabc"

    def test_skips_unpaired_png(self, tmp_path):
        (tmp_path / "solo.png").write_bytes(b"PNG")
        pairs = discover_benchmark_pairs(tmp_path)
        assert len(pairs) == 0

    def test_empty_directory(self, tmp_path):
        pairs = discover_benchmark_pairs(tmp_path)
        assert len(pairs) == 0

    def test_flat_directory(self, tmp_path):
        (tmp_path / "score_001.png").write_bytes(b"PNG")
        (tmp_path / "score_001.gabc").write_text("name:t;\n%%\n(c4) test(f)", encoding="utf-8")
        pairs = discover_benchmark_pairs(tmp_path)
        assert len(pairs) == 1



# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestEvalReport:
    def test_empty_report(self):
        report = EvalReport()
        text = format_eval_report(report)
        assert "No benchmark pairs found" in text

    def test_aggregate_metrics(self):
        s1 = SampleResult(
            image_path=Path("a.png"), ref_path=Path("a.gabc"),
            pred_body="(c4) test(f)", ref_body="(c4) test(f)",
            ged=gabc_edit_distance("(c4) test(f)", "(c4) test(f)"),
            neume_acc=neume_accuracy("(c4) test(f)", "(c4) test(f)"),
            validity=check_structural_validity("(c4) test(f)"),
            elapsed_s=0.5,
        )
        s2 = SampleResult(
            image_path=Path("b.png"), ref_path=Path("b.gabc"),
            pred_body="(c4) wrong(x)", ref_body="(c4) test(f)",
            ged=gabc_edit_distance("(c4) wrong(x)", "(c4) test(f)"),
            neume_acc=neume_accuracy("(c4) wrong(x)", "(c4) test(f)"),
            validity=check_structural_validity("(c4) wrong(x)"),
            elapsed_s=0.3,
        )
        report = EvalReport(samples=[s1, s2])
        assert report.count == 2
        assert report.mean_ged == pytest.approx(
            (s1.ged.normalized + s2.ged.normalized) / 2, abs=1e-6,
        )
        text = format_eval_report(report)
        assert "Evaluated 2 samples" in text
        assert "GED" in text
        assert "Neume" in text

    def test_skipped_in_report(self):
        report = EvalReport(
            skipped=[(Path("bad.gabc"), "empty gabc body")],
        )
        text = format_eval_report(report)
        assert "No benchmark pairs found" in text
        assert "bad.gabc" in text
