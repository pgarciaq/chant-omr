"""Tests for token length audit (#33)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chant_omr.data.token_audit import (
    TokenLengthReport,
    audit_token_lengths,
    format_token_audit,
)
from chant_omr.model.tokenizer import train_tokenizer

GABC_FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"


@pytest.fixture
def tokenizer(tmp_path: Path):
    return train_tokenizer(
        GABC_FIXTURES,
        vocab_size=256,
        output_dir=tmp_path / "tokenizer",
        min_body_len=10,
        use_manifest=False,
    )


@pytest.fixture
def rendered_dir(tmp_path: Path) -> Path:
    rdir = tmp_path / "rendered"
    rdir.mkdir()
    from PIL import Image

    for i, body in enumerate(
        [
            "(c4) Ky(f)ri(gf)e(h) *() e(ixhi)lé(h)i(g)son.(f)",
            "(c3) Gló(hi)ri(h)a(g) in(h) ex(ij)cél(i)sis(h) De(gf)o.(f)",
            "(c2) San(d)ctus,(f) San(gh)ctus,(g) San(f)ctus.(f)",
        ],
    ):
        stem = f"{10000 + i}"
        gabc = f"name: test {i};\n%%\n{body}\n"
        (rdir / f"{stem}.gabc").write_text(gabc, encoding="utf-8")
        Image.new("RGB", (420, 120), color=(255, 255, 255)).save(rdir / f"{stem}.png")
    return rdir


class TestAuditTokenLengths:
    def test_basic_report(self, tokenizer, rendered_dir: Path):
        report = audit_token_lengths(rendered_dir, tokenizer, max_seq_len=2048)
        assert report.total_pairs == 3
        assert report.min_tokens > 0
        assert report.max_tokens >= report.min_tokens
        assert report.exceed_limit == 0

    def test_truncation_detected(self, tokenizer, rendered_dir: Path):
        report = audit_token_lengths(rendered_dir, tokenizer, max_seq_len=5)
        assert report.exceed_limit > 0

    def test_longest_stems_populated(self, tokenizer, rendered_dir: Path):
        report = audit_token_lengths(rendered_dir, tokenizer, top_n=2)
        assert len(report.longest_stems) == 2
        assert report.longest_stems[0][1] >= report.longest_stems[1][1]

    def test_empty_dir_raises(self, tokenizer, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no rendered pairs"):
            audit_token_lengths(empty, tokenizer)


class TestFormatTokenAudit:
    def test_format_no_truncation(self):
        report = TokenLengthReport(
            total_pairs=100,
            min_tokens=10,
            max_tokens=500,
            mean_tokens=200.0,
            p50=180,
            p75=300,
            p90=400,
            p95=450,
            p99=490,
            exceed_limit=0,
            limit=2048,
            longest_stems=[("abc", 500), ("def", 480)],
        )
        text = format_token_audit(report)
        assert "All samples fit" in text
        assert "abc: 500" in text

    def test_format_with_truncation(self):
        report = TokenLengthReport(
            total_pairs=1000,
            min_tokens=10,
            max_tokens=3000,
            mean_tokens=200.0,
            p50=180,
            p75=300,
            p90=400,
            p95=450,
            p99=2100,
            exceed_limit=15,
            limit=2048,
            longest_stems=[("longchant", 3000)],
        )
        text = format_token_audit(report)
        assert "TRUNCATED: 15" in text
        assert "longchant: 3000 tokens *** TRUNCATED" in text
