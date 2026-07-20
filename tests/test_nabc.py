"""Tests for NABC stripping, collapsing, header injection, and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from chant_omr.data import gregobase as gb
from chant_omr.data.gabc_parser import NABC_NEUME_RE, is_nabc_notation
from chant_omr.data.nabc import (
    CollapseStats,
    collapse_nabc_corpus,
    infer_nabc_lines,
    inject_nabc_header,
    strip_nabc_to_plain,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"
NABC_GABC = (FIXTURES / "nabc_sample.gabc").read_bytes()
VALID_GABC = (FIXTURES / "respice_domine.gabc").read_bytes()


class TestStripNabcToPlain:
    def test_single_neume(self):
        text = "name:test;\n%%\n(c|ta)"
        result = strip_nabc_to_plain(text)
        assert "(c)" in result
        assert "|" not in result.split("%%", 1)[1]

    def test_multi_neume_even_indexed(self):
        text = "name:test;\n%%\n(hih|toS2|he|clM)"
        result = strip_nabc_to_plain(text)
        assert "(hihhe)" in result

    def test_non_nabc_passthrough(self):
        text = "name:test;\n%%\n(c4) Ky(f)ri(gf)e(h)"
        result = strip_nabc_to_plain(text)
        assert result == text

    def test_headers_preserved(self):
        text = "name:NABC sample;\noffice-part:Alleluia;\nmode:3;\n%%\n(c4) AL(e|/ta)le(egf|toS2)lú(g)ia.(g) (::)\n"
        result = strip_nabc_to_plain(text)
        assert "name:NABC sample;" in result
        assert "office-part:Alleluia;" in result
        assert "mode:3;" in result

    def test_nabc_lines_header_removed(self):
        text = "name:test;\nnabc-lines: 1;\n%%\n(c|ta)"
        result = strip_nabc_to_plain(text)
        assert "nabc-lines" not in result

    def test_no_separator_unchanged(self):
        text = "name:test;\n%%\n(c4)"
        assert strip_nabc_to_plain(text) == text

    def test_fixture_file(self):
        text = NABC_GABC.decode("utf-8")
        result = strip_nabc_to_plain(text)
        body = result.split("%%", 1)[1]
        assert "|" not in body
        assert "(e)" in result or "(egf)" in result

    def test_multiple_groups(self):
        text = "name:test;\n%%\n(a|x)(b|y)(c)"
        result = strip_nabc_to_plain(text)
        assert "(a)" in result
        assert "(b)" in result
        assert "(c)" in result

    def test_text_between_groups_preserved(self):
        text = "name:test;\n%%\n(c4) Al(e|ta)le(g)lu(h|vi)ia(f)"
        result = strip_nabc_to_plain(text)
        assert " Al" in result
        assert "le" in result
        assert "lu" in result

    def test_no_body_marker(self):
        text = "just some text"
        assert strip_nabc_to_plain(text) == text


class TestCollapseNabcCorpus:
    def test_collapse_basic(self, tmp_path: Path):
        gabc_dir = tmp_path / "gabc"
        gabc_dir.mkdir()
        output_dir = tmp_path / "derived"

        (gabc_dir / "100.gabc").write_bytes(NABC_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia", incipit="Alleluia",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)
        catalog = [gb.CatalogEntry(100, "Alleluia", "Alleluia")]

        stats = collapse_nabc_corpus(
            gabc_dir, output_dir, manifest, catalog, only_if_plain_missing=False,
        )
        assert stats.collapsed == 1
        assert (output_dir / "100.gabc").exists()
        collapsed_text = (output_dir / "100.gabc").read_text(encoding="utf-8")
        assert "|" not in collapsed_text.split("%%", 1)[1]

    def test_collapse_skips_with_twin(self, tmp_path: Path):
        gabc_dir = tmp_path / "gabc"
        gabc_dir.mkdir()
        output_dir = tmp_path / "derived"

        (gabc_dir / "100.gabc").write_bytes(NABC_GABC)
        (gabc_dir / "200.gabc").write_bytes(VALID_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia",
                incipit="NABC sample",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
            gb.ManifestEntry(
                id=200, elem=None, office_part="Alleluia",
                incipit="NABC sample",
                filename="200.gabc", sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "NABC sample"),
            gb.CatalogEntry(200, "Alleluia", "NABC sample"),
        ]

        stats = collapse_nabc_corpus(
            gabc_dir, output_dir, manifest, catalog, only_if_plain_missing=True,
        )
        assert stats.skipped_has_twin == 1
        assert stats.collapsed == 0

    def test_collapse_all_ignores_twin(self, tmp_path: Path):
        gabc_dir = tmp_path / "gabc"
        gabc_dir.mkdir()
        output_dir = tmp_path / "derived"

        (gabc_dir / "100.gabc").write_bytes(NABC_GABC)
        (gabc_dir / "200.gabc").write_bytes(VALID_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia",
                incipit="NABC sample",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
            gb.ManifestEntry(
                id=200, elem=None, office_part="Alleluia",
                incipit="NABC sample",
                filename="200.gabc", sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "NABC sample"),
            gb.CatalogEntry(200, "Alleluia", "NABC sample"),
        ]

        stats = collapse_nabc_corpus(
            gabc_dir, output_dir, manifest, catalog, only_if_plain_missing=False,
        )
        assert stats.collapsed == 1
        assert stats.skipped_has_twin == 0


CORPUS_DIR = Path(__file__).parent.parent / "data" / "gregobase"


class TestCorpusValidation:
    """Validate strip_nabc_to_plain on the real corpus NABC files."""

    @pytest.mark.skipif(
        not CORPUS_DIR.exists(), reason="full corpus not available",
    )
    def test_all_nabc_files_strip_cleanly(self):
        manifest = gb.Manifest.load(CORPUS_DIR / gb.MANIFEST_FILENAME)
        nabc_ids = gb.scan_nabc_ids(CORPUS_DIR, manifest)
        assert len(nabc_ids) > 0, "expected at least one NABC file in corpus"

        failures: list[str] = []
        for entry in manifest.entries:
            if entry.id not in nabc_ids or entry.status != "ok" or not entry.filename:
                continue
            fpath = CORPUS_DIR / entry.filename
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            stripped = strip_nabc_to_plain(text)

            parts = stripped.split("%%", maxsplit=1)
            if len(parts) < 2:
                failures.append(f"{entry.filename}: no body after stripping")
                continue
            body = parts[1]
            if not body.strip():
                failures.append(f"{entry.filename}: empty body after stripping")
                continue
            remaining_pipes = NABC_NEUME_RE.findall(body)
            if remaining_pipes:
                failures.append(
                    f"{entry.filename}: {len(remaining_pipes)} pipe groups remain"
                )

        assert not failures, (
            f"{len(failures)} NABC files failed stripping:\n"
            + "\n".join(failures[:20])
        )
