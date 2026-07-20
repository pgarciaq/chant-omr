"""Tests for Gregorio renderer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chant_omr.data import gregobase as gb
from chant_omr.data import renderer as rd

FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"
RESPICE_GABC = FIXTURES / "respice_domine.gabc"


class TestRenderWorkers:
    def test_resolve_render_workers_explicit(self):
        assert rd.resolve_render_workers(4) == 4

    def test_resolve_render_workers_auto(self, monkeypatch):
        monkeypatch.setenv("CHANT_OMR_RENDER_WORKERS_MAX", "8")
        monkeypatch.setattr(rd.os, "cpu_count", lambda: 22)
        assert rd.resolve_render_workers(0) == 8

    def test_resolve_render_workers_auto_respects_cpu(self, monkeypatch):
        monkeypatch.setenv("CHANT_OMR_RENDER_WORKERS_MAX", "16")
        monkeypatch.setattr(rd.os, "cpu_count", lambda: 4)
        assert rd.resolve_render_workers(-1) == 4

    def test_default_render_workers_cap_invalid_env(self, monkeypatch):
        monkeypatch.setenv("CHANT_OMR_RENDER_WORKERS_MAX", "not-a-number")
        assert rd.default_render_workers_cap() == rd.DEFAULT_WORKERS_CAP

    def test_tex_cache_dir(self, tmp_path: Path):
        assert rd.tex_cache_dir(tmp_path / "rendered") == tmp_path / "rendered" / ".texcache"


class TestGabcHelpers:
    def test_has_double_header(self):
        single = RESPICE_GABC.read_text(encoding="utf-8")
        assert not rd.has_double_header(single)
        assert rd.has_double_header("name: a;\n%%\n(c4) a(f)\n%%\n(c4) b(f)")

    def test_body_only_gabc_text(self):
        raw = RESPICE_GABC.read_text(encoding="utf-8")
        body = rd.body_only_gabc_text(raw, name="Respice Domine")
        assert body.startswith("name: Respice Domine;\n%%\n")
        assert "office-part:" not in body
        assert "(c4) Re(f)spi(g)ce" in body

    def test_body_only_gabc_text_multi_header(self):
        raw = (FIXTURES / "double_header.gabc").read_text(encoding="utf-8")
        body = rd.body_only_gabc_text(raw, name="Auribus")
        assert "%%" not in body.split("%%", 1)[1]  # no second marker in output
        assert "(c3)AU(h)ri(h)bus" in body
        assert "annotation:" not in body

    def test_body_only_gabc_text_extra_headers(self):
        raw = RESPICE_GABC.read_text(encoding="utf-8")
        body = rd.body_only_gabc_text(
            raw, name="test", extra_headers={"nabc-lines": "1"},
        )
        assert "nabc-lines: 1;" in body
        header = body.split("%%", 1)[0]
        assert "nabc-lines: 1;" in header

    def test_extract_render_body(self):
        raw = (FIXTURES / "double_header.gabc").read_text(encoding="utf-8")
        assert rd.extract_render_body(raw).startswith("(c3)AU")

    def test_work_score_stem(self):
        assert rd.work_score_stem(20779, None) == "20779"
        assert rd.work_score_stem(500, 1) == "500_elem1"

    def test_png_filename(self):
        assert rd.png_filename(5000, None) == "5000.png"
        assert rd.png_filename(500, 1) == "500_elem1.png"

    def test_build_nomargin_tex(self):
        tex = rd.build_nomargin_tex("5000")
        assert r"\gregorioscore[a]{5000}" in tex
        assert r"\setmainfont{Libertinus Serif}" in tex
        assert r"\pagewidth=\wd\scorebox" in tex


class TestRenderJobs:
    def _manifest_with_entry(self, tmp_path: Path) -> tuple[Path, Path]:
        gabc_dir = tmp_path / "gregobase"
        rendered_dir = tmp_path / "rendered"
        gabc_dir.mkdir()
        rendered_dir.mkdir()
        (gabc_dir / "5000.gabc").write_bytes(RESPICE_GABC.read_bytes())
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=5000,
                    elem=None,
                    office_part="Introitus",
                    incipit="Respice Domine",
                    filename="5000.gabc",
                    sha256="abc",
                    size_bytes=100,
                    status="ok",
                    source="fixture",
                    error=None,
                )
            ]
        )
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)
        return gabc_dir, rendered_dir

    def test_iter_render_jobs_skips_existing_png(self, tmp_path: Path):
        gabc_dir, rendered_dir = self._manifest_with_entry(tmp_path)
        (rendered_dir / "5000.png").write_bytes(b"png")
        manifest = gb.Manifest.load(gabc_dir / gb.MANIFEST_FILENAME)
        jobs = list(rd.iter_render_jobs(manifest, gabc_dir, rendered_dir))
        assert jobs == []

    def test_iter_render_jobs_force(self, tmp_path: Path):
        gabc_dir, rendered_dir = self._manifest_with_entry(tmp_path)
        (rendered_dir / "5000.png").write_bytes(b"png")
        manifest = gb.Manifest.load(gabc_dir / gb.MANIFEST_FILENAME)
        jobs = list(rd.iter_render_jobs(manifest, gabc_dir, rendered_dir, force=True))
        assert len(jobs) == 1

    def test_append_failure_log(self, tmp_path: Path):
        path = tmp_path / "render_failures.jsonl"
        rd.append_failure_log(path, rd.RenderFailure(1, None, "1.gabc", "boom"))
        line = json.loads(path.read_text(encoding="utf-8").strip())
        assert line["id"] == 1
        assert line["error"] == "boom"


class TestRenderStats:
    def test_success_rate_zero_attempted(self):
        stats = rd.RenderStats()
        assert stats.success_rate == 0.0

    def test_success_rate_all_rendered(self):
        stats = rd.RenderStats(attempted=10, rendered=10)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        stats = rd.RenderStats(attempted=4, rendered=3)
        assert stats.success_rate == pytest.approx(0.75)

    def test_tally_rendered(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("rendered")
        assert stats.rendered == 1
        assert stats.failed == 0

    def test_tally_skipped(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("skipped")
        assert stats.skipped == 1

    def test_tally_skipped_nabc(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("skipped_nabc")
        assert stats.skipped == 1
        assert stats.skipped_nabc == 1
        assert stats.failed == 0

    def test_tally_rendered_nabc(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("rendered_nabc")
        assert stats.rendered == 1
        assert stats.rendered_nabc == 1
        assert stats.failed == 0

    def test_tally_failed_missing(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("failed_missing")
        assert stats.failed == 1
        assert stats.failed_missing == 1

    def test_tally_failed_compile(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("failed_compile")
        assert stats.failed == 1
        assert stats.failed_compile == 1

    def test_tally_unknown_counts_as_compile(self):
        stats = rd.RenderStats(attempted=1)
        stats._tally("something_unexpected")
        assert stats.failed == 1
        assert stats.failed_compile == 1

    def test_failure_category_field(self):
        f = rd.RenderFailure(1, None, "1.gabc", "boom", category="missing")
        d = f.to_dict()
        assert d["category"] == "missing"

    def test_failure_default_category(self):
        f = rd.RenderFailure(1, None, "1.gabc", "boom")
        assert f.category == "compile"


class TestRenderCorpusMocked:
    def test_nabc_skipped(self, tmp_path: Path):
        gabc_dir = tmp_path / "gregobase"
        rendered_dir = tmp_path / "rendered"
        gabc_dir.mkdir()
        (gabc_dir / "16305.gabc").write_bytes((FIXTURES / "nabc_sample.gabc").read_bytes())
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=16305,
                    elem=None,
                    office_part="Alleluia",
                    incipit="NABC sample",
                    filename="16305.gabc",
                    sha256="abc",
                    size_bytes=100,
                    status="ok",
                    source="fixture",
                    error=None,
                )
            ]
        )
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)

        with patch.object(rd, "toolchain_available", return_value=True):
            stats = rd.render_corpus(gabc_dir, rendered_dir, show_progress=False)

        assert stats.skipped == 1
        assert stats.skipped_nabc == 1
        assert stats.failed == 0

    def test_nabc_still_skipped_without_include_flag(self, tmp_path: Path):
        """Regression: default render (no --include-nabc) still skips NABC."""
        gabc_dir = tmp_path / "gregobase"
        rendered_dir = tmp_path / "rendered"
        gabc_dir.mkdir()
        (gabc_dir / "16305.gabc").write_bytes((FIXTURES / "nabc_sample.gabc").read_bytes())
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=16305, elem=None, office_part="Alleluia",
                    incipit="NABC sample", filename="16305.gabc",
                    sha256="abc", size_bytes=100, status="ok",
                    source="fixture", error=None,
                )
            ]
        )
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)

        with patch.object(rd, "toolchain_available", return_value=True):
            stats = rd.render_corpus(
                gabc_dir, rendered_dir, show_progress=False, include_nabc=False,
            )

        assert stats.skipped_nabc == 1
        assert stats.rendered_nabc == 0

    def test_missing_gabc_logs_failure(self, tmp_path: Path):
        gabc_dir = tmp_path / "gregobase"
        rendered_dir = tmp_path / "rendered"
        gabc_dir.mkdir()
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=99,
                    elem=None,
                    office_part="",
                    incipit="Missing",
                    filename="99.gabc",
                    sha256=None,
                    size_bytes=None,
                    status="ok",
                    source="test",
                    error=None,
                )
            ]
        )
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)

        with patch.object(rd, "toolchain_available", return_value=True):
            stats = rd.render_corpus(gabc_dir, rendered_dir, show_progress=False)

        assert stats.attempted == 1
        assert stats.failed == 1
        assert stats.failed_missing == 1
        failures = (rendered_dir / rd.FAILURES_FILENAME).read_text(encoding="utf-8")
        assert "missing GABC" in failures
        rec = json.loads(failures.strip())
        assert rec["category"] == "missing"


class TestCleanup:
    def _populate(self, d: Path) -> None:
        """Create a mix of paired, orphan gabc, and orphan png files."""
        (d / "10000.gabc").write_text("name:t;\n%%\n(c4) a(f)", encoding="utf-8")
        (d / "10000.png").write_bytes(b"PNG")
        (d / "--slug--.gabc").write_text("name:t;\n%%\n(c4) b(f)", encoding="utf-8")
        (d / "10001.png").write_bytes(b"PNG")

    def test_find_orphan_gabc(self, tmp_path: Path):
        self._populate(tmp_path)
        orphans = rd.find_orphan_gabc(tmp_path)
        assert len(orphans) == 1
        assert orphans[0].name == "--slug--.gabc"

    def test_find_png_only_orphans(self, tmp_path: Path):
        self._populate(tmp_path)
        orphans = rd.find_png_only_orphans(tmp_path)
        assert len(orphans) == 1
        assert orphans[0].name == "10001.png"

    def test_cleanup_dry_run(self, tmp_path: Path):
        self._populate(tmp_path)
        stats = rd.cleanup_rendered_dir(tmp_path, dry_run=True)
        assert stats.orphan_gabc_deleted == 1
        assert stats.orphan_png_deleted == 1
        assert (tmp_path / "--slug--.gabc").exists()
        assert (tmp_path / "10001.png").exists()

    def test_cleanup_delete(self, tmp_path: Path):
        self._populate(tmp_path)
        stats = rd.cleanup_rendered_dir(tmp_path, dry_run=False)
        assert stats.orphan_gabc_deleted == 1
        assert stats.orphan_png_deleted == 1
        assert not (tmp_path / "--slug--.gabc").exists()
        assert not (tmp_path / "10001.png").exists()
        assert (tmp_path / "10000.gabc").exists()
        assert (tmp_path / "10000.png").exists()

    def test_is_id_based_stem(self):
        assert rd._is_id_based_stem("10000")
        assert rd._is_id_based_stem("500_elem1")
        assert not rd._is_id_based_stem("--slug--")
        assert not rd._is_id_based_stem("a_progenie")


@pytest.mark.skipif(not rd.toolchain_available(), reason="Gregorio toolchain not installed")
class TestRenderIntegration:
    def test_render_fixture_gabc(self, tmp_path: Path):
        gabc_path = tmp_path / "5000.gabc"
        gabc_path.write_bytes(RESPICE_GABC.read_bytes())
        output = tmp_path / "5000.png"
        rd.render_gabc_to_image(gabc_path, output, dpi=150)
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_render_legacy_slug_with_id_stem(self, tmp_path: Path):
        slug_name = "--factum_est_autem--carmelite_tradition_p.81.gabc"
        gabc_path = tmp_path / slug_name
        gabc_path.write_bytes(RESPICE_GABC.read_bytes())
        output = tmp_path / "20779.png"
        rd.render_gabc_to_image(gabc_path, output, dpi=150, score_stem="20779")
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_render_multi_header_gabc(self, tmp_path: Path):
        gabc_path = tmp_path / "5000.gabc"
        gabc_path.write_bytes((FIXTURES / "double_header.gabc").read_bytes())
        output = tmp_path / "5000.png"
        rd.render_gabc_to_image(gabc_path, output, dpi=150)
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_render_corpus_from_manifest(self, tmp_path: Path):
        gabc_dir = tmp_path / "gregobase"
        rendered_dir = tmp_path / "rendered"
        gabc_dir.mkdir()
        (gabc_dir / "5000.gabc").write_bytes(RESPICE_GABC.read_bytes())
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=5000,
                    elem=None,
                    office_part="Introitus",
                    incipit="Respice Domine",
                    filename="5000.gabc",
                    sha256="abc",
                    size_bytes=100,
                    status="ok",
                    source="fixture",
                    error=None,
                )
            ]
        )
        manifest.save(gabc_dir / gb.MANIFEST_FILENAME)

        stats = rd.render_corpus(gabc_dir, rendered_dir, show_progress=False)
        assert stats.rendered == 1
        assert (rendered_dir / "5000.png").exists()
        assert (rendered_dir / "5000.gabc").exists()

    def test_render_nabc_with_include_flag(self, tmp_path: Path):
        """NABC file renders when include_nabc=True."""
        gabc_path = tmp_path / "16305.gabc"
        gabc_path.write_bytes((FIXTURES / "nabc_sample.gabc").read_bytes())
        output = tmp_path / "16305.png"
        rd.render_gabc_to_image(
            gabc_path, output, dpi=150, include_nabc=True,
        )
        assert output.exists()
        assert output.stat().st_size > 500

    def test_render_nabc_rejected_without_include_flag(self, tmp_path: Path):
        """NABC file raises ValueError without include_nabc."""
        gabc_path = tmp_path / "16305.gabc"
        gabc_path.write_bytes((FIXTURES / "nabc_sample.gabc").read_bytes())
        output = tmp_path / "16305.png"
        import pytest
        with pytest.raises(ValueError, match="NABC"):
            rd.render_gabc_to_image(gabc_path, output, dpi=150)
