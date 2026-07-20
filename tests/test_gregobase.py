"""Tests for GregoBase downloader."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from chant_omr.data import gregobase as gb

FIXTURES = Path(__file__).parent / "fixtures" / "gregobase"
VALID_GABC = (FIXTURES / "respice_domine.gabc").read_bytes()
ELEM_GABC = (FIXTURES / "haec_est_virgo.gabc").read_bytes()


def _mock_response(
    *,
    content: bytes = b"",
    text: str = "",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.content = content
    response.text = text
    response.status_code = status_code
    response.headers = headers or {}
    response.encoding = "utf-8"
    response.raise_for_status = MagicMock()
    return response


class TestParseCatalog:
    def test_parse_catalog_csv(self):
        text = (FIXTURES / "catalog.csv").read_text(encoding="utf-8")
        entries = gb.parse_catalog_csv(text)
        assert len(entries) == 3
        assert entries[0] == gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        assert entries[2] == gb.CatalogEntry(18698, "", "")

    def test_parse_catalog_date(self):
        header = "attachment; filename=gregobase_2026-07-10_17-19.csv"
        assert gb.parse_catalog_date(header) == "2026-07-10T17:19:00"


class TestParseUpdates:
    def test_parse_updates_html(self):
        html = (FIXTURES / "updates.html").read_text(encoding="utf-8")
        ids = gb.parse_updates_html(html)
        assert ids == [20782, 5000]


class TestGabcHelpers:
    def test_is_valid_gabc(self):
        assert gb.is_valid_gabc(VALID_GABC)
        assert not gb.is_valid_gabc(b"")
        assert not gb.is_valid_gabc(b"no marker here")
        assert not gb.is_valid_gabc(b"name:;\n%%\n")

    def test_content_disposition_filename(self):
        headers = {
            "Content-Disposition": "attachment; filename=in--respice_domine--dominican.gabc"
        }
        assert (
            gb.parse_content_disposition_filename(headers)
            == "in--respice_domine--dominican.gabc"
        )

    def test_disk_filename(self):
        assert gb.disk_filename(5000, None) == "5000.gabc"
        assert gb.disk_filename(500, 1) == "500_elem1.gabc"


class TestManifest:
    def test_save_and_load(self, tmp_path: Path):
        manifest = gb.Manifest(
            catalog_date="2026-07-10T17:19:00",
            entries=[
                gb.ManifestEntry(
                    id=5000,
                    elem=None,
                    office_part="Introitus",
                    incipit="Respice Domine",
                    filename="5000.gabc",
                    sha256=gb.sha256_bytes(VALID_GABC),
                    size_bytes=len(VALID_GABC),
                    status="ok",
                    source="live",
                    error=None,
                )
            ],
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = gb.Manifest.load(path)
        assert loaded.catalog_date == manifest.catalog_date
        assert len(loaded.entries) == 1
        assert loaded.entries[0].id == 5000

    def test_manifest_atomic_write(self, tmp_path: Path):
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=1,
                    elem=None,
                    office_part="",
                    incipit="a",
                    filename="1.gabc",
                    sha256="abc",
                    size_bytes=1,
                    status="ok",
                    source="live",
                    error=None,
                )
            ]
        )
        path = tmp_path / gb.MANIFEST_FILENAME
        manifest.save(path)
        assert path.exists()
        assert not path.with_name(path.name + gb.MANIFEST_TMP_SUFFIX).exists()
        assert gb.Manifest.load(path).entries[0].id == 1

    def test_replace_entries_preserves_other_ids(self):
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=1,
                    elem=None,
                    office_part="",
                    incipit="keep",
                    filename="1.gabc",
                    sha256="a",
                    size_bytes=1,
                    status="ok",
                    source="live",
                    error=None,
                ),
                gb.ManifestEntry(
                    id=2,
                    elem=None,
                    office_part="",
                    incipit="old",
                    filename="2.gabc",
                    sha256="b",
                    size_bytes=1,
                    status="ok",
                    source="live",
                    error=None,
                ),
            ]
        )
        replacement = [
            gb.ManifestEntry(
                id=2,
                elem=None,
                office_part="",
                incipit="old",
                filename="2.gabc",
                sha256="c",
                size_bytes=2,
                status="failed",
                source="live",
                error="no valid gabc",
            )
        ]
        previous = manifest.replace_entries_for_id(2, replacement)
        assert len(previous) == 1
        assert previous[0].sha256 == "b"
        assert [entry.id for entry in manifest.entries] == [1, 2]
        assert manifest.entries[0].incipit == "keep"
        assert manifest.entries[1].status == "failed"


class TestDownloadVariants:
    def _session_with_responses(self, responses: list[MagicMock]) -> MagicMock:
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get = MagicMock(side_effect=responses)
        return session

    def test_download_bare_success(self, tmp_path: Path):
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={
                        "Content-Disposition": "attachment; filename=in--respice.gabc"
                    },
                ),
                _mock_response(content=VALID_GABC),
            ]
        )
        entry = gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert len(paths) == 1
        assert paths[0].exists()
        assert new_count == 1
        assert entries[0].status == "ok"
        assert entries[0].elem is None
        assert paths[0].name == "5000.gabc"

    def test_save_elem_filename(self, tmp_path: Path):
        session = self._session_with_responses(
            [
                _mock_response(content=b""),
                _mock_response(
                    content=ELEM_GABC,
                    headers={
                        "Content-Disposition": "attachment; filename=al--haec.gabc"
                    },
                ),
                _mock_response(content=b""),
            ]
        )
        entry = gb.CatalogEntry(500, "Antiphona", "Haec est virgo")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert new_count == 1
        assert entries[0].elem == 1
        assert paths[0].name == "500_elem1.gabc"

    def test_save_uses_id_filename(self, tmp_path: Path):
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={
                        "Content-Disposition": (
                            "attachment; filename=in--respice_domine--dominican.gabc"
                        )
                    },
                ),
                _mock_response(content=VALID_GABC),
            ]
        )
        entry = gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert new_count == 1
        assert paths[0].name == "5000.gabc"
        assert entries[0].filename == "5000.gabc"

    def test_download_generic_content_disposition(self, tmp_path: Path):
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={"Content-Disposition": "attachment; filename=----.gabc"},
                ),
                _mock_response(content=VALID_GABC),
            ]
        )
        entry = gb.CatalogEntry(18698, "", "")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert new_count == 1
        assert paths[0].name == "18698.gabc"
        assert entries[0].filename == "18698.gabc"

    def test_download_elem_dedupe(self, tmp_path: Path):
        duplicate = VALID_GABC
        session = self._session_with_responses(
            [
                _mock_response(
                    content=duplicate,
                    headers={"Content-Disposition": "attachment; filename=a.gabc"},
                ),
                _mock_response(
                    content=duplicate,
                    headers={"Content-Disposition": "attachment; filename=b.gabc"},
                ),
            ]
        )
        entry = gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        _paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert new_count == 1
        assert len(entries) == 1

    def test_download_multi_variant(self, tmp_path: Path):
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={"Content-Disposition": "attachment; filename=bare.gabc"},
                ),
                _mock_response(
                    content=ELEM_GABC,
                    headers={"Content-Disposition": "attachment; filename=elem1.gabc"},
                ),
                _mock_response(content=b""),
            ]
        )
        entry = gb.CatalogEntry(500, "Antiphona", "Haec est virgo")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert new_count == 2
        assert len(entries) == 2
        assert {e.elem for e in entries} == {None, 1}
        assert len(paths) == 2
        assert paths[0].name == "500.gabc"
        assert paths[1].name == "500_elem1.gabc"
        assert gb.sha256_bytes(VALID_GABC) in {e.sha256 for e in entries}
        assert gb.sha256_bytes(ELEM_GABC) in {e.sha256 for e in entries}

    def test_download_invalid_gabc(self, tmp_path: Path):
        session = self._session_with_responses(
            [_mock_response(content=b"") for _ in range(gb.MAX_ELEM + 2)]
        )
        entry = gb.CatalogEntry(9999, "", "Missing")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert paths == []
        assert new_count == 0
        assert len(entries) == 1
        assert entries[0].status == "failed"
        assert entries[0].error == "empty body"

    def test_download_empty_body_stub(self, tmp_path: Path):
        stub = b"name:;\n%%\n"
        session = self._session_with_responses(
            [_mock_response(content=stub) for _ in range(gb.MAX_ELEM + 2)]
        )
        entry = gb.CatalogEntry(18698, "", "")
        manifest = gb.Manifest()
        limiter = gb.RateLimiter(0)

        paths, entries, _skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert paths == []
        assert new_count == 0
        assert len(entries) == 1
        assert entries[0].status == "failed"
        assert entries[0].error == "empty gabc body"

    def test_manifest_resume(self, tmp_path: Path):
        filename = "5000.gabc"
        dest = tmp_path / filename
        dest.write_bytes(VALID_GABC)
        digest = gb.sha256_bytes(VALID_GABC)
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=5000,
                    elem=None,
                    office_part="Introitus",
                    incipit="Respice Domine",
                    filename=filename,
                    sha256=digest,
                    size_bytes=len(VALID_GABC),
                    status="ok",
                    source="live",
                    error=None,
                )
            ]
        )
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={"Content-Disposition": "attachment; filename=legacy.gabc"},
                ),
                _mock_response(content=VALID_GABC),
            ]
        )
        entry = gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        limiter = gb.RateLimiter(0)

        paths, _entries, skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert skipped == 1
        assert new_count == 0
        assert len(paths) == 1

    def test_manifest_resume_legacy_slug(self, tmp_path: Path):
        filename = "in--respice.gabc"
        dest = tmp_path / filename
        dest.write_bytes(VALID_GABC)
        digest = gb.sha256_bytes(VALID_GABC)
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=5000,
                    elem=None,
                    office_part="Introitus",
                    incipit="Respice Domine",
                    filename=filename,
                    sha256=digest,
                    size_bytes=len(VALID_GABC),
                    status="ok",
                    source="live",
                    error=None,
                )
            ]
        )
        session = self._session_with_responses(
            [
                _mock_response(
                    content=VALID_GABC,
                    headers={"Content-Disposition": f"attachment; filename={filename}"},
                ),
                _mock_response(content=VALID_GABC),
            ]
        )
        entry = gb.CatalogEntry(5000, "Introitus", "Respice Domine")
        limiter = gb.RateLimiter(0)

        paths, _entries, skipped, new_count = gb.download_variants_for_id(
            session, entry, tmp_path, manifest, rate_limiter=limiter
        )

        assert skipped == 1
        assert new_count == 0
        assert paths[0].name == filename


class TestRateLimiter:
    def test_rate_limit(self):
        limiter = gb.RateLimiter(0.05)
        start = time.monotonic()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05


class TestRetry:
    def test_retry_transient(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        ok = _mock_response(content=b"ok")
        err = MagicMock(spec=requests.Response)
        err.status_code = 503
        err.raise_for_status = MagicMock()
        session.get = MagicMock(side_effect=[err, ok])

        with patch.object(gb, "time") as mock_time:
            mock_time.sleep = MagicMock()
            response = gb._request_with_retries(session, "http://example.test")
        assert response is ok


class TestSelectCatalogIds:
    def test_limit_flag(self):
        catalog = [
            gb.CatalogEntry(1, "", "a"),
            gb.CatalogEntry(2, "", "b"),
            gb.CatalogEntry(3, "", "c"),
        ]
        manifest = gb.Manifest(
            entries=[
                gb.ManifestEntry(
                    id=1,
                    elem=None,
                    office_part="",
                    incipit="a",
                    filename="1.gabc",
                    sha256="x",
                    size_bytes=1,
                    status="ok",
                    source="live",
                    error=None,
                )
            ]
        )
        selected = gb._select_catalog_ids(catalog, manifest, limit=1, sync_ids=None)
        assert [e.id for e in selected] == [2]

    def test_sync_limit_flag(self):
        catalog = [
            gb.CatalogEntry(1, "", "a"),
            gb.CatalogEntry(2, "", "b"),
            gb.CatalogEntry(3, "", "c"),
        ]
        manifest = gb.Manifest()
        selected = gb._select_catalog_ids(
            catalog, manifest, limit=0, sync_ids=[10, 20, 30], sync_limit=2
        )
        assert [e.id for e in selected] == [10, 20]


class TestDownloadCorpus:
    def test_download_corpus_with_mocks(self, tmp_path: Path):
        catalog_csv = (FIXTURES / "catalog.csv").read_text(encoding="utf-8")

        def fake_get(url, params=None, timeout=None):
            if url.endswith("csv.php"):
                return _mock_response(
                    text=catalog_csv,
                    headers={
                        "Content-Disposition": "attachment; filename=gregobase_2026-07-10_17-19.csv"
                    },
                )
            if "download.php" in url:
                chant_id = int(params["id"])
                if chant_id == 5000:
                    return _mock_response(
                        content=VALID_GABC,
                        headers={
                            "Content-Disposition": "attachment; filename=in--respice.gabc"
                        },
                    )
                return _mock_response(content=b"")
            raise AssertionError(f"unexpected url {url}")

        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get = MagicMock(side_effect=fake_get)

        with patch.object(gb, "make_session", return_value=session):
            stats = gb.download_corpus(tmp_path, limit=1, rate_limit=0)

        assert stats.attempted_ids == 1
        assert stats.downloaded_files >= 1
        manifest = gb.Manifest.load(tmp_path / gb.MANIFEST_FILENAME)
        assert manifest.catalog_date == "2026-07-10T17:19:00"
        assert any(e.id == 5000 and e.status == "ok" for e in manifest.entries)
        assert (tmp_path / "5000.gabc").exists()

    def test_delete_orphan_gabc_files(self, tmp_path: Path):
        legacy = tmp_path / "in--respice.gabc"
        legacy.write_bytes(VALID_GABC)
        new_file = tmp_path / "5000.gabc"
        new_file.write_bytes(VALID_GABC)
        previous = [
            gb.ManifestEntry(
                id=5000,
                elem=None,
                office_part="Introitus",
                incipit="Respice Domine",
                filename=legacy.name,
                sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC),
                status="ok",
                source="live",
                error=None,
            )
        ]
        new_entries = [
            gb.ManifestEntry(
                id=5000,
                elem=None,
                office_part="Introitus",
                incipit="Respice Domine",
                filename=new_file.name,
                sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC),
                status="ok",
                source="live",
                error=None,
            )
        ]
        gb._delete_orphan_gabc_files(tmp_path, previous, new_entries)
        assert not legacy.exists()
        assert new_file.exists()


# ---------------------------------------------------------------------------
# Manifest rebuild (#16)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_incipit_basic(self):
        assert gb.normalize_incipit("Respice Domine") == "respice domine"

    def test_normalize_incipit_accents(self):
        assert gb.normalize_incipit("Résumé café") == "resume cafe"

    def test_normalize_incipit_punctuation(self):
        assert gb.normalize_incipit("Domine, adiuva me!") == "domine adiuva me"

    def test_normalize_incipit_whitespace(self):
        assert gb.normalize_incipit("  hello   world  ") == "hello world"

    def test_normalize_incipit_empty(self):
        assert gb.normalize_incipit("") == ""

    def test_normalize_office_part_basic(self):
        assert gb.normalize_office_part("Introitus") == "introitus"

    def test_normalize_office_part_accents(self):
        assert gb.normalize_office_part("Répons") == "repons"


class TestRebuildManifest:
    def _make_catalog(self) -> list[gb.CatalogEntry]:
        return [
            gb.CatalogEntry(5000, "Introitus", "Respice Domine"),
            gb.CatalogEntry(500, "Antiphona", "Haec est virgo"),
            gb.CatalogEntry(18698, "", ""),
        ]

    def test_rebuild_numeric_filename(self, tmp_path: Path):
        (tmp_path / "5000.gabc").write_bytes(VALID_GABC)
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.matched == 1
        assert stats.skipped == 0
        entry = manifest.entries[0]
        assert entry.id == 5000
        assert entry.elem is None
        assert entry.source == "rebuilt"
        assert entry.status == "ok"

    def test_rebuild_elem_filename(self, tmp_path: Path):
        (tmp_path / "500_elem1.gabc").write_bytes(ELEM_GABC)
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.matched == 1
        entry = manifest.entries[0]
        assert entry.id == 500
        assert entry.elem == 1

    def test_rebuild_unique_header_match(self, tmp_path: Path):
        gabc = b"name: Haec est virgo;\noffice-part: Antiphona;\n%%\n(c4) Ha(f)ec(g)\n"
        (tmp_path / "mystery.gabc").write_bytes(gabc)
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.matched == 1
        assert manifest.entries[0].id == 500

    def test_rebuild_ambiguous_skipped(self, tmp_path: Path):
        catalog = [
            gb.CatalogEntry(100, "Introitus", "Alleluia"),
            gb.CatalogEntry(200, "Introitus", "Alleluia"),
        ]
        gabc = b"name: Alleluia;\noffice-part: Introitus;\n%%\n(c4) Al(f)\n"
        (tmp_path / "unknown.gabc").write_bytes(gabc)
        manifest, stats = gb.rebuild_manifest(tmp_path, catalog)
        assert stats.ambiguous == 1
        assert stats.matched == 0
        assert len(manifest.entries) == 0
        unmatched = (tmp_path / gb.REBUILD_UNMATCHED_FILE).read_text(encoding="utf-8")
        assert "unknown.gabc" in unmatched
        assert "ambiguous" in unmatched

    def test_rebuild_invalid_gabc_skipped(self, tmp_path: Path):
        (tmp_path / "bad.gabc").write_bytes(b"no marker here")
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.invalid == 1
        assert stats.matched == 0
        assert len(manifest.entries) == 0

    def test_rebuild_bak_written(self, tmp_path: Path):
        old_manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=1, elem=None, office_part="", incipit="old",
                filename="1.gabc", sha256="x", size_bytes=1,
                status="ok", source="live", error=None,
            )
        ])
        old_manifest.save(tmp_path / gb.MANIFEST_FILENAME)
        (tmp_path / "5000.gabc").write_bytes(VALID_GABC)
        gb.rebuild_manifest(tmp_path, self._make_catalog())
        bak = tmp_path / "manifest.json.bak"
        assert bak.exists()
        loaded = gb.Manifest.from_dict(json.loads(bak.read_text(encoding="utf-8")))
        assert loaded.entries[0].id == 1

    def test_rebuild_resume_after(self, tmp_path: Path):
        """Rebuilt manifest → download skips rebuilt IDs."""
        (tmp_path / "5000.gabc").write_bytes(VALID_GABC)
        manifest, _ = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert 5000 in manifest.ids_with_success()

    def test_rebuild_slug_match(self, tmp_path: Path):
        gabc = b"name: Respice Domine;\n%%\n(c4) Re(f)\n"
        (tmp_path / "in--respice_domine.gabc").write_bytes(gabc)
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.matched == 1
        assert manifest.entries[0].id == 5000

    def test_rebuild_no_match(self, tmp_path: Path):
        gabc = b"name: Something Unknown;\noffice-part: Weird;\n%%\n(c4) X(f)\n"
        (tmp_path / "weird--file.gabc").write_bytes(gabc)
        manifest, stats = gb.rebuild_manifest(tmp_path, self._make_catalog())
        assert stats.no_match == 1
        assert stats.matched == 0

    def test_rebuild_sha256_and_size(self, tmp_path: Path):
        (tmp_path / "5000.gabc").write_bytes(VALID_GABC)
        manifest, _ = gb.rebuild_manifest(tmp_path, self._make_catalog())
        entry = manifest.entries[0]
        assert entry.sha256 == gb.sha256_bytes(VALID_GABC)
        assert entry.size_bytes == len(VALID_GABC)

    def test_rebuild_enriches_metadata_from_catalog(self, tmp_path: Path):
        (tmp_path / "5000.gabc").write_bytes(VALID_GABC)
        manifest, _ = gb.rebuild_manifest(tmp_path, self._make_catalog())
        entry = manifest.entries[0]
        assert entry.office_part == "Introitus"
        assert entry.incipit == "Respice Domine"


# ---------------------------------------------------------------------------
# NABC twin detection and prefetch (#26)
# ---------------------------------------------------------------------------

NABC_GABC = (FIXTURES / "nabc_sample.gabc").read_bytes()


class TestScanNabcIds:
    def test_detects_nabc_files(self, tmp_path: Path):
        (tmp_path / "100.gabc").write_bytes(NABC_GABC)
        (tmp_path / "200.gabc").write_bytes(VALID_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia", incipit="Alleluia",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
            gb.ManifestEntry(
                id=200, elem=None, office_part="Introitus", incipit="Respice Domine",
                filename="200.gabc", sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC), status="ok", source="live", error=None,
            ),
        ])
        nabc_ids = gb.scan_nabc_ids(tmp_path, manifest)
        assert nabc_ids == {100}

    def test_ignores_failed_entries(self, tmp_path: Path):
        (tmp_path / "100.gabc").write_bytes(NABC_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="", incipit="",
                filename="100.gabc", sha256=None,
                size_bytes=None, status="failed", source="live", error="bad",
            ),
        ])
        assert gb.scan_nabc_ids(tmp_path, manifest) == set()


class TestFindPlainTwins:
    def test_finds_matching_twin(self):
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "Cognoverunt"),
            gb.CatalogEntry(200, "Alleluia", "Cognoverunt"),
            gb.CatalogEntry(300, "Introitus", "Other"),
        ]
        nabc_ids = {100}
        twins = gb.find_plain_twins("Cognoverunt", "Alleluia", 100, catalog, nabc_ids)
        assert [t.id for t in twins] == [200]

    def test_excludes_own_id(self):
        catalog = [gb.CatalogEntry(100, "Alleluia", "Cognoverunt")]
        twins = gb.find_plain_twins("Cognoverunt", "Alleluia", 100, catalog, {100})
        assert twins == []

    def test_excludes_other_nabc_ids(self):
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "Cognoverunt"),
            gb.CatalogEntry(200, "Alleluia", "Cognoverunt"),
        ]
        nabc_ids = {100, 200}
        twins = gb.find_plain_twins("Cognoverunt", "Alleluia", 100, catalog, nabc_ids)
        assert twins == []

    def test_case_insensitive(self):
        catalog = [
            gb.CatalogEntry(100, "alleluia", "COGNOVERUNT"),
            gb.CatalogEntry(200, "Alleluia", "cognoverunt"),
        ]
        nabc_ids = {100}
        twins = gb.find_plain_twins("Cognoverunt", "Alleluia", 100, catalog, nabc_ids)
        assert [t.id for t in twins] == [200]

    def test_empty_incipit_returns_no_twins(self):
        catalog = [gb.CatalogEntry(100, "", ""), gb.CatalogEntry(200, "", "")]
        twins = gb.find_plain_twins("", "", 100, catalog, {100})
        assert twins == []

    def test_no_matching_twin(self):
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "Nos qui vivimus"),
            gb.CatalogEntry(200, "Introitus", "Other"),
        ]
        nabc_ids = {100}
        twins = gb.find_plain_twins("Nos qui vivimus", "Alleluia", 100, catalog, nabc_ids)
        assert twins == []


class TestPrefetchPlainTwins:
    def _session_with_responses(self, responses: list[MagicMock]) -> MagicMock:
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get = MagicMock(side_effect=responses)
        return session

    def test_prefetch_downloads_twin(self, tmp_path: Path):
        (tmp_path / "100.gabc").write_bytes(NABC_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia", incipit="Cognoverunt",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(tmp_path / gb.MANIFEST_FILENAME)
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "Cognoverunt"),
            gb.CatalogEntry(200, "Alleluia", "Cognoverunt"),
        ]
        session = self._session_with_responses([
            _mock_response(content=VALID_GABC),
            _mock_response(content=VALID_GABC),
        ])
        limiter = gb.RateLimiter(0)
        nabc_ids = {100}

        stats = gb.prefetch_plain_twins(
            session, tmp_path, manifest, catalog, limiter, nabc_ids,
        )
        assert stats.twins_found == 1
        assert stats.downloaded == 1
        assert stats.already_present == 0
        assert stats.no_twin == 0
        assert (tmp_path / "200.gabc").exists()

    def test_prefetch_skips_already_present(self, tmp_path: Path):
        (tmp_path / "100.gabc").write_bytes(NABC_GABC)
        (tmp_path / "200.gabc").write_bytes(VALID_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia", incipit="Cognoverunt",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
            gb.ManifestEntry(
                id=200, elem=None, office_part="Alleluia", incipit="Cognoverunt",
                filename="200.gabc", sha256=gb.sha256_bytes(VALID_GABC),
                size_bytes=len(VALID_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(tmp_path / gb.MANIFEST_FILENAME)
        catalog = [
            gb.CatalogEntry(100, "Alleluia", "Cognoverunt"),
            gb.CatalogEntry(200, "Alleluia", "Cognoverunt"),
        ]
        session = MagicMock(spec=requests.Session)
        limiter = gb.RateLimiter(0)

        stats = gb.prefetch_plain_twins(
            session, tmp_path, manifest, catalog, limiter, {100},
        )
        assert stats.twins_found == 1
        assert stats.already_present == 1
        assert stats.downloaded == 0

    def test_prefetch_no_twin(self, tmp_path: Path):
        (tmp_path / "100.gabc").write_bytes(NABC_GABC)
        manifest = gb.Manifest(entries=[
            gb.ManifestEntry(
                id=100, elem=None, office_part="Alleluia",
                incipit="Nos qui vivimus",
                filename="100.gabc", sha256=gb.sha256_bytes(NABC_GABC),
                size_bytes=len(NABC_GABC), status="ok", source="live", error=None,
            ),
        ])
        manifest.save(tmp_path / gb.MANIFEST_FILENAME)
        catalog = [gb.CatalogEntry(100, "Alleluia", "Nos qui vivimus")]
        session = MagicMock(spec=requests.Session)
        limiter = gb.RateLimiter(0)

        stats = gb.prefetch_plain_twins(
            session, tmp_path, manifest, catalog, limiter, {100},
        )
        assert stats.no_twin == 1
        assert stats.downloaded == 0
