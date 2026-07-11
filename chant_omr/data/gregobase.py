"""Download and manage GABC files from GregoBase.

GregoBase (https://gregobase.selapa.net/) hosts ~20k Gregorian chant
transcriptions in GABC format. This module fetches the official catalog
(``csv.php``), downloads GABC via ``download.php`` with ``elem`` variant
handling, and tracks state in a local manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from requests import Response

logger = logging.getLogger(__name__)

GREGOBASE_BASE = "https://gregobase.selapa.net"
CATALOG_URL = f"{GREGOBASE_BASE}/csv.php"
UPDATES_URL = f"{GREGOBASE_BASE}/updates.php"
DOWNLOAD_URL = f"{GREGOBASE_BASE}/download.php"
USER_AGENT = "chant-omr/0.1 (+https://github.com/pgarciaq/chant-omr)"
MAX_ELEM = 20
DEFAULT_RATE_LIMIT = 1.0
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
MANIFEST_FILENAME = "manifest.json"

CHANT_ID_RE = re.compile(r"chant\.php\?id=(\d+)", re.IGNORECASE)
CATALOG_DATE_RE = re.compile(r"gregobase_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})")
CONTENT_DISPOSITION_FILENAME_RE = re.compile(
    r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
    re.IGNORECASE,
)
GENERIC_FILENAME_RE = re.compile(r"^-+\.gabc$", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogEntry:
    """One row from the GregoBase catalog CSV."""

    id: int
    office_part: str
    incipit: str


@dataclass
class ManifestEntry:
    """Download state for one GABC variant."""

    id: int
    elem: int | None
    office_part: str
    incipit: str
    filename: str | None
    sha256: str | None
    size_bytes: int | None
    status: str  # ok | failed | skipped
    source: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        return cls(
            id=int(data["id"]),
            elem=data["elem"],
            office_part=data.get("office_part", ""),
            incipit=data.get("incipit", ""),
            filename=data.get("filename"),
            sha256=data.get("sha256"),
            size_bytes=data.get("size_bytes"),
            status=data["status"],
            source=data.get("source", "live"),
            error=data.get("error"),
        )


@dataclass
class Manifest:
    """Local download state persisted as JSON."""

    catalog_date: str | None = None
    last_sync_date: str | None = None
    entries: list[ManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_date": self.catalog_date,
            "last_sync_date": self.last_sync_date,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            catalog_date=data.get("catalog_date"),
            last_sync_date=data.get("last_sync_date"),
            entries=[ManifestEntry.from_dict(e) for e in data.get("entries", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def remove_entries_for_id(self, chant_id: int) -> None:
        self.entries = [e for e in self.entries if e.id != chant_id]

    def entries_for_id(self, chant_id: int) -> list[ManifestEntry]:
        return [e for e in self.entries if e.id == chant_id]

    def ids_with_success(self) -> set[int]:
        return {e.id for e in self.entries if e.status == "ok"}

    def find_ok_entry(
        self, chant_id: int, elem: int | None, sha256: str
    ) -> ManifestEntry | None:
        for entry in self.entries:
            if (
                entry.id == chant_id
                and entry.elem == elem
                and entry.status == "ok"
                and entry.sha256 == sha256
            ):
                return entry
        return None


@dataclass
class DownloadStats:
    """Summary returned by :func:`download_corpus`."""

    catalog_count: int
    attempted_ids: int
    downloaded_files: int
    skipped_files: int
    failed_ids: int
    paths: list[Path]


class RateLimiter:
    """Enforce a minimum delay between consecutive download requests."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last_at: float | None = None

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        if self._last_at is not None:
            elapsed = now - self._last_at
            remaining = self.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_at = time.monotonic()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_valid_gabc(body: bytes) -> bool:
    return bool(body) and b"%%" in body


def parse_catalog_date(content_disposition: str | None) -> str | None:
    """Parse ``gregobase_YYYY-MM-DD_HH-MM.csv`` into ISO datetime string."""
    if not content_disposition:
        return None
    match = CATALOG_DATE_RE.search(content_disposition)
    if not match:
        return None
    date_part, time_part = match.groups()
    hour, minute = time_part.split("-")
    return f"{date_part}T{hour}:{minute}:00"


def parse_catalog_csv(text: str) -> list[CatalogEntry]:
    """Parse GregoBase catalog CSV (office-part, incipit, id)."""
    reader = csv.reader(io.StringIO(text))
    entries: list[CatalogEntry] = []
    for row in reader:
        if len(row) < 3:
            continue
        office_part, incipit, id_str = row[0], row[1], row[2]
        if not id_str.strip().isdigit():
            continue
        entries.append(
            CatalogEntry(
                id=int(id_str),
                office_part=office_part,
                incipit=incipit,
            )
        )
    return entries


def parse_content_disposition_filename(headers: dict[str, str]) -> str | None:
    """Extract filename from ``Content-Disposition`` header."""
    raw = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not raw:
        return None
    match = CONTENT_DISPOSITION_FILENAME_RE.search(raw)
    if not match:
        return None
    return match.group(1).strip()


def fallback_filename(chant_id: int, elem: int | None) -> str:
    if elem is None:
        return f"{chant_id}.gabc"
    return f"{chant_id}_elem{elem}.gabc"


def is_generic_filename(filename: str) -> bool:
    """Return True for GregoBase placeholder names like ``----.gabc``."""
    return bool(GENERIC_FILENAME_RE.match(filename))


def resolve_filename(
    headers: dict[str, str],
    chant_id: int,
    elem: int | None,
    output_dir: Path,
    digest: str,
) -> str:
    """Choose a unique on-disk filename for one downloaded variant."""
    candidate = parse_content_disposition_filename(headers)
    if candidate and not is_generic_filename(candidate):
        dest = output_dir / candidate
        if not dest.exists() or sha256_bytes(dest.read_bytes()) == digest:
            return candidate
    return fallback_filename(chant_id, elem)


def parse_updates_html(html: str) -> list[int]:
    """Extract unique chant IDs from updates.php HTML."""
    seen: set[int] = set()
    ordered: list[int] = []
    for match in CHANT_ID_RE.finditer(html):
        chant_id = int(match.group(1))
        if chant_id not in seen:
            seen.add(chant_id)
            ordered.append(chant_id)
    return ordered


def _response_headers_dict(response: Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items()}


def fetch_catalog(session: requests.Session) -> tuple[list[CatalogEntry], str | None]:
    response = _request_with_retries(session, CATALOG_URL)
    catalog_date = parse_catalog_date(
        response.headers.get("Content-Disposition")
        or response.headers.get("content-disposition")
    )
    response.encoding = response.encoding or "utf-8"
    return parse_catalog_csv(response.text), catalog_date


def fetch_updates(session: requests.Session, days: int | None = None) -> list[int]:
    url = UPDATES_URL
    if days is not None:
        url = f"{url}?{urlencode({'days': days})}"
    response = _request_with_retries(session, url)
    response.encoding = response.encoding or "utf-8"
    return parse_updates_html(response.text)


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
) -> Response:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if response.status_code in {429, 503}:
                delay = 2**attempt
                logger.warning(
                    "HTTP %s for %s — retry %s/%s in %ss",
                    response.status_code,
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            delay = 2**attempt
            logger.warning(
                "Request failed for %s — retry %s/%s in %ss: %s",
                url,
                attempt + 1,
                MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _download_url_params(chant_id: int, elem: int | None) -> dict[str, str | int]:
    params: dict[str, str | int] = {"id": chant_id, "format": "gabc"}
    if elem is not None:
        params["elem"] = elem
    return params


def fetch_gabc_variant(
    session: requests.Session,
    chant_id: int,
    elem: int | None,
    *,
    rate_limiter: RateLimiter | None = None,
) -> tuple[bytes, dict[str, str], int | None]:
    """Fetch one GABC variant. Returns body, headers, HTTP status."""
    if rate_limiter is not None:
        rate_limiter.wait()
    try:
        response = _request_with_retries(
            session,
            DOWNLOAD_URL,
            params=_download_url_params(chant_id, elem),
        )
        return response.content, _response_headers_dict(response), response.status_code
    except requests.RequestException as exc:
        logger.error("Failed to download id=%s elem=%s: %s", chant_id, elem, exc)
        return b"", {}, None


def download_variants_for_id(
    session: requests.Session,
    catalog_entry: CatalogEntry,
    output_dir: Path,
    manifest: Manifest,
    *,
    rate_limiter: RateLimiter,
) -> tuple[list[Path], list[ManifestEntry], int, int]:
    """Download all unique GABC variants for one catalog ID.

    Returns saved paths, manifest entries, skipped count, and new download count.
    """
    saved_paths: list[Path] = []
    new_entries: list[ManifestEntry] = []
    skipped = 0
    new_downloads = 0
    seen_hashes: set[str] = set()

    for elem in [None, *range(1, MAX_ELEM + 1)]:
        body, headers, status_code = fetch_gabc_variant(
            session,
            catalog_entry.id,
            elem,
            rate_limiter=rate_limiter,
        )

        if not is_valid_gabc(body):
            if elem is not None and not body:
                break
            continue

        digest = sha256_bytes(body)
        filename = resolve_filename(headers, catalog_entry.id, elem, output_dir, digest)

        existing = manifest.find_ok_entry(catalog_entry.id, elem, digest)
        if existing and existing.filename:
            existing_path = output_dir / existing.filename
            if existing_path.exists() and existing_path.read_bytes() == body:
                logger.debug("Skipping existing %s", existing.filename)
                skipped += 1
                new_entries.append(existing)
                saved_paths.append(existing_path)
                seen_hashes.add(digest)
                break

        if digest in seen_hashes:
            break
        seen_hashes.add(digest)

        dest = output_dir / filename
        dest.write_bytes(body)
        new_downloads += 1
        entry = ManifestEntry(
            id=catalog_entry.id,
            elem=elem,
            office_part=catalog_entry.office_part,
            incipit=catalog_entry.incipit,
            filename=filename,
            sha256=digest,
            size_bytes=len(body),
            status="ok",
            source="live",
            error=None,
        )
        new_entries.append(entry)
        saved_paths.append(dest)
        logger.info("Saved %s (%s bytes)", filename, len(body))

    if not any(e.status == "ok" for e in new_entries):
        new_entries = [
            ManifestEntry(
                id=catalog_entry.id,
                elem=None,
                office_part=catalog_entry.office_part,
                incipit=catalog_entry.incipit,
                filename=None,
                sha256=None,
                size_bytes=None,
                status="failed",
                source="live",
                error="no valid gabc",
            )
        ]
        logger.warning(
            "No valid GABC for id=%s (%s)", catalog_entry.id, catalog_entry.incipit
        )

    return saved_paths, new_entries, skipped, new_downloads


def _catalog_index(catalog: list[CatalogEntry]) -> dict[int, CatalogEntry]:
    return {entry.id: entry for entry in catalog}


def _select_catalog_ids(
    catalog: list[CatalogEntry],
    manifest: Manifest,
    *,
    limit: int | None,
    sync_ids: list[int] | None,
) -> list[CatalogEntry]:
    """Return catalog rows to process this run.

    Normal mode: catalog IDs without any successful variant (``limit`` caps batch).
    ``--sync`` adds update IDs for forced refresh (not subject to ``limit``).
    """
    index = _catalog_index(catalog)
    success_ids = manifest.ids_with_success()
    pending = [entry for entry in catalog if entry.id not in success_ids]
    if limit is not None:
        pending = pending[:limit]

    if not sync_ids:
        return pending

    sync_entries: list[CatalogEntry] = []
    sync_id_set: set[int] = set()
    for chant_id in sync_ids:
        sync_id_set.add(chant_id)
        sync_entries.append(
            index.get(chant_id) or CatalogEntry(id=chant_id, office_part="", incipit="")
        )

    pending = [entry for entry in pending if entry.id not in sync_id_set]
    return sync_entries + pending


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_corpus(
    output_dir: Path,
    *,
    limit: int | None = None,
    sync: bool = False,
    sync_days: int | None = None,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> DownloadStats:
    """Download GABC corpus from GregoBase into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = Manifest.load(manifest_path)

    session = make_session()
    catalog, catalog_date = fetch_catalog(session)
    if catalog_date:
        manifest.catalog_date = catalog_date

    sync_ids: list[int] | None = None
    if sync:
        sync_ids = fetch_updates(session, sync_days)
        manifest.last_sync_date = datetime.now(UTC).replace(microsecond=0).isoformat()
        logger.info("updates.php: %s unique IDs to refresh", len(sync_ids))

    to_process = _select_catalog_ids(catalog, manifest, limit=limit, sync_ids=sync_ids)
    rate_limiter = RateLimiter(rate_limit)

    downloaded_files = 0
    skipped_files = 0
    failed_ids = 0
    paths: list[Path] = []

    for catalog_entry in to_process:
        manifest.remove_entries_for_id(catalog_entry.id)
        saved, entries, skipped, new_count = download_variants_for_id(
            session,
            catalog_entry,
            output_dir,
            manifest,
            rate_limiter=rate_limiter,
        )
        manifest.entries.extend(entries)
        skipped_files += skipped
        downloaded_files += new_count

        if any(e.status == "ok" for e in entries):
            paths.extend(saved)
        else:
            failed_ids += 1

        manifest.save(manifest_path)

    return DownloadStats(
        catalog_count=len(catalog),
        attempted_ids=len(to_process),
        downloaded_files=downloaded_files,
        skipped_files=skipped_files,
        failed_ids=failed_ids,
        paths=paths,
    )
