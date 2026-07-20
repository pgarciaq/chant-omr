"""NABC (Neumes Above/Below Chant) utilities.

Provides stripping of NABC pipe annotations from GABC text and batch
collapsing of an NABC corpus into plain GABC for training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chant_omr.data.gabc_parser import NABC_NEUME_RE, is_nabc_notation

if TYPE_CHECKING:
    from chant_omr.data.gregobase import CatalogEntry, Manifest

logger = logging.getLogger(__name__)

# Matches a neume group: opening '(' ... closing ')'
_NEUME_GROUP_RE = re.compile(r"\([^()]*\)")


def strip_nabc_to_plain(text: str) -> str:
    """Strip NABC pipe annotations from full GABC text.

    Algorithm: for each neume group in the body, split content by ``|``,
    keep even-indexed segments (0, 2, 4, ...) which are the standard
    neumes, discard odd-indexed segments (NABC annotations).  Headers
    are preserved, and ``nabc-lines`` header is removed if present.
    """
    parts = text.split("%%", maxsplit=1)
    if len(parts) < 2:
        return text

    header_text, body = parts

    header_lines: list[str] = []
    for line in header_text.splitlines(keepends=True):
        stripped = line.strip().rstrip(";").strip()
        if stripped.lower().startswith("nabc-lines"):
            continue
        header_lines.append(line)
    clean_header = "".join(header_lines)

    def _strip_group(match: re.Match[str]) -> str:
        content = match.group(0)[1:-1]  # strip parens
        if "|" not in content:
            return match.group(0)
        segments = content.split("|")
        plain = "".join(segments[i] for i in range(0, len(segments), 2))
        return f"({plain})"

    clean_body = _NEUME_GROUP_RE.sub(_strip_group, body)
    return f"{clean_header}%%{clean_body}"


@dataclass
class CollapseStats:
    """Summary returned by :func:`collapse_nabc_corpus`."""

    collapsed: int = 0
    skipped_has_twin: int = 0
    skipped_other: int = 0


def collapse_nabc_corpus(
    gabc_dir: Path,
    output_dir: Path,
    manifest: "Manifest",
    catalog: list["CatalogEntry"],
    *,
    only_if_plain_missing: bool = True,
) -> CollapseStats:
    """Collapse NABC files to plain GABC.

    For each NABC entry in the manifest, optionally check whether a plain
    twin already exists (``only_if_plain_missing``).  If not, strip the
    NABC annotations and write the result to ``output_dir/{id}.gabc``.
    """
    from chant_omr.data.gregobase import find_plain_twins, scan_nabc_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = CollapseStats()

    nabc_ids = scan_nabc_ids(gabc_dir, manifest)
    downloaded_ids = manifest.ids_with_success()

    nabc_entries = {
        e.id: e
        for e in manifest.entries
        if e.id in nabc_ids and e.status == "ok" and e.filename
    }

    for nabc_id in sorted(nabc_ids):
        entry = nabc_entries.get(nabc_id)
        if entry is None:
            stats.skipped_other += 1
            continue

        if only_if_plain_missing:
            twins = find_plain_twins(
                entry.incipit, entry.office_part, nabc_id, catalog, nabc_ids,
            )
            if any(t.id in downloaded_ids for t in twins):
                stats.skipped_has_twin += 1
                continue

        fpath = gabc_dir / entry.filename
        if not fpath.exists():
            stats.skipped_other += 1
            continue

        text = fpath.read_text(encoding="utf-8")
        plain = strip_nabc_to_plain(text)
        dest = output_dir / f"{nabc_id}.gabc"
        dest.write_text(plain, encoding="utf-8")
        stats.collapsed += 1
        logger.info("Collapsed NABC id=%d -> %s", nabc_id, dest.name)

    return stats


def infer_nabc_lines(text: str) -> int:
    """Infer the ``nabc-lines`` count from GABC body pipe depth.

    For each neume group with pipes, count the NABC annotation segments
    (odd-indexed after split by ``|``).  The number of NABC lines equals
    the maximum annotation count per neume group.  Falls back to 1 for
    typical GregoBase files.
    """
    parts = text.split("%%", maxsplit=1)
    body = parts[1] if len(parts) == 2 else text

    max_depth = 0
    for match in NABC_NEUME_RE.finditer(body):
        content = match.group(0)[1:-1]
        segments = content.split("|")
        n_annotations = len(segments) // 2
        if n_annotations > max_depth:
            max_depth = n_annotations

    return max(max_depth, 1)


def inject_nabc_header(text: str, n_lines: int | None = None) -> str:
    """Inject ``nabc-lines`` header into GABC text if not already present.

    If ``n_lines`` is ``None``, infer from body via :func:`infer_nabc_lines`.
    """
    if re.search(r"(?i)nabc-lines\s*:", text.split("%%", maxsplit=1)[0] if "%%" in text else ""):
        return text

    if n_lines is None:
        n_lines = infer_nabc_lines(text)

    parts = text.split("%%", maxsplit=1)
    if len(parts) < 2:
        return text

    header, body = parts
    if header.rstrip().endswith("\n") or header.rstrip() == "":
        injected_header = f"{header.rstrip()}\nnabc-lines: {n_lines};\n"
    else:
        injected_header = f"{header}\nnabc-lines: {n_lines};\n"

    return f"{injected_header}%%{body}"
