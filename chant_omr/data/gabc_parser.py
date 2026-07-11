"""Parse GABC notation files into structured representations.

GABC (Gregorio ABC) is the input format for the Gregorio TeX package.
Example:
    name: Kyrie XVII;
    %%
    (c4) Ky(f)ri(gf)e(h) *() e(ixhi)lé(h)i(g)son.(f)

The parser extracts:
- Header fields (name, mode, annotation, etc.)
- Body: interleaved (neume) and text tokens
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

NABC_NEUME_RE = re.compile(r"\([^()]*\|[^()]*\)")
NABC_NOT_SUPPORTED = "NABC notation not supported in v0"


@dataclass
class GABCScore:
    """A parsed GABC file."""

    path: Path | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def name(self) -> str:
        return self.headers.get("name", "")


def parse_gabc(text: str) -> GABCScore:
    """Parse a GABC string into a GABCScore."""
    parts = text.split("%%", maxsplit=1)
    headers = {}
    body = ""

    if len(parts) == 2:
        header_text, body = parts
        for line in header_text.strip().splitlines():
            line = line.strip().rstrip(";")
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip()] = value.strip()
        body = body.strip()
    else:
        body = text.strip()

    return GABCScore(headers=headers, body=body)


def load_gabc(path: Path) -> GABCScore:
    """Load and parse a GABC file from disk."""
    text = path.read_text(encoding="utf-8")
    score = parse_gabc(text)
    score.path = path
    return score


def extract_gabc_body(text: str) -> str:
    """Return neume text after the final ``%%`` marker."""
    parts = text.split("%%")
    if len(parts) < 2:
        body = text.strip()
    else:
        body = parts[-1].strip()
    if not body:
        raise ValueError("empty GABC body")
    return body


def gabc_reject_reason(body: bytes) -> str | None:
    """Return why ``body`` is not a usable GABC file, or ``None`` if valid."""
    if not body:
        return "empty body"
    text = body.decode("utf-8", errors="replace")
    if "%%" not in text:
        return "missing %%"
    try:
        extract_gabc_body(text)
    except ValueError:
        return "empty gabc body"
    return None


def is_nabc_notation(text: str) -> bool:
    """Return True when the score uses NABC pipe annotations."""
    if "nabc-lines" in text.lower():
        return True
    try:
        body = extract_gabc_body(text)
    except ValueError:
        return False
    return bool(NABC_NEUME_RE.search(body))


DEFAULT_MIN_BODY_LEN = 20


def plain_gabc_reject_reason(raw: bytes, *, min_body_len: int = DEFAULT_MIN_BODY_LEN) -> str | None:
    """Return why ``raw`` is not plain trainable GABC, or ``None`` if usable."""
    reason = gabc_reject_reason(raw)
    if reason:
        return reason
    text = raw.decode("utf-8", errors="replace")
    if is_nabc_notation(text):
        return "nabc notation"
    try:
        body = extract_gabc_body(text)
    except ValueError:
        return "empty gabc body"
    if len(body) < min_body_len:
        return "body too short"
    return None


def iter_plain_gabc_bodies(
    gabc_dir: Path,
    manifest: object | None = None,
    *,
    min_body_len: int = DEFAULT_MIN_BODY_LEN,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, body)`` for plain trainable GABC files.

    When *manifest* is provided (a :class:`~chant_omr.data.gregobase.Manifest`),
    only ``status: ok`` entries with an on-disk ``filename`` are considered.
    Otherwise every ``*.gabc`` file directly under *gabc_dir* is scanned.
    """
    gabc_dir = Path(gabc_dir)

    if manifest is not None:
        for entry in manifest.entries:
            if entry.status != "ok" or not entry.filename:
                continue
            path = gabc_dir / entry.filename
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if plain_gabc_reject_reason(raw, min_body_len=min_body_len):
                continue
            yield path, extract_gabc_body(raw.decode("utf-8"))
        return

    for path in sorted(gabc_dir.glob("*.gabc")):
        raw = path.read_bytes()
        if plain_gabc_reject_reason(raw, min_body_len=min_body_len):
            continue
        yield path, extract_gabc_body(raw.decode("utf-8"))
