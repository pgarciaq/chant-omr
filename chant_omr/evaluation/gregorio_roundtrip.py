"""Gregorio round-trip equivalence checker (#47, Option D).

Compiles GABC snippets with ``gregorio -s -S`` (stdin→stdout) and compares
the canonical ``.gtex`` output.  Two GABC strings are considered equivalent
if and only if Gregorio produces the same note-level TeX commands for both.

Falls back gracefully when ``gregorio`` is not installed — all comparisons
return *not equivalent*, and callers fall back to the string-level table.

.. note:: Dependency

   ``gregorio`` ships inside ``texlive-binaries`` on Ubuntu/Debian.
   The version in Ubuntu Jammy (TeX Live 2021) is much older than
   Fedora 44's Gregorio 6.1.0 (TeX Live 2025).  The ``.gtex`` glyph
   names and command structure can differ across major versions, so
   round-trip results are only meaningful when *both* pred and ref are
   compiled by the **same** ``gregorio`` binary (which they always are
   in a single evaluation run).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from functools import lru_cache

log = logging.getLogger(__name__)

_GREGORIO_BIN: str | None = shutil.which("gregorio")

# Transient metadata stripped before comparison:
_SCORE_HASH_RE = re.compile(r"\\GreBeginScore\{[0-9a-f]+\}")
_API_VERSION_RE = re.compile(r"\\GregorioTeXAPIVersion\{[^}]+\}")


def gregorio_available() -> bool:
    """Return True if the ``gregorio`` binary is on PATH."""
    return _GREGORIO_BIN is not None


def gregorio_version() -> str | None:
    """Return the ``gregorio --version`` string, or None if unavailable."""
    if not _GREGORIO_BIN:
        return None
    try:
        result = subprocess.run(
            [_GREGORIO_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip().splitlines()[0] if result.stdout else None
    except (subprocess.TimeoutExpired, OSError):
        return None


@lru_cache(maxsize=4096)
def _compile_gabc_body(gabc_body: str) -> str | None:
    """Compile a GABC body to ``.gtex`` via ``gregorio -S`` (stdout).

    Uses ``-S`` to write to stdout (avoids kpathsea write restrictions).
    Returns the canonical ``.gtex`` content with metadata stripped, or None
    on compilation failure.  Results are cached by input string.
    """
    if not _GREGORIO_BIN:
        return None

    gabc_doc = f"name:__norm__;\n%%\n{gabc_body}\n"

    try:
        result = subprocess.run(
            [_GREGORIO_BIN, "-s", "-S"],
            input=gabc_doc,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if not result.stdout:
        return None

    return _canonicalize_gtex(result.stdout)


def _canonicalize_gtex(raw: str) -> str:
    """Strip transient metadata from ``.gtex`` so content can be compared.

    Removes:
        - Comment lines (``%``)
        - The score hash (changes with source text, not with output)
        - The API version string (changes with gregorio version)
    """
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or not stripped:
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = _SCORE_HASH_RE.sub(r"\\GreBeginScore{}", text)
    text = _API_VERSION_RE.sub(r"\\GregorioTeXAPIVersion{}", text)
    return text


_version_logged = False


def _log_version_once() -> None:
    global _version_logged  # noqa: PLW0603
    if not _version_logged:
        _version_logged = True
        ver = gregorio_version()
        if ver:
            log.info("Gregorio round-trip using: %s", ver)
        else:
            log.warning("Gregorio not found — round-trip normalization disabled")


def gregorio_groups_equivalent(group_a: str, group_b: str) -> bool:
    """Check whether two neume groups produce identical Gregorio output.

    Wraps each group in a minimal GABC document with a ``(c4)`` clef,
    compiles with ``gregorio``, and compares the canonical ``.gtex``.

    Returns False if ``gregorio`` is unavailable or either snippet fails to
    compile — this makes the fallback conservative (non-equivalent when in
    doubt).
    """
    _log_version_once()
    body_a = f"(c4) x{group_a}"
    body_b = f"(c4) x{group_b}"

    gtex_a = _compile_gabc_body(body_a)
    gtex_b = _compile_gabc_body(body_b)

    if gtex_a is None or gtex_b is None:
        return False

    return gtex_a == gtex_b


def gregorio_bodies_equivalent(body_a: str, body_b: str) -> bool:
    """Check whether two full GABC bodies produce identical Gregorio output."""
    gtex_a = _compile_gabc_body(body_a.strip())
    gtex_b = _compile_gabc_body(body_b.strip())

    if gtex_a is None or gtex_b is None:
        return False

    return gtex_a == gtex_b
