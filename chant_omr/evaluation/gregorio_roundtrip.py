"""Gregorio round-trip equivalence checker and compilation validator (#46, #47).

Compiles GABC snippets with ``gregorio -s -S`` (stdin→stdout) and compares
the canonical ``.gtex`` output.  Two GABC strings are considered equivalent
if and only if Gregorio produces the same note-level TeX commands for both.

Also provides :func:`check_gregorio_compilation` for structural validity
checking (#46): returns whether Gregorio can compile the GABC without
errors on stdout or stderr.

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
from dataclasses import dataclass, field
from functools import lru_cache

log = logging.getLogger(__name__)

_GREGORIO_BIN: str | None = shutil.which("gregorio")

# Transient metadata stripped before comparison:
_SCORE_HASH_RE = re.compile(r"\\GreBeginScore\{[0-9a-f]+\}")
_API_VERSION_RE = re.compile(r"\\GregorioTeXAPIVersion\{[^}]+\}")


@dataclass(frozen=True)
class GregorioResult:
    """Raw result from a single ``gregorio`` invocation."""

    gtex: str | None
    stderr: str
    returncode: int


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
def _run_gregorio(gabc_body: str) -> GregorioResult:
    """Run ``gregorio -s -S`` on a GABC body and return the full result.

    Wraps the body in a minimal GABC document, compiles via stdin/stdout
    (avoids kpathsea write restrictions), and returns stdout (``.gtex``),
    stderr (warnings/errors), and the exit code.  Results are cached.
    """
    if not _GREGORIO_BIN:
        return GregorioResult(gtex=None, stderr="gregorio not installed", returncode=-1)

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
    except subprocess.TimeoutExpired:
        return GregorioResult(gtex=None, stderr="gregorio timed out", returncode=-1)
    except OSError as exc:
        return GregorioResult(gtex=None, stderr=str(exc), returncode=-1)

    gtex = _canonicalize_gtex(result.stdout) if result.stdout else None
    return GregorioResult(gtex=gtex, stderr=result.stderr.strip(), returncode=result.returncode)


def _compile_gabc_body(gabc_body: str) -> str | None:
    """Compile a GABC body to ``.gtex`` via ``gregorio -S`` (stdout).

    Returns the canonical ``.gtex`` content with metadata stripped, or None
    on compilation failure.  Thin wrapper around :func:`_run_gregorio`.
    """
    return _run_gregorio(gabc_body).gtex


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


# ---------------------------------------------------------------------------
# Gregorio compilation check (#46)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GregorioCompilationResult:
    """Result of compiling a GABC body with ``gregorio`` for validity."""

    compiles: bool
    errors: list[str] = field(default_factory=list)


def check_gregorio_compilation(gabc_body: str) -> GregorioCompilationResult:
    """Check whether ``gregorio`` can compile a GABC body without errors (#46).

    A body is considered valid only if:
    - ``gregorio`` produces ``.gtex`` output on stdout
    - The exit code is 0
    - There is nothing on stderr (no warnings or errors)

    Returns a :class:`GregorioCompilationResult` with the compilation
    verdict and any error/warning messages from gregorio.

    When ``gregorio`` is not installed, returns a failing result with an
    explanatory error message.
    """
    _log_version_once()
    result = _run_gregorio(gabc_body.strip())

    errors: list[str] = []

    if result.gtex is None:
        errors.append("gregorio produced no output")
    if result.returncode != 0:
        errors.append(f"gregorio exit code {result.returncode}")
    if result.stderr:
        for line in result.stderr.splitlines():
            line = line.strip()
            if line:
                errors.append(line)

    return GregorioCompilationResult(
        compiles=len(errors) == 0,
        errors=errors,
    )
