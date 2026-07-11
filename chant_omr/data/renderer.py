"""Render GABC files into score images using Gregorio + LuaLaTeX.

Pipeline (nomargin tight crop):

    body-only GABC → LuaLaTeX + gregoriotex autocompile → PDF → PNG

Requirements:
    - gregorio (Gregorio 5.2+; CLI renamed from gabc2gregorio)
    - lualatex with ``-shell-escape``
    - pdftoppm (poppler-utils)
    - Libertinus Serif (fontspec / system fonts)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from tqdm import tqdm

from chant_omr.data.gabc_parser import (
    NABC_NOT_SUPPORTED,
    extract_gabc_body,
    is_nabc_notation,
    load_gabc,
    parse_gabc,
)
from chant_omr.data.gregobase import MANIFEST_FILENAME, Manifest, ManifestEntry, disk_filename

logger = logging.getLogger(__name__)

FAILURES_FILENAME = "render_failures.jsonl"
NOMARGIN_TEX_NAME = "nomargin.tex"
DEFAULT_DPI = 300
DEFAULT_WORKERS = 0  # 0 = auto (see resolve_render_workers)
DEFAULT_WORKERS_CAP = 8
TEX_CACHE_DIRNAME = ".texcache"
SCORE_HSIZE = r"10cm"

NOMARGIN_LATEX_TEMPLATE = r"""% !TEX program = LuaLaTeX+se
\documentclass[11pt]{{article}}
\usepackage{{gregoriotex}}
\usepackage{{fontspec}}
\setmainfont{{Libertinus Serif}}
\hoffset-1in
\voffset-1in
\newbox\scorebox
\begin{{document}}
\setbox\scorebox=\vbox{{\hsize={hsize}\relax
  \gregorioscore[a]{{{score_stem}}}
}}
\pagewidth=\wd\scorebox
\pageheight=\dimexpr\ht\scorebox+\dp\scorebox\relax
\shipout\box\scorebox
\end{{document}}
"""


@dataclass
class RenderStats:
    """Summary counters for a render run."""

    manifest_ok: int = 0
    attempted: int = 0
    rendered: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class RenderJob:
    """One manifest entry queued for rendering."""

    entry: ManifestEntry
    gabc_path: Path
    png_path: Path
    gabc_link_path: Path


@dataclass
class RenderFailure:
    """One failed render attempt."""

    id: int
    elem: int | None
    filename: str | None
    error: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "elem": self.elem,
            "filename": self.filename,
            "error": self.error,
            "timestamp": self.timestamp,
        }


def toolchain_available() -> bool:
    """Return True when gregorio, lualatex, and pdftoppm are on PATH."""
    return all(shutil.which(cmd) for cmd in ("gregorio", "lualatex", "pdftoppm"))


def default_render_workers_cap() -> int:
    """Upper bound for auto worker count (override via ``CHANT_OMR_RENDER_WORKERS_MAX``)."""
    raw = os.environ.get("CHANT_OMR_RENDER_WORKERS_MAX", str(DEFAULT_WORKERS_CAP))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_WORKERS_CAP


def resolve_render_workers(workers: int) -> int:
    """Return worker count; ``0`` or negative values mean auto-detect."""
    if workers > 0:
        return workers
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, default_render_workers_cap()))


def tex_cache_dir(output_dir: Path) -> Path:
    """Persistent LuaTeX cache directory for a render output tree."""
    return output_dir / TEX_CACHE_DIRNAME


def png_filename(chant_id: int, elem: int | None) -> str:
    """Output PNG basename matching GregoBase id-based GABC names."""
    return disk_filename(chant_id, elem).removesuffix(".gabc") + ".png"


def work_score_stem(chant_id: int, elem: int | None) -> str:
    """Gregorio autocompile stem; always id-based regardless of on-disk slug."""
    return disk_filename(chant_id, elem).removesuffix(".gabc")


def has_double_header(text: str) -> bool:
    """Return True when the GABC contains more than one ``%%`` marker."""
    return text.count("%%") > 1


def extract_render_body(text: str) -> str:
    """Return neume text after the final ``%%`` marker."""
    return extract_gabc_body(text)


def body_only_gabc_text(text: str, *, name: str) -> str:
    """Strip headers and rebuild a minimal body-only GABC for rendering."""
    score = parse_gabc(text)
    display_name = name.strip() or score.name or "chant"
    body = extract_render_body(text)
    return f"name: {display_name};\n%%\n{body}\n"


def build_nomargin_tex(score_stem: str, *, hsize: str = SCORE_HSIZE) -> str:
    """Return the nomargin LuaLaTeX wrapper for autocompile ``[a]``."""
    return NOMARGIN_LATEX_TEMPLATE.format(score_stem=score_stem, hsize=hsize)


def append_failure_log(path: Path, failure: RenderFailure) -> None:
    """Append one JSON line to the render failure log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(failure.to_dict()) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            handle.write(line)


def link_or_copy_gabc(
    source: Path,
    destination: Path,
    *,
    body_text: str | None = None,
) -> None:
    """Symlink ``source`` into ``rendered/``; copy body-only GABC on cross-device failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        payload = body_text if body_text is not None else source.read_text(encoding="utf-8")
        destination.write_text(payload, encoding="utf-8")


def iter_render_jobs(
    manifest: Manifest,
    gabc_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> Iterator[RenderJob]:
    """Yield manifest ``ok`` entries that still need rendering."""
    for entry in manifest.entries:
        if entry.status != "ok" or not entry.filename:
            continue
        gabc_path = gabc_dir / entry.filename
        png_name = png_filename(entry.id, entry.elem)
        png_path = output_dir / png_name
        gabc_link_path = output_dir / entry.filename
        if png_path.exists() and not force:
            continue
        yield RenderJob(
            entry=entry,
            gabc_path=gabc_path,
            png_path=png_path,
            gabc_link_path=gabc_link_path,
        )


def _tex_cache_for_lualatex(work_dir: Path, cache_dir: Path | None = None) -> Path:
    """Return a writable LuaTeX cache directory for one lualatex invocation."""
    if cache_dir is not None:
        path = cache_dir
    elif os.environ.get("TEXMFCACHE"):
        path = Path(os.environ["TEXMFCACHE"])
    else:
        path = work_dir / ".texcache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_lualatex(
    work_dir: Path,
    tex_name: str = NOMARGIN_TEX_NAME,
    *,
    cache_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cache = _tex_cache_for_lualatex(work_dir, cache_dir)
    env = os.environ.copy()
    env["TEXMFCACHE"] = str(cache)
    return subprocess.run(
        [
            "lualatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-shell-escape",
            tex_name,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env=env,
    )


def _run_pdftoppm(
    work_dir: Path,
    pdf_name: str,
    output_stem: str,
    *,
    dpi: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), pdf_name, output_stem],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def render_gabc_to_image(
    gabc_path: Path,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    work_dir: Path | None = None,
    display_name: str | None = None,
    score_stem: str | None = None,
) -> Path:
    """Render one GABC file to a tight-crop PNG via Gregorio autocompile."""
    if not toolchain_available():
        raise RuntimeError("Gregorio toolchain not available (gregorio, lualatex, pdftoppm)")

    raw_text = gabc_path.read_text(encoding="utf-8")
    if is_nabc_notation(raw_text):
        raise ValueError(NABC_NOT_SUPPORTED)

    score = load_gabc(gabc_path)

    label = display_name or score.name or gabc_path.stem
    body_text = body_only_gabc_text(raw_text, name=label)
    stem = score_stem or gabc_path.stem

    cleanup = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="chant-omr-render-"))

    try:
        gabc_work = work_dir / f"{stem}.gabc"
        tex_path = work_dir / NOMARGIN_TEX_NAME
        pdf_path = work_dir / NOMARGIN_TEX_NAME.replace(".tex", ".pdf")

        gabc_work.write_text(body_text, encoding="utf-8")
        tex_path.write_text(build_nomargin_tex(stem), encoding="utf-8")

        latex_result = _run_lualatex(work_dir)
        if latex_result.returncode != 0 or not pdf_path.exists():
            detail = (latex_result.stderr or latex_result.stdout or "").strip()
            raise RuntimeError(f"lualatex failed: {detail[-2000:]}")

        ppm_result = _run_pdftoppm(work_dir, pdf_path.name, stem, dpi=dpi)
        produced = work_dir / f"{stem}.png"
        if ppm_result.returncode != 0 or not produced.exists():
            detail = (ppm_result.stderr or ppm_result.stdout or "").strip()
            raise RuntimeError(f"pdftoppm failed: {detail[-2000:]}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), output_path)
        return output_path
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def _render_job(
    job: RenderJob,
    *,
    dpi: int,
    force: bool,
    failures_path: Path,
) -> tuple[str, bool, str | None]:
    """Worker entry point: returns (status, changed, error)."""
    if job.png_path.exists() and not force:
        return ("skipped", False, None)

    if not job.gabc_path.exists():
        error = f"missing GABC: {job.gabc_path}"
        append_failure_log(
            failures_path,
            RenderFailure(job.entry.id, job.entry.elem, job.entry.filename, error),
        )
        return ("failed", False, error)

    try:
        raw_text = job.gabc_path.read_text(encoding="utf-8")
        if is_nabc_notation(raw_text):
            raise ValueError(NABC_NOT_SUPPORTED)

        body_text = body_only_gabc_text(raw_text, name=job.entry.incipit or job.gabc_path.stem)
        stem = work_score_stem(job.entry.id, job.entry.elem)
        render_gabc_to_image(
            job.gabc_path,
            job.png_path,
            dpi=dpi,
            display_name=job.entry.incipit or None,
            score_stem=stem,
        )
        link_or_copy_gabc(job.gabc_path, job.gabc_link_path, body_text=body_text)
        return ("rendered", True, None)
    except Exception as exc:  # noqa: BLE001 — log and continue batch
        error = str(exc)
        append_failure_log(
            failures_path,
            RenderFailure(job.entry.id, job.entry.elem, job.entry.filename, error),
        )
        return ("failed", False, error)


def _init_render_worker(cache_dir: str) -> None:
    """ProcessPoolExecutor initializer: share LuaTeX font cache across jobs."""
    os.environ["TEXMFCACHE"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


def render_corpus(
    gabc_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    dpi: int = DEFAULT_DPI,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    show_progress: bool = True,
) -> RenderStats:
    """Render manifest ``ok`` GABC entries into ``output_dir``."""
    if not toolchain_available():
        raise RuntimeError("Gregorio toolchain not available (gregorio, lualatex, pdftoppm)")

    worker_count = resolve_render_workers(workers)
    manifest_path = gabc_dir / MANIFEST_FILENAME
    manifest = Manifest.load(manifest_path)
    failures_path = output_dir / FAILURES_FILENAME
    cache_dir = tex_cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_render_jobs(manifest, gabc_dir, output_dir, force=force))
    if limit is not None:
        jobs = jobs[:limit]

    stats = RenderStats(
        manifest_ok=sum(1 for e in manifest.entries if e.status == "ok" and e.filename),
        attempted=len(jobs),
    )

    if not jobs:
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(jobs, disable=not show_progress, unit="score")

    if worker_count <= 1:
        _init_render_worker(str(cache_dir))
        for job in progress:
            status, _, _ = _render_job(job, dpi=dpi, force=force, failures_path=failures_path)
            if status == "rendered":
                stats.rendered += 1
            elif status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
        return stats

    render_fn = partial(_render_job, dpi=dpi, force=force, failures_path=failures_path)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_render_worker,
        initargs=(str(cache_dir),),
    ) as executor:
        futures = [executor.submit(render_fn, job) for job in jobs]
        for future in as_completed(futures):
            progress.update(1)
            status, _, _ = future.result()
            if status == "rendered":
                stats.rendered += 1
            elif status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
    progress.close()
    return stats


def render_batch(
    gabc_dir: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    workers: int = DEFAULT_WORKERS,
) -> list[Path]:
    """Backward-compatible wrapper returning rendered PNG paths."""
    stats = render_corpus(gabc_dir, output_dir, dpi=dpi, workers=workers, show_progress=False)
    if stats.failed:
        logger.warning("render_batch finished with %s failures", stats.failed)
    return sorted(output_dir.glob("*.png"))
