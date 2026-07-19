"""Benchmark evaluation harness for chant OMR (#14).

Discovers (image, GABC) pairs from benchmark or rendered directories,
runs the model, and computes GED / neume accuracy / structural validity.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from chant_omr.data.dataset import catalog_id_from_render_stem, is_test_split
from chant_omr.data.gabc_parser import extract_gabc_body
from chant_omr.evaluation.metrics import (
    GEDResult,
    NeumeAccuracyResult,
    StructuralValidityResult,
    check_structural_validity,
    gabc_edit_distance,
    neume_accuracy,
)


@dataclass
class SampleResult:
    """Evaluation result for a single (image, reference) pair."""

    image_path: Path
    ref_path: Path
    pred_body: str
    ref_body: str
    ged: GEDResult
    neume_acc: NeumeAccuracyResult
    validity: StructuralValidityResult
    elapsed_s: float


@dataclass
class EvalReport:
    """Aggregate evaluation report across all benchmark pairs."""

    samples: list[SampleResult] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ged(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.ged.normalized for s in self.samples) / len(self.samples)

    @property
    def mean_neume_accuracy(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.neume_acc.accuracy for s in self.samples) / len(self.samples)

    @property
    def structural_validity_rate(self) -> float:
        if not self.samples:
            return 0.0
        valid = sum(1 for s in self.samples if s.validity.is_valid)
        return valid / len(self.samples)

    @property
    def total_elapsed_s(self) -> float:
        return sum(s.elapsed_s for s in self.samples)


def discover_benchmark_pairs(
    benchmark_dir: Path,
    *,
    test_split_only: bool = False,
) -> list[tuple[Path, Path]]:
    """Find (image, gabc) pairs in ``benchmarks/{book}/page_NNN.{png,gabc}`` layout.

    Also supports flat directories (rendered corpus style) where
    ``*.png`` and ``*.gabc`` share the same stem.

    When *test_split_only* is True, only pairs whose stem parses as a
    GregoBase catalog ID in the test split (``catalog_id % 20 == 0``)
    are returned.  Non-numeric stems (e.g. ``page_001``) are skipped
    when this filter is active.
    """
    benchmark_dir = Path(benchmark_dir)
    pairs: list[tuple[Path, Path]] = []

    for png_path in sorted(benchmark_dir.rglob("*.png")):
        gabc_path = png_path.with_suffix(".gabc")
        if not gabc_path.is_file():
            continue
        if test_split_only:
            try:
                cat_id = catalog_id_from_render_stem(png_path.stem)
            except ValueError:
                continue
            if not is_test_split(cat_id):
                continue
        pairs.append((png_path, gabc_path))

    return pairs


def _read_ref_body(gabc_path: Path) -> str:
    """Read and extract the GABC body from a reference file."""
    text = gabc_path.read_text(encoding="utf-8")
    return extract_gabc_body(text)


def evaluate_checkpoint(
    checkpoint_path: Path,
    benchmark_dir: Path,
    *,
    config_path: Path | None = None,
    device: str = "cpu",
    beam_width: int = 3,
    max_length: int = 2048,
    repetition_penalty: float = 1.1,
    grammar_constrained: bool = False,
    limit: int | None = None,
    test_split_only: bool = False,
    progress_callback: "Callable[[int, int, Path, float], None] | None" = None,
) -> EvalReport:
    """Run evaluation on all benchmark pairs and return an ``EvalReport``.

    Loads the model once, then runs inference on each pair.  When
    *test_split_only* is True, only test-split rendered pairs are
    evaluated (useful for ``data/rendered/`` automated eval).
    """
    import torch

    from chant_omr.inference.beam_search import DecodeConfig, decode_token_ids
    from chant_omr.inference.checkpoint import load_model_from_checkpoint
    from chant_omr.inference.predict import resolve_inference_device
    from chant_omr.inference.preprocess import prepare_inference_tensor

    pairs = discover_benchmark_pairs(benchmark_dir, test_split_only=test_split_only)
    if not pairs:
        return EvalReport()

    if limit is not None:
        pairs = pairs[:limit]

    torch_device = resolve_inference_device(device)
    model, tokenizer, _meta = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device=torch_device,
    )

    decode_config = DecodeConfig(
        beam_width=beam_width,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        grammar_constrained=grammar_constrained,
    )

    report = EvalReport()

    for img_path, gabc_path in pairs:
        try:
            ref_body = _read_ref_body(gabc_path)
        except (ValueError, UnicodeDecodeError) as exc:
            report.skipped.append((gabc_path, str(exc)))
            continue

        t0 = time.monotonic()
        pixel_values = prepare_inference_tensor(img_path, device=torch_device)
        with torch.inference_mode():
            token_ids = decode_token_ids(model, pixel_values, tokenizer, decode_config)
        pred_body = tokenizer.decode(token_ids, skip_special_tokens=True)
        elapsed = time.monotonic() - t0

        ged = gabc_edit_distance(pred_body, ref_body)
        neume_acc = neume_accuracy(pred_body, ref_body)
        validity = check_structural_validity(pred_body)

        report.samples.append(SampleResult(
            image_path=img_path,
            ref_path=gabc_path,
            pred_body=pred_body,
            ref_body=ref_body,
            ged=ged,
            neume_acc=neume_acc,
            validity=validity,
            elapsed_s=elapsed,
        ))

        if progress_callback is not None:
            progress_callback(len(report.samples), len(pairs), img_path, elapsed)

    return report


def format_eval_report(report: EvalReport) -> str:
    """Format an ``EvalReport`` as a human-readable summary."""
    lines: list[str] = []

    if report.count == 0:
        lines.append("No benchmark pairs found. Nothing to evaluate.")
        if report.skipped:
            lines.append(f"Skipped: {len(report.skipped)}")
            for path, reason in report.skipped:
                lines.append(f"  {path}: {reason}")
        return "\n".join(lines)

    lines.append(f"Evaluated {report.count} samples")
    lines.append("")
    lines.append("Aggregate metrics:")
    lines.append(f"  GED (mean normalized):     {report.mean_ged:.4f}")
    lines.append(f"  Neume accuracy (mean):     {report.mean_neume_accuracy:.4f}")
    lines.append(f"  Structural validity:       {report.structural_validity_rate:.1%}")
    lines.append(f"  Total inference time:      {report.total_elapsed_s:.1f}s")
    lines.append(
        f"  Mean time per sample:      {report.total_elapsed_s / report.count:.2f}s"
    )

    lines.append("")
    lines.append("Per-sample results:")
    lines.append(f"{'Sample':<40} {'GED':>6} {'Neume%':>7} {'Valid':>5}")
    lines.append("-" * 60)
    for s in report.samples:
        name = s.image_path.name
        if len(name) > 38:
            name = "..." + name[-35:]
        lines.append(
            f"{name:<40} {s.ged.normalized:>6.3f} {s.neume_acc.accuracy:>6.1%} "
            f"{'yes' if s.validity.is_valid else 'NO':>5}"
        )

    if report.skipped:
        lines.append("")
        lines.append(f"Skipped {len(report.skipped)} pairs:")
        for path, reason in report.skipped:
            lines.append(f"  {path.name}: {reason}")

    return "\n".join(lines)
