"""Scan rendered corpus and report token-length distribution (#33)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chant_omr.data.dataset import discover_rendered_pairs
from chant_omr.data.gabc_parser import extract_gabc_body
from chant_omr.model.tokenizer import GABCTokenizer


@dataclass(frozen=True)
class TokenLengthReport:
    """Token-length distribution over a rendered corpus."""

    total_pairs: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    p50: int
    p75: int
    p90: int
    p95: int
    p99: int
    exceed_limit: int
    limit: int
    longest_stems: list[tuple[str, int]]


def audit_token_lengths(
    rendered_dir: Path,
    tokenizer: GABCTokenizer,
    *,
    max_seq_len: int = 2048,
    top_n: int = 10,
    min_body_len: int = 20,
) -> TokenLengthReport:
    """Tokenize every rendered GABC body and compute length statistics."""
    samples = discover_rendered_pairs(rendered_dir, min_body_len=min_body_len)
    if not samples:
        raise ValueError(f"no rendered pairs found in {rendered_dir}")

    lengths: list[tuple[str, int]] = []
    for sample in samples:
        body = extract_gabc_body(sample.gabc_path.read_text(encoding="utf-8"))
        ids = tokenizer.encode(body, add_special_tokens=True)
        lengths.append((sample.stem, len(ids)))

    lengths.sort(key=lambda item: item[1])
    token_counts = [n for _, n in lengths]
    total = len(token_counts)

    def percentile(p: float) -> int:
        idx = int(p / 100.0 * (total - 1))
        return token_counts[idx]

    longest = sorted(lengths, key=lambda item: item[1], reverse=True)[:top_n]

    return TokenLengthReport(
        total_pairs=total,
        min_tokens=token_counts[0],
        max_tokens=token_counts[-1],
        mean_tokens=sum(token_counts) / total,
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        exceed_limit=sum(1 for n in token_counts if n > max_seq_len),
        limit=max_seq_len,
        longest_stems=longest,
    )


def format_token_audit(report: TokenLengthReport) -> str:
    """Human-readable token audit report."""
    lines = [
        f"Token length audit ({report.total_pairs} rendered pairs, limit={report.limit})",
        "",
        f"  min:  {report.min_tokens}",
        f"  p50:  {report.p50}",
        f"  p75:  {report.p75}",
        f"  p90:  {report.p90}",
        f"  p95:  {report.p95}",
        f"  p99:  {report.p99}",
        f"  max:  {report.max_tokens}",
        f"  mean: {report.mean_tokens:.1f}",
        "",
    ]
    if report.exceed_limit > 0:
        pct = report.exceed_limit / report.total_pairs * 100
        lines.append(
            f"  TRUNCATED: {report.exceed_limit} samples ({pct:.2f}%) "
            f"exceed max_seq_len={report.limit}"
        )
    else:
        lines.append(f"  All samples fit within max_seq_len={report.limit}")
    lines.append("")
    lines.append(f"  Top {len(report.longest_stems)} longest:")
    for stem, n in report.longest_stems:
        marker = " *** TRUNCATED" if n > report.limit else ""
        lines.append(f"    {stem}: {n} tokens{marker}")
    return "\n".join(lines)
