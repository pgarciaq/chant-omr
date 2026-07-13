"""Decode diagnostics for overfit / smoke-test validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from chant_omr.data.gabc_parser import extract_gabc_body
from chant_omr.inference.beam_search import DecodeConfig, greedy_decode
from chant_omr.model.chant_omr_model import ChantOMR
from chant_omr.model.tokenizer import GABCTokenizer


@dataclass(frozen=True)
class PredictMetrics:
    """Side-by-side teacher-forcing vs greedy decode diagnostics."""

    encoder_patches: int
    memory_l2_norm: float
    greedy_preview: str
    gold_path: Path | None = None
    teacher_forcing_loss: float | None = None
    teacher_forcing_acc: float | None = None
    greedy_body_match: bool | None = None
    greedy_token_acc: float | None = None


def _content_token_ids(token_ids: list[int], *, bos_id: int, eos_id: int, pad_id: int) -> list[int]:
    """Strip special tokens from a generated or gold sequence."""
    out: list[int] = []
    for token_id in token_ids:
        if token_id in {bos_id, pad_id}:
            continue
        if token_id == eos_id:
            break
        out.append(token_id)
    return out


def _token_accuracy(predicted: list[int], gold: list[int]) -> float:
    if not gold:
        return 0.0
    matches = sum(pred == ref for pred, ref in zip(predicted, gold, strict=False))
    return matches / len(gold)


def resolve_reference_gabc(image_path: Path) -> Path | None:
    """Return a sidecar ``.gabc`` path when it exists next to the image."""
    sidecar = image_path.with_suffix(".gabc")
    return sidecar if sidecar.is_file() else None


def compute_predict_metrics(
    model: ChantOMR,
    pixel_values: torch.Tensor,
    tokenizer: GABCTokenizer,
    *,
    decode_config: DecodeConfig,
    reference_gabc_path: Path | None = None,
    preview_chars: int = 60,
) -> tuple[list[int], PredictMetrics]:
    """Encode, greedy-decode, and collect teacher-forcing vs greedy diagnostics."""
    pad_id = tokenizer.pad_id
    device = pixel_values.device

    with torch.inference_mode():
        memory = model.encode(pixel_values)
        greedy_ids = greedy_decode(
            model,
            memory,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            max_length=decode_config.max_length,
            repetition_penalty=decode_config.repetition_penalty,
        )
        greedy_body = tokenizer.decode(greedy_ids, skip_special_tokens=True)
        preview = greedy_body[:preview_chars]
        if len(greedy_body) > preview_chars:
            preview += "..."

        metrics = PredictMetrics(
            encoder_patches=memory.shape[1],
            memory_l2_norm=float(memory.norm().item()),
            greedy_preview=preview,
            gold_path=reference_gabc_path,
        )

        if reference_gabc_path is None:
            return greedy_ids, metrics

        gold_text = extract_gabc_body(reference_gabc_path.read_text(encoding="utf-8"))
        gold_ids = tokenizer.encode(gold_text, add_special_tokens=True)
        if len(gold_ids) < 2:
            return greedy_ids, metrics

        decoder_input = torch.tensor([gold_ids[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([gold_ids[1:]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(decoder_input)
        logits = model.decoder(decoder_input, memory, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        pred = logits.argmax(dim=-1)
        valid = labels[0] != pad_id
        if valid.any():
            tf_acc = float((pred[0][valid] == labels[0][valid]).float().mean().item())
        else:
            tf_acc = 0.0

        greedy_content = _content_token_ids(
            greedy_ids,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=pad_id,
        )
        gold_content = _content_token_ids(
            gold_ids,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=pad_id,
        )
        metrics = PredictMetrics(
            encoder_patches=metrics.encoder_patches,
            memory_l2_norm=metrics.memory_l2_norm,
            greedy_preview=metrics.greedy_preview,
            gold_path=reference_gabc_path,
            teacher_forcing_loss=float(loss.item()),
            teacher_forcing_acc=tf_acc,
            greedy_body_match=greedy_body == gold_text,
            greedy_token_acc=_token_accuracy(greedy_content, gold_content),
        )

    return greedy_ids, metrics


def format_predict_metrics(metrics: PredictMetrics) -> str:
    """Human-readable metrics block for ``--dump-metrics``."""
    lines = ["--- predict metrics ---", f"encoder_patches: {metrics.encoder_patches}"]
    lines.append(f"memory_l2_norm: {metrics.memory_l2_norm:.2f}")

    if metrics.gold_path is None:
        lines.append("gold: (no sidecar .gabc found)")
    else:
        lines.append(f"gold: {metrics.gold_path}")
        if metrics.teacher_forcing_loss is not None:
            lines.append("teacher_forcing:")
            lines.append(f"  loss: {metrics.teacher_forcing_loss:.4f}")
            lines.append(f"  acc:  {metrics.teacher_forcing_acc * 100:.1f}%")
        if metrics.greedy_body_match is not None:
            lines.append("greedy (beam_width=1 diagnostic):")
            lines.append(f"  preview: {metrics.greedy_preview}")
            lines.append(f"  exact_body_match: {'yes' if metrics.greedy_body_match else 'no'}")
            lines.append(f"  token_acc_vs_gold: {metrics.greedy_token_acc * 100:.1f}%")
    if metrics.gold_path is None:
        lines.append("greedy (beam_width=1 diagnostic):")
        lines.append(f"  preview: {metrics.greedy_preview}")

    return "\n".join(lines)
