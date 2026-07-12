"""Assemble full GABC files from predicted bodies (ADR 0013)."""

from __future__ import annotations

DEFAULT_OMR_NAME = "OMR output"


def assemble_gabc(body: str, *, name: str = DEFAULT_OMR_NAME) -> str:
    """Wrap a decoded GABC body in a minimal valid file template."""
    body = body.strip()
    if not body:
        raise ValueError("decoded GABC body is empty")
    safe_name = name.strip() or DEFAULT_OMR_NAME
    return f"name: {safe_name};\n%%\n{body}\n"
