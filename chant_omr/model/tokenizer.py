"""BPE tokenizer for GABC notation.

Trains a byte-pair encoding tokenizer on the GABC corpus. GABC has a
compact vocabulary:

    Structural: ( ) / // :: * + - . , ; f g h i j k l m
    Neumes:     punctum, virga, clivis, podatus, torculus, etc.
               encoded as letter sequences inside parentheses
    Text:       Latin syllables between neume groups
    Clefs:      (c1) (c2) (c3) (c4) (f3) (f4)
    Modifiers:  ~ < > ' ` etc.

A BPE vocabulary of 1000-2000 tokens should capture common neume
patterns and Latin syllable sequences efficiently.
"""

from __future__ import annotations

from pathlib import Path


def train_tokenizer(
    gabc_dir: Path,
    vocab_size: int = 2048,
    output_path: Path | None = None,
):
    """Train a BPE tokenizer on a corpus of GABC files.

    Args:
        gabc_dir: Directory containing .gabc files.
        vocab_size: Target vocabulary size.
        output_path: Where to save the trained tokenizer.

    Returns:
        Trained tokenizer.
    """
    raise NotImplementedError("Tokenizer training not yet implemented")
