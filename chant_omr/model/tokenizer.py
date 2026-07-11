"""BPE tokenizer for GABC notation.

Trains a byte-pair encoding tokenizer on plain GABC **bodies** (text after the
final ``%%``). Headers are not part of the vocabulary; inference prepends them
separately when writing a full ``.gabc`` file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from chant_omr.data.gabc_parser import iter_plain_gabc_bodies
from chant_omr.data.gregobase import MANIFEST_FILENAME, Manifest

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

TOKENIZER_FILENAME = "tokenizer.json"
META_FILENAME = "meta.json"
DEFAULT_VOCAB_SIZE = 2048
DEFAULT_OUTPUT_DIR = Path("data/tokenizer")


@dataclass
class TokenizerTrainStats:
    """Summary counters from a tokenizer training run."""

    corpus_files: int
    corpus_chars: int
    vocab_size: int
    min_body_len: int
    gabc_dir: str
    output_dir: str
    trained_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class GABCTokenizer:
    """Thin wrapper around a HuggingFace ``tokenizers`` BPE model."""

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.pad_id = self._require_id(PAD_TOKEN)
        self.bos_id = self._require_id(BOS_TOKEN)
        self.eos_id = self._require_id(EOS_TOKEN)
        self.unk_id = self._require_id(UNK_TOKEN)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _require_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"special token missing from vocabulary: {token}")
        return token_id

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode a GABC body string to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a GABC body string."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, output_dir: Path, meta: TokenizerTrainStats | None = None) -> Path:
        """Persist ``tokenizer.json`` (and optional ``meta.json``) under *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = output_dir / TOKENIZER_FILENAME
        self._tokenizer.save(str(tokenizer_path))
        if meta is not None:
            meta_path = output_dir / META_FILENAME
            meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")
        return tokenizer_path

    @classmethod
    def load(cls, path: Path) -> GABCTokenizer:
        """Load a tokenizer from a directory or ``tokenizer.json`` file."""
        path = Path(path)
        tokenizer_path = path / TOKENIZER_FILENAME if path.is_dir() else path
        return cls(Tokenizer.from_file(str(tokenizer_path)))


def _build_tokenizer(vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    return tokenizer, trainer


def _attach_post_processor(tokenizer: Tokenizer) -> None:
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )


def collect_plain_corpus(
    gabc_dir: Path,
    *,
    min_body_len: int,
    use_manifest: bool = True,
) -> tuple[list[str], int]:
    """Return ``(bodies, file_count)`` for plain trainable GABC under *gabc_dir*."""
    gabc_dir = Path(gabc_dir)
    manifest = None
    if use_manifest:
        manifest_path = gabc_dir / MANIFEST_FILENAME
        if manifest_path.is_file():
            manifest = Manifest.load(manifest_path)

    bodies: list[str] = []
    file_count = 0
    for _path, body in iter_plain_gabc_bodies(gabc_dir, manifest, min_body_len=min_body_len):
        bodies.append(body)
        file_count += 1
    return bodies, file_count


def train_tokenizer(
    gabc_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    output_dir: Path | None = None,
    *,
    min_body_len: int = 20,
    use_manifest: bool = True,
) -> GABCTokenizer:
    """Train a BPE tokenizer on plain GABC bodies and save artifacts.

    Args:
        gabc_dir: Directory containing ``.gabc`` files (and optional manifest).
        vocab_size: Target vocabulary size (including special tokens).
        output_dir: Where to write ``tokenizer.json`` and ``meta.json``.
        min_body_len: Skip bodies shorter than this many characters.
        use_manifest: When ``manifest.json`` exists, only train on ``ok`` entries.

    Returns:
        Trained :class:`GABCTokenizer`.
    """
    gabc_dir = Path(gabc_dir)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)

    bodies, file_count = collect_plain_corpus(
        gabc_dir,
        min_body_len=min_body_len,
        use_manifest=use_manifest,
    )
    if not bodies:
        raise ValueError(f"no plain GABC bodies found under {gabc_dir}")

    tokenizer, trainer = _build_tokenizer(vocab_size)
    tokenizer.train_from_iterator(bodies, trainer)
    _attach_post_processor(tokenizer)

    wrapper = GABCTokenizer(tokenizer)
    stats = TokenizerTrainStats(
        corpus_files=file_count,
        corpus_chars=sum(len(body) for body in bodies),
        vocab_size=wrapper.vocab_size,
        min_body_len=min_body_len,
        gabc_dir=str(gabc_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        trained_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
    wrapper.save(output_dir, meta=stats)
    return wrapper
