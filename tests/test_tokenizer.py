"""Tests for BPE tokenizer training and encoding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chant_omr.data.gabc_parser import extract_gabc_body, iter_plain_gabc_bodies
from chant_omr.model.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    META_FILENAME,
    PAD_TOKEN,
    TOKENIZER_FILENAME,
    UNK_TOKEN,
    GABCTokenizer,
    train_tokenizer,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "gregobase"


@pytest.fixture
def fixture_gabc_dir() -> Path:
    return FIXTURES_DIR


class TestPlainCorpusIterator:
    def test_includes_plain_scores(self, fixture_gabc_dir: Path):
        bodies = [body for _path, body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)]
        assert "(c4) Re(f)spi(g)ce(h) Do(j)mi(j)ne.(h)" in bodies
        assert any("AU(h)ri(h)bus" in body for body in bodies)

    def test_excludes_nabc(self, fixture_gabc_dir: Path):
        paths = [
            path.name
            for path, _body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)
        ]
        assert "nabc_sample.gabc" not in paths

    def test_excludes_empty_stub(self, fixture_gabc_dir: Path):
        paths = [
            path.name
            for path, _body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)
        ]
        assert "empty_stub.gabc" not in paths

    def test_min_body_len_filter(self, fixture_gabc_dir: Path):
        all_paths = {
            path.name for path, _body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)
        }
        long_only_paths = {
            path.name for path, _body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=30)
        }
        assert "haec_est_virgo.gabc" in all_paths
        assert "haec_est_virgo.gabc" not in long_only_paths
        assert "double_header.gabc" in long_only_paths
        assert "respice_domine.gabc" in long_only_paths


class TestTrainTokenizer:
    def test_train_on_fixture_corpus(self, fixture_gabc_dir: Path, tmp_path: Path):
        tokenizer = train_tokenizer(
            fixture_gabc_dir,
            vocab_size=128,
            output_dir=tmp_path,
            min_body_len=10,
            use_manifest=False,
        )
        assert isinstance(tokenizer, GABCTokenizer)
        assert tokenizer.vocab_size <= 128
        assert tokenizer.vocab_size >= 4
        assert (tmp_path / TOKENIZER_FILENAME).is_file()
        assert (tmp_path / META_FILENAME).is_file()

        meta = json.loads((tmp_path / META_FILENAME).read_text(encoding="utf-8"))
        assert meta["corpus_files"] >= 3
        assert meta["corpus_chars"] > 0
        assert meta["min_body_len"] == 10

    def test_empty_corpus_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="no plain GABC bodies"):
            train_tokenizer(
                empty_dir,
                vocab_size=64,
                output_dir=tmp_path / "out",
                use_manifest=False,
            )


class TestGABCTokenizerRoundTrip:
    @pytest.fixture
    def tokenizer(self, fixture_gabc_dir: Path, tmp_path: Path) -> GABCTokenizer:
        return train_tokenizer(
            fixture_gabc_dir,
            vocab_size=256,
            output_dir=tmp_path / "tokenizer",
            min_body_len=10,
            use_manifest=False,
        )

    def test_special_token_ids_reserved(self, tokenizer: GABCTokenizer):
        assert tokenizer._tokenizer.token_to_id(PAD_TOKEN) == tokenizer.pad_id
        assert tokenizer._tokenizer.token_to_id(BOS_TOKEN) == tokenizer.bos_id
        assert tokenizer._tokenizer.token_to_id(EOS_TOKEN) == tokenizer.eos_id
        assert tokenizer._tokenizer.token_to_id(UNK_TOKEN) == tokenizer.unk_id

    def test_round_trip_corpus_bodies(self, tokenizer: GABCTokenizer, fixture_gabc_dir: Path):
        for _path, body in iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10):
            ids = tokenizer.encode(body)
            assert tokenizer.bos_id in ids
            assert tokenizer.eos_id in ids
            assert tokenizer.decode(ids) == body

    def test_round_trip_without_special_tokens(
        self, tokenizer: GABCTokenizer, fixture_gabc_dir: Path
    ):
        _path, body = next(iter(iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)))
        ids = tokenizer.encode(body, add_special_tokens=False)
        assert tokenizer.bos_id not in ids
        assert tokenizer.eos_id not in ids
        assert tokenizer.decode(ids) == body

    def test_body_only_not_headers(self, tokenizer: GABCTokenizer, fixture_gabc_dir: Path):
        raw = (fixture_gabc_dir / "respice_domine.gabc").read_text(encoding="utf-8")
        body = extract_gabc_body(raw)
        assert "name:" not in body
        ids = tokenizer.encode(body, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == body
        assert "Respice Domine" not in decoded

    def test_load_saved_tokenizer(self, fixture_gabc_dir: Path, tmp_path: Path):
        output_dir = tmp_path / "saved"
        trained = train_tokenizer(
            fixture_gabc_dir,
            vocab_size=256,
            output_dir=output_dir,
            min_body_len=10,
            use_manifest=False,
        )
        loaded = GABCTokenizer.load(output_dir)
        _path, body = next(iter(iter_plain_gabc_bodies(fixture_gabc_dir, min_body_len=10)))
        assert loaded.decode(loaded.encode(body)) == body
        assert loaded.vocab_size == trained.vocab_size
