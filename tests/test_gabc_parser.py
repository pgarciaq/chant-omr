"""Tests for GABC parser."""

from chant_omr.data.gabc_parser import (
    NABC_NOT_SUPPORTED,
    extract_gabc_body,
    gabc_reject_reason,
    is_nabc_notation,
    parse_gabc,
)


class TestParseGABC:
    def test_simple_score(self):
        text = "name: Kyrie;\n%%\n(c4) Ky(f)ri(gf)e(h)"
        score = parse_gabc(text)
        assert score.name == "Kyrie"
        assert score.body == "(c4) Ky(f)ri(gf)e(h)"

    def test_no_header(self):
        text = "(c4) Ky(f)ri(gf)e(h)"
        score = parse_gabc(text)
        assert score.headers == {}
        assert score.body == "(c4) Ky(f)ri(gf)e(h)"

    def test_multiple_headers(self):
        text = "name: Kyrie XVII;\nmode: 6;\nannotation: XVII;\n%%\n(c4) Ky(f)ri(gf)e(h)"
        score = parse_gabc(text)
        assert score.name == "Kyrie XVII"
        assert score.headers["mode"] == "6"
        assert score.headers["annotation"] == "XVII"

    def test_empty_name(self):
        text = "%%\n(c4) Al(f)le(h)lu(g)ia.(f)"
        score = parse_gabc(text)
        assert score.name == ""

    def test_body_preserved(self):
        body = "(c4) Ky(f)ri(gf)e(h) *(;) e(ixhi)lé(h)i(g)son.(f) (::)"
        text = f"name: Kyrie;\n%%\n{body}"
        score = parse_gabc(text)
        assert score.body == body


class TestGabcValidation:
    def test_extract_gabc_body_last_segment(self):
        text = "name: a;\n%%\nannotation: 8\n%%\n(c4) Ky(f)ri(gf)e(h)"
        assert extract_gabc_body(text) == "(c4) Ky(f)ri(gf)e(h)"

    def test_extract_gabc_body_empty(self):
        import pytest

        with pytest.raises(ValueError, match="empty GABC body"):
            extract_gabc_body("name:;\n%%\n")

    def test_gabc_reject_reason_empty_stub(self):
        assert gabc_reject_reason(b"name:;\n%%\n") == "empty gabc body"

    def test_is_nabc_notation_pipe(self):
        text = "name: x;\n%%\n(c4) AL(e|/ta)le(g)ia.(g)"
        assert is_nabc_notation(text)

    def test_is_nabc_notation_plain(self):
        text = "name: x;\n%%\n(c4) Ky(f)ri(gf)e(h)"
        assert not is_nabc_notation(text)

    def test_nabc_not_supported_constant(self):
        assert "NABC" in NABC_NOT_SUPPORTED
