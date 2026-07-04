"""Tests for GABC parser."""

from chant_omr.data.gabc_parser import GABCScore, parse_gabc


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
