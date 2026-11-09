"""K-1.3 — EnglishG2P.to_phrases against the template's golden texts (T1 exact;
golden/phonetics-phrases.json): 38 texts x {dictionary, no dictionary}, including numbers >= 1e6 written
digit by digit from the JS number format (``1e+21``, ``Infinity``), abbreviations, typographic characters.

Red proof (TESTPLAN): spell big numbers from Python ``int`` instead of the JS number format.
"""

import functools

from chimera_vocal.text.english import EnglishG2P
from support import golden


@functools.lru_cache(maxsize=None)
def g2ps() -> tuple[EnglishG2P, EnglishG2P]:
    return EnglishG2P.default(), EnglishG2P()


def test_golden_phrases(mismatches):
    d, nd = g2ps()
    cases = golden("phonetics-phrases")
    assert len(cases) == 38
    m = mismatches("to_phrases")
    for c in cases:
        m.check(f"{c['text'][:60]!r} dict", c["dict"], d.to_phrases(c["text"]))
        m.check(f"{c['text'][:60]!r} no dict", c["noDict"], nd.to_phrases(c["text"]))
    m.report()


def test_big_numbers_follow_the_js_number_format():
    """>= 1e6: digit by digit from Number.prototype.toString (english.rs:297-323); exactly representable
    integers above 2^53 round like JS (123456789012345678 -> "123456789012345680")."""
    _, nd = g2ps()
    words = [w for p in nd.to_phrases("123456789012345678") for w in p.words]
    assert words == ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "one",
                     "two", "three", "four", "five", "six", "eight", "zero"]
    words = [w for p in nd.to_phrases("1000000000000000000000") for w in p.words]   # 1e21 -> "1e+21"
    assert words == ["one", "two", "one"]
    assert nd.to_phrases("9" * 400) == []                                            # "Infinity": no sounds
