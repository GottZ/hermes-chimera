"""K-1.2 — English G2P against the template's golden words (T1 exact; golden/phonetics-g2p.json):
504 words x {dictionary, no dictionary} x {reduce, full}, letter-to-sound, 9 ARPAbet strings.
Plus the template's own unit tests of the front end (crates/chimera/tests/english.rs).

The dictionary is the plugin's ``chimera_vocal/data/cmudict.txt`` via ``EnglishG2P.default()``
(byte-identical to the template's gunzipped cmudict.txt.gz, K-1.9).

Red proof: letter-to-sound rules tried in alphabetical instead of source order.
"""

import functools

from chimera_vocal.core.types import Intonation
from chimera_vocal.text.english import EnglishG2P, arpabet_to_segments
from support import golden


@functools.lru_cache(maxsize=None)
def g2ps() -> tuple[EnglishG2P, EnglishG2P]:
    return EnglishG2P.default(), EnglishG2P()


def ipa(segs) -> str:
    return "".join(s.ipa for s in segs)


def test_golden_g2p(mismatches):
    d, nd = g2ps()
    fx = golden("phonetics-g2p")
    assert len(fx["words"]) == 504 and len(fx["arpabet"]) == 9
    m = mismatches("g2p")
    for c in fx["words"]:
        w = c["w"]
        m.check(f"{w!r} dict reduce", c["dict"]["reduce"], d.word(w, True))
        m.check(f"{w!r} dict full", c["dict"]["full"], d.word(w, False))
        m.check(f"{w!r} no dict reduce", c["noDict"]["reduce"], nd.word(w, True))
        m.check(f"{w!r} no dict full", c["noDict"]["full"], nd.word(w, False))
        m.check(f"{w!r} letter_to_sound", c["lts"], nd.letter_to_sound(w))
    for c in fx["arpabet"]:
        m.check(f"arpabet {c['p']!r}", c["segments"], arpabet_to_segments(c["p"]))
    assert m.n == 504 * 5 + 9
    m.report()


def test_letter_to_sound_ignores_dictionary(mismatches):
    """letter_to_sound is the rule path only: the same with and without a dictionary."""
    d, nd = g2ps()
    m = mismatches("letter_to_sound with dictionary")
    for c in golden("phonetics-g2p")["words"][:60]:
        m.check(repr(c["w"]), c["lts"], d.letter_to_sound(c["w"]))
    m.report()


# --- crates/chimera/tests/english.rs ---------------------------------------------------------------------

def test_dictionary_words_with_stress_and_allophones():
    d, _ = g2ps()
    p = d.to_phrases("Hello world")[0]
    assert ipa(p.segments) == "həloʊwɝɫd"
    assert next(s for s in p.segments if s.ipa == "oʊ").stress == 1
    assert ipa(d.to_phrases("water")[0].segments) == "wɔɾɚ"


def test_punctuation_makes_phrases_and_question_intonation():
    d, _ = g2ps()
    ph = d.to_phrases("Is it on? Yes, it is.")
    assert len(ph) == 3
    assert ph[0].intonation is Intonation.QUESTION
    assert ph[2].intonation is Intonation.STATEMENT


def test_numbers_are_spelled_out_and_unknown_words_fall_back_to_rules():
    d, nd = g2ps()
    assert d.to_phrases("42")[0].words == ["forty", "two"]
    assert len(ipa(nd.letter_to_sound("zorblat"))) > 4


def test_a_phrase_lists_the_words_said_one_per_word_end():
    d, _ = g2ps()
    for text in ("hello -- world", "rock ' roll", "it's - fine"):
        for p in d.to_phrases(text):
            ends = sum(1 for s in p.segments if s.word_end is True)
            assert len(p.words) == ends, f"{text!r}: {p.words}"
    assert d.to_phrases("hello -- world")[0].words == ["hello", "world"]


def test_words_that_are_also_object_property_names_do_not_break_the_front_end():
    d, nd = g2ps()
    for w in ("constructor", "toString", "valueOf", "hasOwnProperty", "__proto__", "prototype", "__class__",
              "__init__", "get", "keys", "items"):
        assert len(d.to_phrases(f"The {w} is ready.")[0].segments) > 5, w
    assert len(nd.to_phrases("constructor")[0].segments) > 3


def test_load_dict_later_line_wins_and_needs_a_word_before_the_tab():
    """english.rs:119-129: split at '\\n', first tab at index > 0, HashMap insert (later line wins)."""
    g = EnglishG2P()
    g.load_dict("zork\tZ AO1 R K\nzork\tB IY1\n\tP IY1\nnotab P IY1\n")
    assert [(s.ipa, s.stress) for s in g.word("zork", False)] == [("b", 0), ("i", 1)]
    assert ipa(g.word("notab", False)) == ipa(EnglishG2P().word("notab", False))   # no entry: rules


def test_dictionary_mapping_constructor_matches_load_dict():
    a = EnglishG2P({"zork": "Z AO1 R K"})
    b = EnglishG2P()
    b.load_dict("zork\tZ AO1 R K")
    assert [s.to_json() for s in a.word("zork", True)] == [s.to_json() for s in b.word("zork", True)]
    assert ipa(a.word("zork", True)) == "zɔɹk"
