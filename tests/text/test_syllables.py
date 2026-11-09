"""K-1.4 — lyrics -> syllables against the template's goldens (T1 exact; golden/phonetics-syllables.json):
33 lyrics x {dictionary, no dictionary}, 44 spelling splits, 8 syllabifications. Plus the template's unit
tests (crates/chimera/tests/syllables.rs).

``syllabify_segments`` is compared on ``onset``/``nucleus``/``coda`` only (the template returns
``SyllableParts``).

Red proof: recognise a melisma dash only at the start of a word.
"""

import functools

from chimera_vocal.text.english import EnglishG2P, arpabet_to_segments
from chimera_vocal.text.syllables import lyrics_to_syllables, split_spelling, syllabify_segments
from support import golden


@functools.lru_cache(maxsize=None)
def g2ps() -> tuple[EnglishG2P, EnglishG2P]:
    return EnglishG2P.default(), EnglishG2P()


def parts(p) -> dict:
    return {"onset": [s.to_json() for s in p.onset], "nucleus": p.nucleus.to_json(),
            "coda": [s.to_json() for s in p.coda]}


def test_golden_syllables(mismatches):
    d, nd = g2ps()
    fx = golden("phonetics-syllables")
    assert (len(fx["lyrics"]), len(fx["spelling"]), len(fx["syllabify"])) == (33, 44, 8)
    m = mismatches("syllables")
    for c in fx["lyrics"]:
        m.check(f"{c['text']!r} dict", c["dict"], lyrics_to_syllables(c["text"], d))
        m.check(f"{c['text']!r} no dict", c["noDict"], lyrics_to_syllables(c["text"], nd))
    for c in fx["spelling"]:
        m.check(f"split_spelling({c['word']!r}, {c['n']})", c["parts"], split_spelling(c["word"], c["n"]))
    for c in fx["syllabify"]:
        m.check(f"syllabify {c['w']!r}", c["parts"], [parts(p) for p in syllabify_segments(d.word(c["w"], False))])
    assert m.n == 33 * 2 + 44 + 8
    m.report()


# --- crates/chimera/tests/syllables.rs -------------------------------------------------------------------

DICT = {"twinkle": "T W IH1 NG K AH0 L", "little": "L IH1 T AH0 L", "star": "S T AA1 R", "how": "HH AW1",
        "i": "AY1", "wonder": "W AH1 N D ER0", "what": "W AH1 T", "you": "Y UW1", "are": "AA1 R",
        "extra": "EH1 K S T R AH0", "of": "AH1 V"}


def ipa(xs) -> str:
    return "".join(x.ipa for x in xs)


def test_maximal_onset_with_legal_english_clusters():
    tw = syllabify_segments(arpabet_to_segments(DICT["twinkle"]))
    assert [(ipa(s.onset), s.nucleus.ipa, ipa(s.coda)) for s in tw] == [("tw", "ɪ", "ŋ"), ("k", "ə", "l")]
    ex = syllabify_segments(arpabet_to_segments(DICT["extra"]))      # k + str -> ek-stra
    assert [ipa(s.onset) for s in ex] == ["", "stɹ"]
    assert ipa(ex[0].coda) == "k"


def test_spelling_splits_to_match():
    assert split_spelling("twinkle", 2) == ["twin", "kle"]
    assert split_spelling("little", 2) == ["lit", "tle"]
    assert split_spelling("wonder", 2) == ["won", "der"]
    assert split_spelling("are", 1) == ["are"]


def test_lyrics_words_melisma_phrase_ends_unreduced_function_words():
    syl = lyrics_to_syllables("Twinkle little star, how I - wonder", EnglishG2P(DICT))
    assert [s.text for s in syl] == ["Twin", "kle", "lit", "tle", "star", "how", "I", "~", "won", "der"]
    assert syl[4].phrase_end
    assert syl[7].melisma is True
    assert syl[7].nucleus.ipa == "aɪ"
    assert syl[1].word_end and not syl[0].word_end


def test_sung_function_words_keep_full_vowels():
    """The singing path calls word(w, reduce=False) (syllables.rs:202): 'of' keeps ʌ, unlike speech."""
    syl = lyrics_to_syllables("of", EnglishG2P(DICT))
    assert [s.nucleus.ipa for s in syl] == ["ʌ"]
