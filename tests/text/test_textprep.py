"""K-1.8 — prepare_text, the plugin's own normalisation before the template G2P.

This table is the specification of ``prepare_text`` (exact strings). Rule order is binding:
(1) NFKC + typographic characters -> ASCII, (2) abbreviations, (3) numbers, (4) diacritics -> base letters,
(5) everything outside ``[A-Za-z0-9 '.,;:!?-]`` -> space, brackets -> comma pause, whitespace and comma
cleanup. Conventions fixed here:
- number words are lower case, space separated, without "and" ("one hundred five"); ordinals change the last
  word ("twenty first"); times "H:MM" -> "three thirty" / "ten oh five" / "seven o'clock"; integers below
  10^15 become words, larger ones stay digits (the template reads them digit by digit);
- a number word next to a letter is separated by one space ("MP3" -> "MP three"), not next to a hyphen;
- "No." becomes "number" only before a digit; "St." is "Saint" before a capitalised word, else "Street";
  an abbreviation at the very end of the text keeps one "." (sentence end);
- "a.m."/"p.m." -> "ay em"/"pee em", "U.S." -> "U S" (letter names the dictionary knows);
- whitespace collapses to one space, no space before ``.,;:!?``; a comma right before other punctuation or at
  either end is dropped;
- the result is "" when no letter or digit remains (the speech path raises TextError);
  prepare_text never raises for a str.

Red proof (TESTPLAN): run the abbreviation step after the number step ("No. 5" then misses its digit).
"""

import functools
import re

import pytest

from chimera_vocal.text.english import EnglishG2P
from chimera_vocal.text.textprep import prepare_text

CASES = [
    # (1) NFKC, typography, whitespace
    ("It’s fine.", "It's fine."),
    ("‘quoted’ word", "'quoted' word"),
    ("“Hello,” she said.", "Hello, she said."),
    ("Wait… what?", "Wait... what?"),
    ("yes – no — maybe", "yes - no - maybe"),
    ("ﬁne ﬂow", "fine flow"),
    ("Ｆｕｌｌ width", "Full width"),
    ("x²", "x two"),
    ("Hello\tworld\nagain", "Hello world again"),
    ("  spaced   out  ", "spaced out"),
    ("non breaking", "non breaking"),
    # (2) abbreviations
    ("Dr. Smith is here.", "Doctor Smith is here."),
    ("Mr. and Mrs. Jones", "Mister and Missus Jones"),
    ("Ms. Lee", "Miz Lee"),
    ("Prof. Brown", "Professor Brown"),
    ("St. Louis", "Saint Louis"),
    ("Main St.", "Main Street."),
    ("cats vs. dogs", "cats versus dogs"),
    ("apples, pears, etc.", "apples, pears, et cetera."),
    ("fruit, e.g. apples", "fruit, for example apples"),
    ("one thing, i.e. this", "one thing, that is this"),
    ("No. 5 is next", "number five is next"),
    ("No.7", "number seven"),
    ("I said no.", "I said no."),
    ("at 7 a.m. sharp", "at seven ay em sharp"),
    ("3 p.m.", "three pee em."),
    ("Jan. 5", "January five"),
    ("Sept. 1st", "September first"),
    ("Dec. 25th", "December twenty fifth"),
    ("the U.S. economy", "the U S economy"),
    ("Jr. and Sr.", "Junior and Senior."),
    ("Mt. Everest", "Mount Everest"),
    ("He met the General. Then he left.", "He met the General. Then he left."),
    ("It is approx. 5 km away.", "It is approximately five km away."),
    # (3) numbers
    ("1,000,000", "one million"),
    ("I have 1,000,000 reasons.", "I have one million reasons."),
    ("1,234,567", "one million two hundred thirty four thousand five hundred sixty seven"),
    ("3.14", "three point one four"),
    ("3.14.", "three point one four."),
    ("0.5", "zero point five"),
    ("-2.5", "minus two point five"),
    ("It was -5 outside.", "It was minus five outside."),
    ("1st", "first"),
    ("2nd 3rd 4th", "second third fourth"),
    ("11th 12th 13th", "eleventh twelfth thirteenth"),
    ("21st", "twenty first"),
    ("20th century", "twentieth century"),
    ("100th", "one hundredth"),
    ("101st", "one hundred first"),
    ("3:30", "three thirty"),
    ("10:05", "ten oh five"),
    ("7:00", "seven o'clock"),
    ("12:45 p.m.", "twelve forty five pee em."),
    ("42", "forty two"),
    ("105", "one hundred five"),
    ("1000", "one thousand"),
    ("1999", "one thousand nine hundred ninety nine"),
    ("2026", "two thousand twenty six"),
    ("1000000000000", "one trillion"),
    ("999999999999999", "nine hundred ninety nine trillion nine hundred ninety nine billion nine hundred "
                        "ninety nine million nine hundred ninety nine thousand nine hundred ninety nine"),
    ("1000000000000000", "1000000000000000"),
    ("0000000000000000000007", "seven"),
    ("10000000000000000th", "10000000000000000th"),
    ("9" * 5000, "9" * 5000),
    ("1,2,3", "one,two,three"),
    ("1,234.56", "one thousand two hundred thirty four point five six"),
    ("COVID-19", "COVID-nineteen"),
    ("MP3", "MP three"),
    # (4) diacritics
    ("café", "cafe"),
    ("café", "cafe"),
    ("naïve café", "naive cafe"),
    ("Zürich", "Zurich"),
    ("Ångström", "Angstrom"),
    ("Straße", "Strasse"),
    ("œuvre", "oeuvre"),
    ("Øresund", "Oresund"),
    ("Łódź", "Lodz"),
    ("þorn and ðat", "thorn and dat"),
    ("Æsir", "AEsir"),
    ("İstanbul", "Istanbul"),
    # (5) other characters, brackets, cleanup
    ("Hello (world) again", "Hello, world, again"),
    ("Call me (maybe).", "Call me, maybe."),
    ("a [b] c", "a, b, c"),
    ("(Note) this", "Note, this"),
    ("#hashtag & stuff", "hashtag stuff"),
    ("C++ rocks", "C rocks"),
    ("rock'n'roll", "rock'n'roll"),
    ("well-known", "well-known"),
    ("a/b", "a b"),
    ("Hi \U0001F600", "Hi"),
    # (6) nothing speakable -> ""
    ("", ""),
    ("   ", ""),
    ("Привет, мир!", ""),
    ("日本語", ""),
    ("\U0001F600\U0001F389", ""),
    ("!!! ???", ""),
    ("- - -", ""),
]

ALLOWED = re.compile(r"[A-Za-z0-9 '.,;:!?-]*")


def test_table_size():
    assert len(CASES) >= 60
    assert len({i for i, _ in CASES}) == len(CASES)


@pytest.mark.parametrize("text,expected", CASES, ids=[repr(i)[:40] for i, _ in CASES])
def test_textprep(text, expected):
    got = prepare_text(text)
    assert got == expected
    assert got.isascii() and ALLOWED.fullmatch(got), got
    assert prepare_text(got) == got, "not idempotent"


@functools.lru_cache(maxsize=None)
def g2p() -> EnglishG2P:
    return EnglishG2P.default()


def test_textprep_fixes_template_weaknesses():
    """The documented G2P weaknesses on LLM text (module docstring of chimera_vocal.text.textprep) are gone after prepare_text."""
    g = g2p()
    ph = g.to_phrases(prepare_text("I have 1,000,000 reasons."))
    assert len(ph) == 1 and ph[0].words[-3:] == ["one", "million", "reasons"]
    ph = g.to_phrases(prepare_text("Dr. Smith said 3.14."))
    assert len(ph) == 1 and ph[0].words == ["doctor", "smith", "said", "three", "point", "one", "four"]
    ph = g.to_phrases(prepare_text("A café at 3:30."))
    assert [w for p in ph for w in p.words] == ["a", "cafe", "at", "three", "thirty"]


_ALPHABET = ("abcXYZ019 .,;:!?-'\"()[]{}/\\#&%$€£°²³½\t\n\r ’“–…éüßæøœłþðİı"
             "́̃͡ПриветΣσς日本語😀\ud83d﻿​ ")


def test_prepare_text_never_raises_and_stays_in_contract():
    """2 000 random strings (fixed seed) incl. lone surrogates, combining marks, other scripts, digit runs."""
    import random

    rnd = random.Random(20261109)
    for _ in range(2000):
        text = "".join(rnd.choice(_ALPHABET) for _ in range(rnd.randint(0, 60)))
        if rnd.random() < 0.05:
            text += "7" * rnd.randint(14, 5000)
        got = prepare_text(text)
        assert got.isascii() and ALLOWED.fullmatch(got), (text, got)
        assert prepare_text(got) == got, (text, got)
        assert got == "" or re.search(r"[A-Za-z0-9]", got), (text, got)


def test_front_end_never_raises_on_raw_or_prepared_text():
    """to_phrases and lyrics_to_syllables accept any str (the template's own behaviour on odd input)."""
    import random

    from chimera_vocal.text.syllables import lyrics_to_syllables

    rnd = random.Random(42)
    g = g2p()
    for _ in range(500):
        text = "".join(rnd.choice(_ALPHABET) for _ in range(rnd.randint(0, 40)))
        g.to_phrases(text)
        g.to_phrases(prepare_text(text))
        lyrics_to_syllables(text, g)
