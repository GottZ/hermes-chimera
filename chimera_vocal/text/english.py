"""English text -> phoneme segments (port of phonetics/english.rs).

CMUdict (ARPAbet with stress) when loaded, letter-to-sound rules otherwise; American flapping and dark /l/;
numbers spelled out; punctuation -> phrase breaks and intonation. The behaviour, including its quirks, is the
template's, so plans match its goldens exactly: numbers from 1e6 on are read digit by digit from the JS
number format ("1e+21", "Infinity"), any character outside ``[a-z'-]`` separates words, and only
``.!?;:,`` end a phrase. The plugin's own clean-up of LLM text lives in ``textprep.prepare_text``.
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Mapping

from ..core.types import Intonation, Phrase, Segment

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_ARPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
VOWELS = frozenset(("ɑ", "æ", "ʌ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ɝ", "ɚ", "eɪ", "ɪ", "i", "oʊ", "ɔɪ", "ʊ", "u"))

# The dictionary keeps one pronunciation per word; for these homographs it kept the rarer one (the verb
# "wind", past-tense "read"). Telling them apart needs grammar; the common reading wins.
_HOMOGRAPHS = {"wind": "W IH1 N D", "read": "R IY1 D", "live": "L IH1 V", "lead": "L IY1 D",
               "close": "K L OW1 Z", "wound": "W UW1 N D"}

# monosyllabic function words are unstressed in running speech (and reduce)
_FUNCTION_WORDS = frozenset((
    "a an the of to in on at by for from with as and or but nor so if is am are was were be been do does did has "
    "have had will would can could should shall may might must it its i you he she we they me him her us them my "
    "your his our their this that than then there not no").split(" "))
_REDUCIBLE = frozenset(("ʌ", "æ", "ɑ", "ɛ", "ɔ", "u", "ʊ"))
_KEEP_FULL = frozenset(("you", "do", "to", "who", "not", "no"))

# JavaScript's \s (and what String.prototype.trim removes)
JS_SPACE = frozenset("\t\n\x0b\x0c\r \xa0     　﻿"
                     + "".join(chr(c) for c in range(0x2000, 0x200B)))
_JS_SPACE_RE = re.compile("[" + "".join(sorted(JS_SPACE)) + "]")
_PHRASE_PUNCT = frozenset(".!?;:,")
_SPLIT_PUNCT = re.compile(r"([.!?;:,]+)")
_NOT_WORD = re.compile(r"[^a-z'\-]")
_DIGITS = re.compile(r"[0-9]+")


class EnglishG2P:
    """English letter-to-phoneme front end.

    ``dictionary`` maps a lower-case word to its CMUdict phones ("HH AH0 L OW1"); None = no dictionary
    (homographs and letter-to-sound rules only). Instances are immutable after construction/``load_dict`` and
    safe to share between threads.
    """

    _default: "EnglishG2P | None" = None
    _default_lock = threading.Lock()

    def __init__(self, dictionary: Mapping[str, str] | None = None) -> None:
        self._dict: dict[str, str] | None = dict(dictionary) if dictionary is not None else None

    @classmethod
    def default(cls) -> "EnglishG2P":
        """The shared instance with the bundled CMU Pronouncing Dictionary, loaded on first use (once per
        process, also when several threads ask at the same time)."""
        g = EnglishG2P._default
        if g is None:
            with EnglishG2P._default_lock:
                g = EnglishG2P._default
                if g is None:
                    g = EnglishG2P()
                    g.load_dict((DATA_DIR / "cmudict.txt").read_text(encoding="utf-8"))
                    EnglishG2P._default = g
        return g

    def is_loaded(self) -> bool:
        return self._dict is not None

    def load_dict(self, text: str) -> None:
        """``word<TAB>PHONES`` lines: split at newlines, the first tab must follow at least one character, a
        later line wins."""
        m: dict[str, str] = {}
        for line in text.split("\n"):
            tab = line.find("\t")
            if tab > 0:
                m[line[:tab]] = line[tab + 1:]
        self._dict = m

    def to_phrases(self, text: str) -> list[Phrase]:
        """Text -> phrases of segments, one per run of ``.!?;:,`` (english.rs:132-177)."""
        phrases: list[Phrase] = []
        words: list[str] = []

        def flush(delim: str) -> None:
            if not words:
                return
            segs: list[Segment] = []
            # the words said: a token without sound ("--", a lone apostrophe) is left out, so the n-th word ends
            # at the n-th word end
            said: list[str] = []
            for w in words:
                ws = self.word(w, True)
                if not ws:
                    continue
                ws[-1].word_end = True
                segs.extend(ws)
                said.append(w)
            _apply_allophones(segs)
            question = "?" in delim
            if any(c in delim for c in ".!?"):
                pause = 0.35
            elif delim:
                pause = 0.18
            else:
                pause = 0.3
            words.clear()
            if segs:
                phrases.append(Phrase(segs, Intonation.QUESTION if question else Intonation.STATEMENT, pause, said))

        for part in _SPLIT_PUNCT.split(_expand_numbers(text.replace("’", "'"))):
            if part and all(c in _PHRASE_PUNCT for c in part):
                flush(part)
            else:
                words.extend(w for w in _NOT_WORD.split(part.lower()) if w)
        flush("")
        return phrases

    def word(self, w: str, reduce: bool) -> list[Segment]:
        """One word -> segments with lexical stress; ``reduce``: weak forms of function words (speech)."""
        clean = w.lower().strip("'")
        entry = _HOMOGRAPHS.get(clean)
        if entry is None and self._dict is not None:
            entry = self._dict.get(clean)
        if entry:
            segs = arpabet_to_segments(entry)
            if reduce and clean in _FUNCTION_WORDS:
                nv = sum(1 for s in segs if s.ipa in VOWELS)
                for s in segs:
                    s.stress = 0
                    # short function words reduce toward schwa ("of" -> əv, "and" -> ənd), except /i ɪ eɪ aɪ oʊ/
                    if nv == 1 and s.ipa in _REDUCIBLE and clean not in _KEEP_FULL:
                        s.ipa = "ə"
            return segs
        if "-" in clean:   # hyphenated compounds
            return [s for p in clean.split("-") for s in self.word(p, reduce)]
        return letter_to_sound(clean)

    def letter_to_sound(self, w: str) -> list[Segment]:
        """Rule-based pronunciation (the fallback for words missing from the dictionary)."""
        return letter_to_sound(w)


def arpabet_to_segments(phones: str) -> list[Segment]:
    """ARPAbet phones -> segments; vowels carry the stress digit, consonants 0, unknown tokens are skipped."""
    out: list[Segment] = []
    for ph in _JS_SPACE_RE.split(phones):
        if not ph:
            continue
        if ph[-1] in "012":
            base, digit = ph[:-1], ph[-1]
        else:
            base, digit = ph, ""
        if not base or not all("A" <= c <= "Z" for c in base):
            continue
        stress = 1 if digit == "1" else 2 if digit == "2" else 0
        ipa = _ARPA.get(base)
        if ipa is None:
            continue
        if base == "AH" and stress == 0:
            ipa = "ə"
        if base == "ER" and stress == 0:
            ipa = "ɚ"
        out.append(Segment(ipa, stress=stress if ipa in VOWELS else 0))
    return out


def _apply_allophones(segs: list[Segment]) -> None:
    """American English allophones, in place: flapping and velarised coda /l/."""
    for i in range(1, len(segs) - 1):
        prev, nxt, s = segs[i - 1], segs[i + 1], segs[i]
        pv = prev.ipa in VOWELS or prev.ipa == "ɹ"
        nv = nxt.ipa in VOWELS
        next_stressed = nxt.stress is not None and nxt.stress != 0
        # /t d/ between a vowel and an unstressed vowel -> tap ("water", "got it"), but not at the start of a
        # word ("were disputing" keeps its /d/)
        if s.ipa in ("t", "d") and pv and nv and not next_stressed and prev.word_end is not True:
            s.ipa = "ɾ"
    for i, s in enumerate(segs):
        if s.ipa != "l":
            continue
        next_vowel = i + 1 < len(segs) and segs[i + 1].ipa in VOWELS
        # coda /l/ (before a consonant, a pause or a word end) is velarised
        if not next_vowel or s.word_end is True:
            s.ipa = "ɫ"


# ---------------------------------------------------------------- numbers
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _number_words(n: float) -> str:
    if n < 20.0:
        return _ONES[int(n)]
    if n < 100.0:
        r = math.fmod(n, 10.0)
        return _TENS[int(math.floor(n / 10.0))] + (" " + _ONES[int(r)] if r != 0.0 else "")
    if n < 1000.0:
        r = math.fmod(n, 100.0)
        return _ONES[int(math.floor(n / 100.0))] + " hundred" + (" " + _number_words(r) if r != 0.0 else "")
    if n < 1e6:
        r = math.fmod(n, 1000.0)
        return _number_words(math.floor(n / 1000.0)) + " thousand" + (" " + _number_words(r) if r != 0.0 else "")
    # digit by digit from the JS string form; its non-digits (an exponent, "Infinity") give empty words
    return " ".join(_ONES[ord(c) - 48] if "0" <= c <= "9" else "" for c in _js_integer_to_string(n))


def _js_integer_to_string(n: float) -> str:
    """``Number.prototype.toString`` of a non-negative integer-valued double: shortest round-trip digits,
    exponent form from 1e21 ("1e+21")."""
    if math.isinf(n):
        return "Infinity"
    r = repr(n)
    if n >= 1e21:
        return r                       # Python's repr already writes "1e+21" / "1.2345e+25"
    if "e" not in r:
        return r[:-2] if r.endswith(".0") else r
    mant, exp = r.split("e")
    digits = mant.replace(".", "")
    return digits + "0" * (int(exp) - (len(digits) - 1))


def _expand_numbers(t: str) -> str:
    # parseInt(digits, 10): the correctly rounded double ("9" * 400 -> inf)
    return _DIGITS.sub(lambda m: " " + _number_words(float(m.group(0))) + " ", t)


# ---------------------------------------------------------------- letter-to-sound fallback
# (pre, grapheme, post, phones), tried in this order at each position -- the order is part of the behaviour
# (it is mostly, not strictly, longest first). Context: '#' word boundary, 'V' vowel letter, 'C' any other
# character that is there, else the literal letter.
_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("", "tion", "", "ʃ ə n"), ("", "sion", "", "ʒ ə n"), ("", "ture", "", "tʃ ɚ"), ("", "ough", "#", "oʊ"),
    ("", "augh", "", "ɔ"), ("", "igh", "", "aɪ"), ("", "eigh", "", "eɪ"), ("", "tch", "", "tʃ"), ("", "dge", "", "dʒ"),
    ("", "sch", "", "s k"), ("", "ch", "", "tʃ"), ("", "sh", "", "ʃ"), ("", "th", "", "θ"), ("", "ph", "", "f"),
    ("", "wh", "", "w"), ("", "ck", "", "k"), ("", "ng", "#", "ŋ"), ("", "ng", "", "ŋ g"), ("", "qu", "", "k w"),
    ("#", "kn", "", "n"), ("#", "wr", "", "ɹ"), ("", "gh", "", ""),
    ("", "ee", "", "i"), ("", "ea", "", "i"), ("", "oo", "", "u"), ("", "ou", "", "aʊ"), ("", "ow", "#", "oʊ"),
    ("", "ow", "", "aʊ"), ("", "oi", "", "ɔɪ"), ("", "oy", "", "ɔɪ"), ("", "ai", "", "eɪ"), ("", "ay", "", "eɪ"),
    ("", "au", "", "ɔ"), ("", "aw", "", "ɔ"), ("", "ie", "#", "aɪ"), ("", "ie", "", "i"), ("", "ei", "", "eɪ"),
    ("", "ey", "#", "i"), ("", "oa", "", "oʊ"), ("", "ue", "#", "u"),
    ("", "er", "#", "ɚ"), ("", "er", "C", "ɝ"), ("", "ir", "", "ɝ"), ("", "ur", "", "ɝ"), ("", "or", "", "ɔ ɹ"),
    ("", "ar", "", "ɑ ɹ"), ("", "le", "#", "ə ɫ"), ("", "ed", "#", "d"), ("", "es", "#", "z"), ("C", "s", "#", "s"),
    ("V", "s", "#", "z"), ("", "y", "#", "i"), ("#", "y", "", "j"), ("C", "y", "", "aɪ"), ("", "e", "#", ""),
    ("", "a", "CV", "eɪ"), ("", "a", "", "æ"), ("", "e", "", "ɛ"), ("", "i", "CV", "aɪ"), ("", "i", "", "ɪ"),
    ("", "o", "CV", "oʊ"), ("", "o", "", "ɑ"), ("", "u", "CV", "u"), ("", "u", "", "ʌ"), ("", "y", "", "ɪ"),
    ("", "c", "e", "s"), ("", "c", "i", "s"), ("", "c", "y", "s"), ("", "c", "", "k"), ("", "g", "e", "dʒ"),
    ("", "g", "i", "dʒ"), ("", "g", "", "g"), ("", "x", "", "k s"), ("", "j", "", "dʒ"), ("", "q", "", "k"),
    ("", "b", "", "b"), ("", "d", "", "d"), ("", "f", "", "f"), ("", "h", "", "h"), ("", "k", "", "k"),
    ("", "l", "", "l"), ("", "m", "", "m"), ("", "n", "", "n"), ("", "p", "", "p"), ("", "r", "", "ɹ"),
    ("", "s", "", "s"), ("", "t", "", "t"), ("", "v", "", "v"), ("", "w", "", "w"), ("", "z", "", "z"),
)
_SPELL_VOWELS = "aeiouy"


def _utf16(word: str) -> str:
    """The word as UTF-16 code units, one Python character each: the rules index the word the way the JS
    string did (identical to ``word`` for text within the Basic Multilingual Plane)."""
    if word.isascii():
        return word
    b = word.encode("utf-16-le", "surrogatepass")
    return "".join(chr(b[i] | (b[i + 1] << 8)) for i in range(0, len(b), 2))


def _match_ctx(w: str, i: int, ctx: str, step: int) -> bool:
    pos = i
    for k in range(len(ctx)):
        c = ctx[k] if step == 1 else ctx[len(ctx) - 1 - k]
        ch = w[pos] if 0 <= pos < len(w) else None
        if c == "#":
            if ch is not None:
                return False
        elif c == "V":
            if ch is None or ch not in _SPELL_VOWELS:
                return False
        elif c == "C":
            if ch is None or ch in _SPELL_VOWELS:
                return False
        elif ch != c:
            return False
        pos += step
    return True


def letter_to_sound(word: str) -> list[Segment]:
    """Letter-to-sound rules (english.rs:409-449); the first vowel is stressed."""
    w = _utf16(word)
    phones: list[str] = []
    i = 0
    while i < len(w):
        for pre, g, post, ph in _RULES:
            if w.startswith(g, i) and _match_ctx(w, i - 1, pre, -1) and _match_ctx(w, i + len(g), post, 1):
                if ph:
                    phones.extend(ph.split(" "))
                i += len(g)
                break
        else:
            i += 1
    out: list[Segment] = []
    stressed = False
    for p in phones:
        v = p in VOWELS
        out.append(Segment(p, stress=1 if v and not stressed else 0))
        stressed = stressed or v
    return out
