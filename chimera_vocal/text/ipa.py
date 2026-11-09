"""IPA input -> segments (port of phonetics/ipa.rs).

Stress marks ˈ ˌ (or '), length ː (or :), tie bars, affricates and diphthongs as one token, word gaps
(space . | ,) with a pause segment after , and |. Unicode NFC comes from ``unicodedata`` (the template ships a
table generated from its JS engine; both agree on IPA input). Unknown symbols are dropped silently; combining
diacritics after a symbol are ignored.
"""

from __future__ import annotations

import unicodedata

from ..core.types import Segment
from ..plan.inventory import SYMBOLS

VOWEL_CHARS = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɝɚ"
_MULTI = ("tʃ", "dʒ", "ts", "dz", "pf", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ")
_ALIASES = (("t͡ʃ", "tʃ"), ("d͡ʒ", "dʒ"), ("t͡s", "ts"), ("d͡z", "dz"), ("ɡ", "g"), ("ʧ", "tʃ"), ("ʤ", "dʒ"),
            ("r̥", "r"), ("ɐ̯", "ɐ"))
_CONSONANTS = frozenset(SYMBOLS)
_KNOWN = _CONSONANTS | frozenset(VOWEL_CHARS) | frozenset(_MULTI)


def tokenize_ipa(s: str) -> list[Segment]:
    """IPA string -> segments (ipa.rs:31-96)."""
    s = unicodedata.normalize("NFC", s)
    for a, b in _ALIASES:
        s = s.replace(a, b)
    s = s.replace("͜", "").replace("͡", "")    # tie bars
    out: list[Segment] = []
    stress = 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "ˈ" or ch == "'":
            stress = 1
            i += 1
            continue
        if ch == "ˌ":
            stress = 2
            i += 1
            continue
        if ch in " .|,":
            if out:
                out[-1].word_end = True
            if ch in ",|":
                out.append(Segment("_", dur=0.16))
            i += 1
            continue
        if ch == "ː" or ch == ":":
            if out:
                last = out[-1]
                base = last.dur
                if base is None:
                    base = 0.15 if last.ipa[:1] and last.ipa[0] in VOWEL_CHARS else 0.09
                last.dur = base * 1.7
            i += 1
            continue
        tok = ""
        for m in _MULTI:
            if s.startswith(m, i):
                tok = m
        if not tok:
            tok = ch
        i += len(tok)
        while i < n and "̀" <= s[i] <= "ͯ":    # combining diacritics we do not model
            i += 1
        if tok not in _KNOWN:
            continue
        is_v = tok[0] in VOWEL_CHARS and tok not in _CONSONANTS
        out.append(Segment(tok, stress=stress if is_v else 0))
        if is_v:
            stress = 0
    return out
