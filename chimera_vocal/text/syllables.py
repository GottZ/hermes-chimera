"""Lyrics -> sung syllables (port of phonetics/syllables.rs).

Words come from the pronouncing dictionary unreduced (sung function words keep full vowels), are split into
syllables by the maximal-onset principle with English onset clusters, and the spelling is split to match for
display ("twin-kle"). Typed hyphens force syllable breaks; a lone "-" (or "~") holds the previous vowel onto
the next note (melisma), as in vocaloid editors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, Union

from ..core.body import BODY_MIX_MALE
from ..core.types import Segment, Syllable
from .english import JS_SPACE

if TYPE_CHECKING:
    from .english import EnglishG2P

WordSource = Union["EnglishG2P", Callable[[str, bool], list[Segment]]]

# legal English onsets (IPA)
_ONSETS = frozenset((
    "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "ɹ", "w", "j", "tʃ", "dʒ",
    "pl", "bl", "kl", "gl", "fl", "sl", "pɹ", "bɹ", "tɹ", "dɹ", "kɹ", "gɹ", "fɹ", "θɹ", "ʃɹ",
    "sp", "st", "sk", "sm", "sn", "sw", "sf", "tw", "kw", "dw", "gw", "θw",
    "pj", "bj", "kj", "gj", "mj", "nj", "fj", "vj", "hj", "lj",
    "spl", "spɹ", "stɹ", "skɹ", "skw", "skj", "spj",
))
_LYRIC_PUNCT = frozenset(".,;:!?\n")


@dataclass
class SyllableParts:
    """One syllable of a word before it gets its text (syllables.rs:24-33)."""

    onset: list[Segment]
    nucleus: Segment
    coda: list[Segment]

    def to_json(self) -> dict:
        return {"onset": [s.to_json() for s in self.onset], "nucleus": self.nucleus.to_json(),
                "coda": [s.to_json() for s in self.coda]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SyllableParts":
        return cls([Segment.from_json(s) for s in d["onset"]], Segment.from_json(d["nucleus"]),
                   [Segment.from_json(s) for s in d["coda"]])


def syllabify_segments(segs: Sequence[Segment]) -> list[SyllableParts]:
    """Split one word's segments into onset / nucleus / coda syllables (maximal onset)."""
    from ..plan.planner import is_vowel

    segs = list(segs)
    vi = [i for i, s in enumerate(segs) if is_vowel(s.ipa, BODY_MIX_MALE)]
    out: list[SyllableParts] = []
    start = 0
    for n, v in enumerate(vi):
        end = len(segs)                       # exclusive end of this syllable
        if n + 1 < len(vi):
            nxt = vi[n + 1]
            # consonants between this vowel and the next: the longest legal onset goes right
            between = segs[v + 1:nxt]
            k = 0
            for length in range(min(3, len(between)), 0, -1):
                if "".join(s.ipa for s in between[len(between) - length:]) in _ONSETS:
                    k = length
                    break
            end = nxt - k
        out.append(SyllableParts(segs[start:v], segs[v], segs[v + 1:end]))
        start = end
    return out


def _utf16_units(s: str) -> list[int]:
    b = s.encode("utf-16-le", "surrogatepass")
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b), 2)]


_V = frozenset(map(ord, "aeiouy"))
_V_CI = frozenset(map(ord, "aeiouyAEIOUY"))


def _eq_ci(c: int, lower: str) -> bool:
    return c == ord(lower) or c == ord(lower.upper())


def _slice(w: list[int], a: int, b: int) -> str:
    a, b = min(a, len(w)), min(b, len(w))
    if a >= b:
        return ""
    raw = b"".join(u.to_bytes(2, "little") for u in w[a:b])
    return raw.decode("utf-16-le", "replace")


def split_spelling(word: str, n: int) -> list[str]:
    """Split a written word into ``n`` chunks at its vowel groups (approximate hyphenation). The spelling is
    indexed in UTF-16 code units, as the JS string was."""
    if n <= 1:
        return [word]
    w = _utf16_units(word)
    lower = _utf16_units(word.lower())
    groups: list[list[int]] = []
    i = 0
    while i < len(lower):
        if lower[i] in _V:
            s = i
            while i < len(lower) and lower[i] in _V:
                i += 1
            groups.append([s, i])
        else:
            i += 1
    length = len(w)
    silent_e = length >= 2 and _eq_ci(w[-1], "e") and w[-2] not in _V_CI
    le = (length >= 3 and _eq_ci(w[-1], "e") and _eq_ci(w[-2], "l") and w[-3] not in _V_CI
          and not _eq_ci(w[-3], "l"))
    # a final silent e ("like") is not a syllable, but consonant + "le" is ("twin-kle", "lit-tle")
    if len(groups) > 1 and silent_e and not le and groups[-1][0] + 1 == length:
        groups.pop()
    if len(groups) < n:
        # fewer written vowels than sung syllables ("fire"): hold the word, then continuation marks
        return [word] + ["~"] * (n - 1)
    while len(groups) > n:
        last = groups.pop()
        groups[-1][1] = last[1]
    cuts = [0]
    for k in range(len(groups) - 1):
        a, b = groups[k][1], groups[k + 1][0]       # consonant letters a..b between vowel groups
        # a digraph stays together on the right ("any-where", "fa-ther"); otherwise one consonant goes right
        # and longer runs split in the middle ("lit-tle", "won-der")
        gap = b - a
        dig = gap >= 2 and any(_eq_ci(w[b - 2], p[0]) and _eq_ci(w[b - 1], p[1])
                               for p in ("th", "sh", "ch", "ph", "wh", "ck", "gh"))
        cuts.append(b - 2 if dig else a + (0 if gap <= 1 else gap // 2))
    cuts.append(length)
    return [_slice(w, cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def _is_word_char(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "'"


def _lyric_tokens(text: str) -> list[str]:
    """Words, free-standing melisma marks and phrase punctuation, like the JS regex
    ``/[A-Za-z'][A-Za-z'\\-]*|(?<=\\s|^)[~-](?=\\s|$)|[.,;:!?\\n]/g``."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if _is_word_char(c):
            start = i
            i += 1
            while i < n and (_is_word_char(text[i]) or text[i] == "-"):
                i += 1
            out.append(text[start:i])
            continue
        melisma = (c in "~-" and (i == 0 or text[i - 1] in JS_SPACE) and (i + 1 == n or text[i + 1] in JS_SPACE))
        if melisma or c in _LYRIC_PUNCT:
            out.append(c)
        i += 1
    return out


def lyrics_to_syllables(text: str, g2p: WordSource) -> list[Syllable]:
    """Lyrics -> syllables. ``g2p`` is an ``EnglishG2P`` (or any ``word(w, reduce)`` callable)."""
    word = g2p.word if hasattr(g2p, "word") else g2p
    out: list[Syllable] = []
    for tok in _lyric_tokens(text.replace("’", "'")):
        if len(tok) == 1 and tok in _LYRIC_PUNCT:
            if out:
                out[-1].phrase_end = True
            continue
        if tok in ("-", "~"):
            if out:
                prev = out[-1]
                out.append(Syllable("~", [], dataclasses.replace(prev.nucleus), [], True, prev.word_end, False))
            continue
        # typed hyphens split syllables explicitly ("ti-ger"); otherwise the dictionary decides
        parts = [p for p in tok.split("-") if p]
        syls: list[Syllable] = []
        for part in parts:
            sy = syllabify_segments(word(part.lower(), False))
            if len(parts) > 1:
                spell = [part] + ["~"] * max(0, len(sy) - 1)
            else:
                spell = split_spelling(part, len(sy))
            for k, s in enumerate(sy):
                syls.append(Syllable(spell[k] if k < len(spell) else "~", s.onset, s.nucleus, s.coda, None, False,
                                     False))
        if syls:
            syls[-1].word_end = True
        out.extend(syls)
    return out
