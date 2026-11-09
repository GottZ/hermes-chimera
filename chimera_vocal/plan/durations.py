"""Segment durations by the Klatt (1979) / MITalk rules (Allen, Hunnicutt & Klatt 1987, ch. 9; port of
phonetics/durations.rs):

    DUR = MIN + (INH - MIN) * PRCNT

with PRCNT the product of context factors: phrase-final lengthening, non-final shortening, non-word-final and
polysyllabic shortening, unstressed shortening (MIN halved), the vowel length cue of the following consonant
(bat/bad), cluster shortening, and aspiration. Inherent/minimum durations (ms) follow Klatt's table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Union

from ..core.body import BodyMix
from ..core.types import PlanOptions, Segment
from .inventory import Manner, Voicing, phoneme

# Klatt's inherent / minimum durations (ms)
_TABLE: dict[str, tuple[float, float]] = {
    # vowels
    "i": (155.0, 55.0), "ɪ": (135.0, 40.0), "e": (190.0, 70.0), "eɪ": (190.0, 70.0), "ɛ": (150.0, 70.0),
    "æ": (230.0, 80.0), "a": (230.0, 80.0), "ɑ": (240.0, 100.0), "ɒ": (240.0, 100.0), "ɔ": (240.0, 100.0),
    "o": (220.0, 80.0), "oʊ": (220.0, 80.0), "ʊ": (160.0, 60.0), "u": (210.0, 70.0), "ʌ": (140.0, 60.0),
    "ə": (120.0, 60.0), "ɚ": (180.0, 80.0), "ɝ": (180.0, 80.0), "aɪ": (250.0, 150.0), "aʊ": (240.0, 100.0),
    "ɔɪ": (280.0, 150.0), "ɨ": (110.0, 40.0),
    # consonants
    "p": (85.0, 50.0), "t": (75.0, 50.0), "k": (80.0, 60.0), "b": (85.0, 60.0), "d": (75.0, 50.0),
    "g": (80.0, 60.0), "ʔ": (40.0, 30.0), "tʃ": (70.0, 50.0), "dʒ": (50.0, 30.0), "f": (100.0, 80.0),
    "v": (60.0, 40.0), "θ": (90.0, 60.0), "ð": (50.0, 30.0), "s": (105.0, 60.0), "z": (75.0, 40.0),
    "ʃ": (105.0, 80.0), "ʒ": (70.0, 40.0), "h": (80.0, 20.0), "m": (70.0, 60.0), "n": (60.0, 50.0),
    "ŋ": (95.0, 45.0), "l": (80.0, 40.0), "ɫ": (80.0, 40.0), "ɹ": (80.0, 30.0), "w": (80.0, 60.0),
    "j": (80.0, 40.0), "ɾ": (20.0, 20.0),
}


@dataclass(frozen=True)
class SegInfo:
    """What the duration rules need to know about a segment (serde camelCase like the template)."""

    vowel: bool
    manner: Manner | None = None
    voiced: bool | None = None
    dur: float | None = None          # fallback inherent duration (s) for phonemes outside the table

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SegInfo":
        m = d.get("manner")
        return cls(bool(d["vowel"]), Manner(m) if m is not None else None, d.get("voiced"), d.get("dur"))

    def to_json(self) -> dict:
        out: dict[str, Any] = {"vowel": self.vowel}
        if self.manner is not None:
            out["manner"] = self.manner.value
        if self.voiced is not None:
            out["voiced"] = self.voiced
        if self.dur is not None:
            out["dur"] = self.dur
        return out


InfoSource = Union[PlanOptions, Callable[[Segment], SegInfo]]


def segment_info(seg: Segment, body: BodyMix) -> SegInfo:
    """The planner's view of a segment: vowel, or the inventory's manner/voicing/duration."""
    from .planner import is_vowel

    if is_vowel(seg.ipa, body):
        return SegInfo(True)
    d = phoneme(seg.ipa, body)
    if d is None:
        return SegInfo(False, None, False, None)
    return SegInfo(False, d.manner, d.voicing is Voicing.VOICED, d.dur)


def klatt_durations(segs: Sequence[Segment], opts: InfoSource) -> list[float]:
    """Durations (s) for one phrase of segments; explicit ``dur`` values are kept.

    ``opts`` is either ``PlanOptions`` (segment info from the inventory for ``opts.body``, as the planner uses
    it) or a callback ``Segment -> SegInfo``, called once per segment, in order, with the caller's objects.
    """
    if isinstance(opts, PlanOptions):
        body = BodyMix.parse(opts.body)
        inf = [segment_info(s, body) for s in segs]
    else:
        inf = [opts(s) for s in segs]
    n = len(segs)
    # words: index ranges split after wordEnd
    word = [0] * n
    word_start: list[int] = []
    for i in range(n):
        if i == 0 or segs[i - 1].word_end is True:
            word_start.append(i)
        word[i] = len(word_start) - 1
    vowels_in_word: dict[int, int] = {}
    for i in range(n):
        if inf[i].vowel:
            vowels_in_word[word[i]] = vowels_in_word.get(word[i], 0) + 1
    last_vowel = -1
    for i in range(n):
        if inf[i].vowel:
            last_vowel = i

    out: list[float] = []
    for i, s in enumerate(segs):
        if s.dur is not None:
            out.append(s.dur)
            continue
        fallback = inf[i].dur if inf[i].dur is not None else 0.08
        inh, mn = _TABLE.get(s.ipa, (1000.0 * fallback, 600.0 * fallback))
        p = 1.0
        if inf[i].vowel:
            stress = s.stress if s.stress is not None else 0
            final_syl = i == last_vowel
            word_final_syl = True
            k = i + 1
            while k < n and word[k] == word[i]:
                if inf[k].vowel:
                    word_final_syl = False
                k += 1
            p *= 1.4 if final_syl else 0.6      # clause-final lengthening / non-final shortening
            if not word_final_syl:
                p *= 0.85                       # non-word-final shortening
            if vowels_in_word.get(word[i], 1) > 1:
                p *= 0.8                        # polysyllabic shortening
            if stress == 0:
                mn *= 0.5                       # unstressed: minimum halved, shortened
                p *= 0.7
            # the following consonant (same word): voiced fricatives lengthen, voiceless stops shorten
            if i + 1 < n and word[i + 1] == word[i] and not inf[i + 1].vowel:
                nx = inf[i + 1]
                voiced = nx.voiced is True
                f = 1.0
                if nx.manner is Manner.FRICATIVE:
                    f = 1.6 if voiced else 1.0
                elif nx.manner is Manner.STOP or nx.manner is Manner.AFFRICATE:
                    f = 1.2 if voiced else 0.7
                elif nx.manner is Manner.NASAL:
                    f = 0.85
                p *= f if final_syl else 0.7 + 0.3 * f
            if i + 1 < n and inf[i + 1].vowel:  # vowel next to vowel
                p *= 1.2
            if i > 0 and inf[i - 1].vowel:
                p *= 0.7
            d = mn + (inh - mn) * p
            # aspiration after a voiceless stop eats into a stressed vowel's onset: give it back
            if stress > 0 and i > 0 and inf[i - 1].manner is Manner.STOP and inf[i - 1].voiced is not True:
                d += 20.0
            out.append(d / 1000.0)
            continue
        # consonants
        word_initial = i == word_start[word[i]]
        if i > last_vowel and inf[i].manner in (Manner.NASAL, Manner.LATERAL, Manner.APPROXIMANT):
            p *= 1.4
        if not word_initial:
            p *= 0.85
        prev_c = i > 0 and not inf[i - 1].vowel
        next_c = i + 1 < n and not inf[i + 1].vowel
        if prev_c and next_c:
            p *= 0.5
        elif prev_c or next_c:
            p *= 0.7
        out.append((mn + (inh - mn) * p) / 1000.0)
    return out
