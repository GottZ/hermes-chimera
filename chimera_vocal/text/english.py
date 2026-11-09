"""English G2P: text -> phrases, word -> segments (port of phonetics/english.rs).

Pipeline of ``to_phrases`` (english.rs:132-177): typographic apostrophe, number expansion with the JS number
format for values >= 1e6, split at runs of ``.!?;:,``, word lookup (homographs, CMUdict, hyphen split,
letter-to-sound), function-word reduction, allophones (flap, dark l), intonation and pause per phrase.

S1 stub (INTERFACES §14): signatures only; the implementation follows in S2.
"""

from __future__ import annotations

from typing import Mapping

from ..core.types import Phrase, Segment


class EnglishG2P:
    """English letter-to-phoneme front end (english.rs:96-237).

    ``dictionary`` maps a lower-case word to its CMUdict phones ("HH AH0 L OW1"); None = no dictionary
    (homographs + letter-to-sound only, like ``EnglishG2P::new()``).
    """

    def __init__(self, dictionary: Mapping[str, str] | None = None) -> None:
        raise NotImplementedError

    @classmethod
    def default(cls) -> "EnglishG2P":
        """Lazy, thread-safe; loads ``chimera_vocal/data/cmudict.txt`` once per process."""
        raise NotImplementedError

    def load_dict(self, text: str) -> None:
        """``word<TAB>PHONES`` lines (english.rs:119-129: split at '\\n', first tab at index > 0, later line wins)."""
        raise NotImplementedError

    def to_phrases(self, text: str) -> list[Phrase]:
        raise NotImplementedError

    def word(self, w: str, reduce: bool) -> list[Segment]:
        raise NotImplementedError

    def letter_to_sound(self, w: str) -> list[Segment]:
        """Rule-based fallback for out-of-dictionary words (english.rs:349-449)."""
        raise NotImplementedError


def arpabet_to_segments(phones: str) -> list[Segment]:
    """ARPAbet phones -> segments with stress (english.rs:239-266)."""
    raise NotImplementedError
