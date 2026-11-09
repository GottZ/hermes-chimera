"""Lyrics -> sung syllables (port of phonetics/syllables.rs).

S1 stub (INTERFACES §14): signatures only; the implementation follows in S2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..core.types import Segment, Syllable

if TYPE_CHECKING:
    from .english import EnglishG2P


def lyrics_to_syllables(text: str, g2p: "EnglishG2P") -> list[Syllable]:
    """syllables.rs:180-220 (words unreduced, melisma marks, phrase punctuation incl. line breaks)."""
    raise NotImplementedError


def syllabify_segments(segs: Sequence[Segment]) -> list[Syllable]:
    """Maximal-onset split of one word (syllables.rs:33-60).

    The template returns ``Vec<SyllableParts>`` (onset/nucleus/coda only); see request
    design/requests/C1-2.md. Callers and tests use only ``.onset``, ``.nucleus``, ``.coda``.
    """
    raise NotImplementedError


def split_spelling(word: str, n: int) -> list[str]:
    """Split a written word into ``n`` chunks at its vowel groups (syllables.rs:78-138)."""
    raise NotImplementedError
