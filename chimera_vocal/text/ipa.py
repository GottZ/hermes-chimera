"""IPA input -> segments (port of phonetics/ipa.rs; NFC via unicodedata instead of the generated table).

S1 stub (INTERFACES §14): signatures only; the implementation follows in S2.
"""

from __future__ import annotations

from ..core.types import Segment


def tokenize_ipa(s: str) -> list[Segment]:
    """ipa.rs:31-96: NFC, aliases, tie bars; stress marks, word/pause separators, length marks, known symbols."""
    raise NotImplementedError
