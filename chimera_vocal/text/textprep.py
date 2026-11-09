"""Plugin-own text preparation before ``EnglishG2P.to_phrases`` (DESIGN §6.2; not part of the template).

Rules, in this order: (1) NFKC + typographic characters to ASCII, (2) abbreviations, (3) numbers,
(4) Latin letters with diacritics to their base letters, (5) remaining characters outside
``[A-Za-z0-9 '.,;:!?-]`` to spaces, brackets to comma pauses, whitespace/punctuation cleanup.
Returns "" when nothing speakable (no letter or digit) remains; the caller raises ``TextError``
(design/requests/C1-4.md). Never raises for a ``str``.

S1 stub (INTERFACES §14): signature only; the implementation follows in S2.
"""

from __future__ import annotations


def prepare_text(text: str) -> str:
    raise NotImplementedError
