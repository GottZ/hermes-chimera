"""Seam to the text/planning functions used by the music and speech paths.

``planner_api()`` returns the functions compile.py/schema.py/speech.py and the Engine call: by default the real
``text``/``plan`` modules (imported lazily on first use), or an override installed with ``set_planner_api`` (the test
suite installs a stub planner that records its arguments). No numpy import.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.types import PhraseInput, Plan, PlanOptions, SongOptions, SungNote, Syllable


@dataclass(frozen=True)
class PlannerAPI:
    plan_phrases: Callable[[Sequence[PhraseInput], PlanOptions], Plan]          # planner.rs:519-530
    plan_song: Callable[[Sequence[SungNote], SongOptions], Plan]                 # sing.rs:101-177
    lyrics_to_syllables: Callable[[str, Any], list[Syllable]]                   # syllables.rs:180-220
    prepare_text: Callable[[str], str]                                          # textprep.py
    default_g2p: Callable[[], Any]                                              # EnglishG2P.default()


_lock = threading.Lock()
_override: PlannerAPI | None = None


def set_planner_api(api: PlannerAPI | None) -> PlannerAPI | None:
    """Install ``api`` (None restores the real functions); returns the previous override."""
    global _override
    with _lock:
        prev, _override = _override, api
    return prev


def planner_api() -> PlannerAPI:
    """The override if one is installed, else the real functions (lazy import)."""
    api = _override
    if api is not None:
        return api
    return _real_api()


_real: PlannerAPI | None = None


def _real_api() -> PlannerAPI:
    global _real
    if _real is None:
        from ..plan.planner import plan_phrases
        from ..plan.sing import plan_song
        from ..text.english import EnglishG2P
        from ..text.syllables import lyrics_to_syllables
        from ..text.textprep import prepare_text
        _real = PlannerAPI(plan_phrases=plan_phrases, plan_song=plan_song, lyrics_to_syllables=lyrics_to_syllables,
                           prepare_text=prepare_text, default_g2p=EnglishG2P.default)
    return _real


__all__ = ["PlannerAPI", "planner_api", "set_planner_api"]
