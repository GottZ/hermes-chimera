"""Seam to the C1 text/planning functions used by the music and speech paths (C4-internal).

``planner_api()`` returns the functions compile.py/schema.py/speech.py and the Engine call: by default the real C1
modules (imported lazily on first use), or an override installed with ``set_planner_api`` (tests use a stub planner
that records its arguments, BUILD-ORDER §1 C4 fakes). No numpy import.
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
    prepare_text: Callable[[str], str]                                          # textprep.py (D§6.2)
    default_g2p: Callable[[], Any]                                              # EnglishG2P.default()


_lock = threading.Lock()
_override: PlannerAPI | None = None


def set_planner_api(api: PlannerAPI | None) -> PlannerAPI | None:
    """Install ``api`` (None restores the real C1 functions); returns the previous override."""
    global _override
    with _lock:
        prev, _override = _override, api
    return prev


def planner_api() -> PlannerAPI:
    """The override if one is installed, else the real C1 functions (lazy import)."""
    api = _override
    if api is not None:
        return api
    return _real_api()


def _real_api() -> PlannerAPI:
    raise NotImplementedError


__all__ = ["PlannerAPI", "planner_api", "set_planner_api"]
