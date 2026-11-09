"""Hermes binding of the chimera vocal synth (C5; INTERFACES §10-§15).

S1: signatures only (BUILD-ORDER §3); the implementation follows in S2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SKILL_DESCRIPTION = ("Write scores for chimera_render: voices, ranges, lyrics per note, canon/choir examples, "
                     "MEDIA delivery.")
SKILL_FRONTMATTER = {"name": "music", "description": SKILL_DESCRIPTION, "version": "0.1.0",
                     "metadata": {"hermes": {"tags": ["music", "singing", "audio"],
                                             "requires_tools": ["chimera_render"]}}}


def skill_path() -> Path:
    """``<plugin root>/skills/music/SKILL.md`` (INTERFACES §13)."""
    raise NotImplementedError


def register(ctx: Any) -> None:
    """INTERFACES §15: TTS provider, 4 tools (toolset ``chimera``), ``/chimera``, ``hermes chimera``, skill ``music``.
    No heavy imports, no I/O besides manifest-relative paths, no threads (runs in every Hermes process)."""
    raise NotImplementedError
