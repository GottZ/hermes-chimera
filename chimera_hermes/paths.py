"""Storage locations and clean-up (INTERFACES §11.3).

S1: signatures only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

SLUG_MAX = 48


def slug(raw: str | None, default: str) -> str:
    """NFKD -> ASCII, lower, every run of non [a-z0-9] -> "-", strip "-", cut to 48; empty or a Windows device name
    (con, prn, aux, nul, com1-9, lpt1-9) -> default; never contains CR/LF/space (P1-22)."""
    raise NotImplementedError


def hermes_home() -> Path:
    """``hermes_constants.get_hermes_home()`` of the current profile."""
    raise NotImplementedError


def audio_dir(create: bool = True) -> Path:
    """``<get_hermes_dir("cache/audio", "audio_cache")>/chimera``, resolved per call."""
    raise NotImplementedError


def cache_dir() -> Path:
    """``<HERMES_HOME>/cache/chimera``."""
    raise NotImplementedError


def numba_cache_dir() -> Path:
    """``<HERMES_HOME>/cache/chimera/numba`` (EngineConfig.cache_dir)."""
    raise NotImplementedError


def tmp_dir() -> Path:
    """``<HERMES_HOME>/cache/chimera/tmp`` (mixer temp files)."""
    raise NotImplementedError


def new_output_stem(raw: str | None, default: str, *, now: float | None = None) -> tuple[Path, str]:
    """(audio_dir(), "<YYYYMMDD-HHMMSS>-<slug>[-n]"); ``<stem>.wav`` is reserved (created empty, O_EXCL)."""
    raise NotImplementedError


def cleanup(keep_days: float, *, now: float | None = None, keep_cache: Iterable[str] = ()) -> int:
    """Delete own rendered files older than keep_days (0 = keep), tmp entries older than one day and numba cache
    folders not in keep_cache older than keep_days; never raises; returns the number removed."""
    raise NotImplementedError
