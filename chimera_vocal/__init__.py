"""chimera_vocal: Python port of the chimera-vocal-synth engine (public API owned by C4, INTERFACES §5).

Importing this package never imports numpy, numba or anything from Hermes (INTERFACES §1 rule 1): numpy is
imported inside functions only, numba only inside render worker processes (``worker.py``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from .music.mixer import MixInfo
    from .music.schema import ScoreIssue, ScoreReport
    from .presets import VoicePreset

__version__ = "0.1.0"

__all__ = [
    "__version__", "EngineConfig", "Audio", "BackendStatus", "ScoreResult", "Engine", "get_engine", "reconfigure",
    "ChimeraError", "ScoreError", "TextError", "RenderCancelled", "BackendError",
]


# --- errors -------------------------------------------------------------------------------------------------

class ChimeraError(Exception):
    """Base class of every error the engine raises on purpose."""


class ScoreError(ChimeraError, ValueError):
    """Invalid score; ``errors`` lists the error-severity issues (path + message with the fix)."""

    def __init__(self, errors: "list[ScoreIssue]", message: str | None = None) -> None:
        self.errors = list(errors)
        super().__init__(message or "; ".join(f"{e.path}: {e.message}" for e in self.errors) or "invalid score")


class TextError(ChimeraError, ValueError):
    """Nothing speakable after text preparation (the TTS provider maps it to RuntimeError)."""


class RenderCancelled(ChimeraError):
    """The request's cancel event was set; queued jobs were dropped and running workers stopped."""


class BackendError(ChimeraError, RuntimeError):
    """numpy missing, or no worker could render and the in-process fallback is off / failed."""


# --- configuration and results ------------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    backend: Literal["auto", "numba", "numpy"] = "auto"
    workers: int = 0                 # 0 -> min(4, max(2, os.cpu_count() - 1)); one slot reserved for speech (§9.1)
    in_process_fallback: bool = True
    cache_dir: str | None = None     # numba cache base; None -> numba-style user cache dir + "/chimera"
    sample_rate: int = 48000         # fixed: 48 kHz out / 96 kHz internal; other values -> ValueError
    text_prep: bool = True
    max_voices: int = 8
    max_score_seconds: float = 300.0
    max_voice_seconds: float = 1200.0
    worker_idle_s: float = 60.0


@dataclass(frozen=True)
class Audio:
    samples: "np.ndarray"            # float32, (n,) mono or (n, 2) stereo
    sample_rate: int

    def duration_s(self) -> float:
        raise NotImplementedError

    def to_wav(self, path, *, bits: int = 16) -> str:
        raise NotImplementedError

    def pcm16_bytes(self) -> bytes:
        raise NotImplementedError


@dataclass(frozen=True)
class BackendStatus:
    numpy_ok: bool
    numba_installed: bool                   # find_spec only - the host never imports numba
    numba_version: str | None
    numba_state: Literal["absent", "cold", "warming", "ready", "failed"]
    numba_error: str | None
    cache_dir: str | None
    marker: dict | None
    rtf_estimate: dict[str, float]          # {"numba": …, "numpy": …} EMA of render_s/audio_s per job
    workers: int
    workers_alive: int
    restart_needed: bool                    # numpy/numba on disk after this process started (D§7, P1-19)


@dataclass(frozen=True)
class ScoreResult:
    path_wav: str
    path_mp3: str | None
    seconds: float
    sample_rate: int
    channels: int
    voices: int
    kernel: Literal["numba", "numpy"]       # ONE kernel per request (§9.1, P1-12)
    render_s: float
    warnings: list[str]
    mix: "MixInfo"


# --- engine -------------------------------------------------------------------------------------------------

class Engine:
    """Thread-safe; one per process via the runtime registry (§9.4). Construction is light (no numpy, no process)."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        raise NotImplementedError

    def speak(self, text: str, voice: "str | VoicePreset" = "male", *, speed: float | None = None,
              settle_first: bool = True, cancel: threading.Event | None = None) -> "tuple[Audio, str]":
        raise NotImplementedError

    def speak_pcm16(self, text: str, voice: "str | VoicePreset" = "male", *, speed: float | None = None,
                    cancel: threading.Event | None = None) -> Iterator[bytes]:
        raise NotImplementedError

    def check_score(self, score: "dict | str") -> "ScoreReport":
        raise NotImplementedError

    def render_score(self, score: "dict | str", out_dir: str, stem: str, *, want_mp3: bool,
                     progress: Callable[[float], None] | None = None,
                     cancel: threading.Event | None = None) -> ScoreResult:
        raise NotImplementedError

    def status(self) -> BackendStatus:
        raise NotImplementedError

    def warmup(self) -> None:
        raise NotImplementedError

    def hold_warm(self, on: bool) -> None:
        raise NotImplementedError

    def selftest(self) -> Any:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


def get_engine(config: EngineConfig | None = None) -> Engine:
    """The process-wide engine of this package version (runtime registry, §9.4); the first config wins."""
    raise NotImplementedError


def reconfigure(config: EngineConfig) -> Engine:
    """Shut the registered engine down and replace it with one built from ``config``."""
    raise NotImplementedError
