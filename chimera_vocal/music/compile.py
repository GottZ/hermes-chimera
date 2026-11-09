"""Score -> one RenderJob per voice (INTERFACES §8.3), faithful to ``chimera render notes`` (render.rs:481-575).

Also ``render_job``: one job through a HumanVoice (worker or in-process). The voice is built by
``voice_factory()`` - the real ``dsp.voice.HumanVoice`` unless a factory was installed with ``set_voice_factory``
(tests: FakeVoice; the worker installs one from ``CHIMERA_WORKER_VOICE_FACTORY``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from ..core.body import BodyMix
from ..core.types import Gesture, Seeds, SourceParams

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .schema import ScoreReport

VoiceFactory = Callable[["RenderJob", Any, Any], Any]    # (job, kernel, geometry) -> HumanVoice-like object

PRE_ROLL_S = 0.30        # silent pre-roll of a late voice (P1-06, E1-A §3.5)
TAIL_S = 0.4             # render notes tail (render.rs:561)
DYN_PRESSURE = {"pp": 0.55, "p": 0.7, "mp": 0.85, "mf": 1.0, "f": 1.2, "ff": 1.4}
DYN_GAIN_DB = {"pp": -12.0, "p": -8.0, "mp": -4.0, "mf": 0.0, "f": 2.0, "ff": 4.0}
DYN_RAMP_S = 0.08
DYN_STEPS = 8


@dataclass(frozen=True)
class Command:
    at: int                          # relative to the job's first output sample
    op: Literal["source", "gate", "artic", "schedule", "release", "cancel"]
    data: dict


@dataclass(frozen=True)
class RenderJob:
    id: str
    body: BodyMix
    seeds: Seeds
    fs_out: int = 48000
    oversample: int = 2
    source: SourceParams = SourceParams()
    artic: dict | None = None
    gestures: list[Gesture] = ()
    commands: list[Command] = ()
    n_out: int = 0
    start_block: int = 0
    settle_f0: bool = False

    def to_json(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_json(cls, d: dict) -> "RenderJob":
        raise NotImplementedError


@dataclass(frozen=True)
class VoiceMix:
    offset: int                      # = 64·job.start_block
    gain_db: float
    pan: float
    gain_env: "np.ndarray | None"    # float32[n_out] or None


_factory_lock = threading.Lock()
_voice_factory: VoiceFactory | None = None


def set_voice_factory(factory: VoiceFactory | None) -> VoiceFactory | None:
    """Install a voice factory (None restores the real HumanVoice); returns the previous one."""
    global _voice_factory
    with _factory_lock:
        prev, _voice_factory = _voice_factory, factory
    return prev


def voice_factory() -> VoiceFactory:
    """The installed factory, else one that builds ``dsp.voice.HumanVoice`` from the job."""
    f = _voice_factory
    if f is not None:
        return f
    return _human_voice


def _human_voice(job: RenderJob, kernel: Any, geometry: Any) -> Any:
    raise NotImplementedError


def compile_score(report: "ScoreReport", *, g2p: Any, fs_out: int = 48000) -> "tuple[list[RenderJob], list[VoiceMix], int]":
    """-> (one job per voice, mixing data, n_mix)."""
    raise NotImplementedError


def render_job(job: RenderJob, kernel: Any, geometry: Any, *, out: "np.ndarray | None" = None,
               chunk: int = 48000, progress: Callable[[int], None] | None = None,
               cancel: Callable[[], bool] | None = None) -> "np.ndarray":
    """float32[n_out]; chunked (progress after each chunk, cancel checked between chunks); deterministic."""
    raise NotImplementedError
