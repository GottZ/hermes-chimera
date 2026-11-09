"""RenderScheduler: worker processes for both kernels, one kernel per request, a reserved speech slot,
cancellation by process end (INTERFACES §9.1). No numpy import at module level."""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .music.compile import RenderJob


@dataclass(frozen=True)
class Request:
    id: str
    priority: Literal["speech", "music"]
    kernel: Literal["numba", "numpy"]        # fixed at submit (P1-12)


class PcmSink(Protocol):
    def write(self, offset: int, pcm: "np.ndarray") -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class RenderStats:
    n_out: int
    seconds: float
    kernel: str
    worker_pid: int | None


def default_workers(cpu_count: int | None) -> int:
    """EngineConfig.workers = 0 -> min(4, max(2, cpu_count − 1)) (cpu_count None -> 2)."""
    raise NotImplementedError


class MemorySink:
    """PcmSink into a preallocated float32 array of ``n`` samples (speech windows)."""

    def __init__(self, n: int) -> None:
        raise NotImplementedError

    @property
    def data(self) -> "np.ndarray":
        raise NotImplementedError

    def write(self, offset: int, pcm: "np.ndarray") -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FileSink:
    """PcmSink into a raw little-endian float32 file of ``n`` samples (music voices, D§8.4)."""

    def __init__(self, path: str, n: int) -> None:
        raise NotImplementedError

    def write(self, offset: int, pcm: "np.ndarray") -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class RenderScheduler:
    def __init__(self, *, workers: int, cache_base: str | None, root: str, idle_s: float = 60.0,
                 in_process_fallback: bool = True, backend: Literal["auto", "numba", "numpy"] = "auto") -> None:
        # ``backend``: request C4-2 (choose_kernel needs EngineConfig.backend)
        raise NotImplementedError

    def choose_kernel(self) -> Literal["numba", "numpy"]:
        raise NotImplementedError

    def submit(self, req: Request, job: "RenderJob", *, sink: PcmSink,
               cancel: threading.Event) -> "concurrent.futures.Future[RenderStats]":
        raise NotImplementedError

    def hold_warm(self, on: bool) -> None:
        raise NotImplementedError

    def status(self) -> dict:
        """Keys (C4): workers, workers_alive, worker_pids, running, queued_speech, queued_music, numba_state,
        numba_error, rtf (EMA per kernel), held_warm."""
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


def _numba_installed() -> bool:
    """find_spec("numba") is not None - never imports numba (P1-11)."""
    raise NotImplementedError


def _marker_ok(cache_base: str | None) -> dict | None:
    """dsp.backend.marker_ok(cache_base) (a file read, no numba import); None if C3's backend is unavailable."""
    raise NotImplementedError
