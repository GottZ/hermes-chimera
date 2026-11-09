"""Render-time estimate (INTERFACES §8.5). No numpy import."""

from __future__ import annotations

from typing import Mapping, Sequence

DEFAULT_RTF = {"numba": 0.45, "numpy": 5.0}   # per job on one core; replaced by the measured EMA
DEFAULT_COMPILE_S = 30.0                       # warm-up worker compile; S3 sets the measured value


def estimate_seconds(job_seconds: Sequence[float], *, rtf: float, parallelism: int,
                     compile_s: float = 0.0, overhead_s: float = 0.5) -> float:
    """Longest-processing-time assignment of job_seconds·rtf to ``parallelism`` slots + compile_s + overhead_s."""
    raise NotImplementedError


def music_slots(workers: int) -> int:
    """Music render slots for ``workers`` worker processes: workers − 1, at least 1 (§9.1)."""
    raise NotImplementedError


def estimate_kernels(job_seconds: Sequence[float], *, workers: int, rtf: Mapping[str, float] | None = None,
                     numba_state: str = "absent", compile_s: float = DEFAULT_COMPILE_S,
                     overhead_s: float = 0.5) -> dict[str, float]:
    """{"numba": …, "numpy": …} for a score on ``workers``; the compile surcharge only while numba_state != "ready"."""
    raise NotImplementedError
