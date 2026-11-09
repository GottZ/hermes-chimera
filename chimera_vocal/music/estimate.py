"""Render-time estimate. No numpy import."""

from __future__ import annotations

from typing import Mapping, Sequence

DEFAULT_RTF = {"numba": 0.45, "numpy": 5.0}   # per job on one core; replaced by the measured EMA
DEFAULT_COMPILE_S = 30.0                       # warm-up worker compile (upper estimate until measured)


def estimate_seconds(job_seconds: Sequence[float], *, rtf: float, parallelism: int,
                     compile_s: float = 0.0, overhead_s: float = 0.5) -> float:
    """Longest-processing-time assignment of job_seconds·rtf to ``parallelism`` slots + compile_s + overhead_s."""
    slots = [0.0] * max(1, int(parallelism))
    for cost in sorted((float(s) * rtf for s in job_seconds), reverse=True):
        i = min(range(len(slots)), key=slots.__getitem__)
        slots[i] += cost
    return max(slots) + compile_s + overhead_s


def music_slots(workers: int) -> int:
    """Music render slots for ``workers`` worker processes: workers − 1 (one stays free for speech), at least 1."""
    return max(1, int(workers) - 1)


def estimate_kernels(job_seconds: Sequence[float], *, workers: int, rtf: Mapping[str, float] | None = None,
                     numba_state: str = "absent", compile_s: float = DEFAULT_COMPILE_S,
                     overhead_s: float = 0.5) -> dict[str, float]:
    """{"numba": …, "numpy": …} for a score on ``workers``; the compile surcharge only while numba_state != "ready"."""
    r = dict(DEFAULT_RTF)
    if rtf:
        r.update(rtf)
    par = music_slots(workers)
    return {
        "numba": estimate_seconds(job_seconds, rtf=r["numba"], parallelism=par,
                                  compile_s=0.0 if numba_state == "ready" else compile_s, overhead_s=overhead_s),
        "numpy": estimate_seconds(job_seconds, rtf=r["numpy"], parallelism=par, overhead_s=overhead_s),
    }
