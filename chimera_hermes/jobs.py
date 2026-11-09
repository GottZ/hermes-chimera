"""Render jobs that outlive one tool call (INTERFACES §11.6; DESIGN §8.6; P1-18, P1-22).

S1: signatures only (JobState is interface data and complete).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class JobState:
    id: str                      # "chm-" + secrets.token_hex(16) (128 bit, P1-22)
    owner: str | None            # conversation_id from the handler kwargs (index.md:886-899), None outside sessions
    kind: Literal["render", "speak"]
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float
    created: float
    started: float | None
    finished: float | None
    result: dict | None
    error: str | None


def new_job_id() -> str:
    """``"chm-" + secrets.token_hex(16)``."""
    raise NotImplementedError


class JobRegistry:
    """Process-wide (runtime registry), thread-safe; keeps finished jobs ``retention_s`` (2 h)."""

    def __init__(self, *, retention_s: float = 7200.0, clock: Callable[[], float] = time.time) -> None:
        raise NotImplementedError

    def start(self, kind: str, owner: str | None, fn: Callable[[Callable[[float], None], threading.Event], dict], *,
              eta_s: float) -> JobState:
        """Run ``fn(progress, cancel)`` in a daemon thread; returns a snapshot of the new job."""
        raise NotImplementedError

    def snapshot(self, job_id: str) -> JobState:
        raise NotImplementedError

    def eta_s(self, job_id: str) -> float:
        """Remaining seconds (from progress and elapsed time, else the start estimate)."""
        raise NotImplementedError

    def wait(self, job_id: str, owner: str | None, timeout_s: float) -> JobState:
        """Unknown id or owner mismatch (both non-None and different) -> KeyError (reported as "unknown job")."""
        raise NotImplementedError

    def cancel_all(self) -> None:
        raise NotImplementedError


def wait_budget_s(requested: float | None) -> float:
    """d = tool_deadline_s(); d is None -> min(requested or 1800, 1800); d < 60 -> 0; else
    max(1.0, min(requested or ∞, d − 45, 375)) (P1-18)."""
    raise NotImplementedError
