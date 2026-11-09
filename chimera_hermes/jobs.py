"""Render jobs that outlive one tool call.

Hermes gives a tool call a fixed time limit (420 s by default) that a plugin cannot extend, and drops the result of
a call that runs over. Every render therefore runs as a job in this process: the tool waits up to a budget below the
limit and otherwise returns ``status: running`` with a job id; ``chimera_job`` waits again. Job ids are 128-bit random
and bound to the conversation that started them.
"""

from __future__ import annotations

import dataclasses
import math
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass
class JobState:
    id: str                      # "chm-" + 32 hex digits (128 bit)
    owner: str | None            # conversation_id of the call that started it; None outside a session
    kind: Literal["render", "speak"]
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float
    created: float
    started: float | None
    finished: float | None
    result: dict | None
    error: str | None


def new_job_id() -> str:
    return "chm-" + secrets.token_hex(16)


def _owner(owner: Any) -> str | None:
    return owner if isinstance(owner, str) and owner else None


class JobRegistry:
    """Thread-safe; one per process; finished jobs are kept ``retention_s`` (2 h)."""

    def __init__(self, *, retention_s: float = 7200.0, clock: Callable[[], float] = time.time) -> None:
        self._retention = retention_s
        self._clock = clock
        self._cond = threading.Condition()
        self._jobs: dict[str, JobState] = {}
        self._eta: dict[str, float] = {}
        self._cancel: dict[str, threading.Event] = {}

    def _purge(self) -> None:
        now = self._clock()
        for jid in [j.id for j in self._jobs.values()
                    if j.finished is not None and now - j.finished > self._retention]:
            self._jobs.pop(jid, None)
            self._eta.pop(jid, None)
            self._cancel.pop(jid, None)

    def start(self, kind: str, owner: str | None, fn: Callable[[Callable[[float], None], threading.Event], dict], *,
              eta_s: float) -> JobState:
        """Run ``fn(progress, cancel)`` on a daemon thread; returns a snapshot of the new job."""
        job = JobState(id=new_job_id(), owner=_owner(owner), kind=kind, status="queued",  # type: ignore[arg-type]
                       progress=0.0, created=self._clock(), started=None, finished=None, result=None, error=None)
        cancel = threading.Event()
        with self._cond:
            self._purge()
            self._jobs[job.id] = job
            self._eta[job.id] = float(eta_s) if isinstance(eta_s, (int, float)) and math.isfinite(eta_s) else 0.0
            self._cancel[job.id] = cancel

        def progress(p: float) -> None:
            with self._cond:
                try:
                    job.progress = min(1.0, max(job.progress, float(p)))
                except (TypeError, ValueError):
                    pass

        def run() -> None:
            with self._cond:
                job.status = "running"
                job.started = self._clock()
            try:
                result = fn(progress, cancel)
                status, error = "done", None
            except Exception as e:  # noqa: BLE001 - every failure becomes the job's error
                result = None
                cancelled = cancel.is_set() or type(e).__name__ == "RenderCancelled"
                status, error = ("cancelled" if cancelled else "failed"), (str(e) or type(e).__name__)
            with self._cond:
                job.status = status  # type: ignore[assignment]
                job.result = result
                job.error = error
                if status == "done":
                    job.progress = 1.0
                job.finished = self._clock()
                self._cond.notify_all()

        threading.Thread(target=run, name=f"chimera-{kind}-{job.id[4:12]}", daemon=True).start()
        return self.snapshot(job.id)

    def snapshot(self, job_id: str) -> JobState:
        with self._cond:
            return dataclasses.replace(self._jobs[job_id])

    def eta_s(self, job_id: str) -> float:
        """Remaining seconds: from progress and elapsed time once started, else the start estimate."""
        with self._cond:
            job = self._jobs[job_id]
            est = self._eta.get(job_id, 0.0)
            if job.started is None or job.progress <= 0.0:
                return est
            elapsed = self._clock() - job.started
            return max(0.0, elapsed * (1.0 - job.progress) / job.progress)

    def wait(self, job_id: str, owner: str | None, timeout_s: float) -> JobState:
        """Wait until the job ends or ``timeout_s`` passes. An unknown id, or an owner that differs from the job's
        (both set), raises KeyError - reported to the model as "unknown job"."""
        owner = _owner(owner)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            self._purge()
            job = self._jobs.get(job_id) if isinstance(job_id, str) else None
            if job is None or (owner is not None and job.owner is not None and owner != job.owner):
                raise KeyError(job_id)
            while job.finished is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return dataclasses.replace(job)

    def cancel_all(self) -> None:
        with self._cond:
            for ev in self._cancel.values():
                ev.set()


def wait_budget_s(requested: float | None) -> float:
    """How long one tool call may wait for its job: unbounded limit -> min(requested or 1800, 1800) s; a limit under
    60 s -> 0 (return the job id at once); otherwise max(1, min(requested or ∞, limit − 45, 375)) s. The 45 s keep a
    margin for the model's own turn."""
    from .settings import tool_deadline_s

    d = tool_deadline_s()
    if d is None:
        return float(min(requested or 1800.0, 1800.0))
    if d < 60.0:
        return 0.0
    return float(max(1.0, min(requested or math.inf, d - 45.0, 375.0)))
