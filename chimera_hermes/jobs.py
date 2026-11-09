"""Render jobs that outlive one tool call.

Hermes gives a tool call a fixed time limit (420 s by default) that a plugin cannot extend, and drops the result of
a call that runs over. Every render therefore runs as a job in this process: the tool waits up to a budget below the
limit and otherwise returns ``status: running`` with a job id; ``chimera_job`` waits again (or cancels). Job ids are
128-bit random and bound to the conversation/session that started them (``sessions.py``). The job thread runs in a
copy of the caller's context, so Hermes' per-profile state (the active HERMES_HOME, kept in context variables) is the
caller's.
"""

from __future__ import annotations

import contextvars
import dataclasses
import math
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal


@dataclass
class JobState:
    id: str                      # "chm-" + 32 hex digits (128 bit)
    owner: Any                   # caller ids of the call that started it (str or set of str); None outside a session
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


def _owner(owner: Any) -> frozenset | None:
    """Caller identity as a set of ids: a string is one id, an iterable several; empty -> None."""
    if isinstance(owner, str):
        return frozenset({owner}) if owner else None
    if owner is None:
        return None
    try:
        ids = frozenset(x for x in owner if isinstance(x, str) and x)
    except TypeError:
        return None
    return ids or None


def _foreign(job_owner: Any, owner: Any) -> bool:
    """True when both callers are known and share no id."""
    a, b = _owner(job_owner), _owner(owner)
    return a is not None and b is not None and not (a & b)


Related = Callable[[], Iterable[str]]


def _still_foreign(job_owner: Any, owner: frozenset | None, related: Related | None) -> bool:
    """``_foreign``, then once more with the caller's related ids (``related()`` runs only on a mismatch)."""
    if not _foreign(job_owner, owner):
        return False
    if related is None:
        return True
    try:
        extra = _owner(related())
    except Exception:  # noqa: BLE001 - no related ids known
        return True
    return extra is None or _foreign(job_owner, (owner or frozenset()) | extra)


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
        ids = _owner(owner)
        stored = None if ids is None else (owner if isinstance(owner, str) else ids)
        job = JobState(id=new_job_id(), owner=stored, kind=kind, status="queued",  # type: ignore[arg-type]
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

        caller = contextvars.copy_context()
        threading.Thread(target=caller.run, args=(run,), name=f"chimera-{kind}-{job.id[4:12]}", daemon=True).start()
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

    def _owned(self, job_id: str, owner: Any, related: Related | None) -> JobState:
        """The job ``job_id`` if ``owner`` may see it; KeyError otherwise. ``related()`` (other ids of the caller,
        e.g. the sessions its conversation passed through) is asked outside the lock and only on a mismatch."""
        owner = _owner(owner)
        with self._cond:
            self._purge()
            job = self._jobs.get(job_id) if isinstance(job_id, str) else None
            if job is None:
                raise KeyError(job_id)
            job_owner = job.owner
        if _still_foreign(job_owner, owner, related):
            raise KeyError(job_id)
        return job

    def wait(self, job_id: str, owner: Any, timeout_s: float, *, related: Related | None = None) -> JobState:
        """Wait until the job ends or ``timeout_s`` passes. An unknown id, or an owner that differs from the job's
        (both set, also after ``related``), raises KeyError - reported to the model as "unknown job"."""
        job = self._owned(job_id, owner, related)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            while job.finished is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return dataclasses.replace(job)

    def cancel(self, job_id: str, owner: Any, *, related: Related | None = None) -> None:
        """Set the cancel event of one job (KeyError like ``wait`` for an unknown or foreign job); the engine drops
        its queued work and ends its running workers."""
        self._owned(job_id, owner, related)
        with self._cond:
            ev = self._cancel.get(job_id)
        if ev is not None:
            ev.set()

    def has_unfinished(self) -> bool:
        with self._cond:
            return any(j.finished is None for j in self._jobs.values())

    def cancel_owned(self, ids: Iterable[str]) -> list[str]:
        """Cancel every unfinished job whose owner shares an id with ``ids`` (jobs without an owner never match);
        returns their ids."""
        wanted = _owner(frozenset(ids))
        if wanted is None:
            return []
        out = []
        with self._cond:
            for j in self._jobs.values():
                mine = _owner(j.owner)
                if j.finished is None and mine is not None and mine & wanted:
                    ev = self._cancel.get(j.id)
                    if ev is not None:
                        ev.set()
                    out.append(j.id)
        return out

    def cancel_all(self) -> None:
        with self._cond:
            for ev in self._cancel.values():
                ev.set()


def wait_budget_s(requested: float | None) -> float:
    """How long one tool call may wait for its job: ``requested`` 0 -> 0 (return the job id at once); unbounded
    limit -> min(requested, 1800) s (1800 when not given); a limit under 60 s -> 0; otherwise
    max(1, min(requested, limit − 45, 375)) s. The 45 s keep a margin for the model's own turn."""
    from .settings import tool_deadline_s

    if requested is not None and requested <= 0:
        return 0.0
    d = tool_deadline_s()
    if d is None:
        return float(min(requested if requested is not None else 1800.0, 1800.0))
    if d < 60.0:
        return 0.0
    return float(max(1.0, min(requested if requested is not None else math.inf, d - 45.0, 375.0)))
