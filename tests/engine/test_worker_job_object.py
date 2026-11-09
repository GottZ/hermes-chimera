"""Job objects for render workers behind a launcher (Windows code path, driven on any OS through a kernel32 fake).

On Windows a venv's python.exe is a launcher: the process the scheduler starts is not the interpreter that renders,
which is the launcher's child. The scheduler assigns the started process to a kill-on-close job object only after
Popen returns, so the launcher may already have started the interpreter outside that job. The worker therefore
reports its interpreter pid in its first frame (the hello), and the host puts that interpreter into a job as well:
the worker's job, or - when that job refuses it because the launcher already put it into a job of its own - a new
kill-on-close job held with the worker.

Here a small shell script plays the launcher (it starts the interpreter as its child and waits for it), the
platform switch ``scheduler._JOB_OBJECTS`` is turned on, and ``scheduler._kernel32`` returns a fake that records the
calls. Tests K-S5b.1 ... K-S5b.8.
"""

from __future__ import annotations

import itertools
import logging
import os
import stat
import sys
import threading

import pytest

import chimera_vocal.scheduler as sched_mod
from chimera_vocal.scheduler import MemorySink, Request
from chimera_vocal.worker import read_frame, write_frame
from support.fakes import FACTORY_SPEC, fake_job, pid_state, spawn_worker, wait_until

WORKER_JOB = 0x77                      # the job handle the fake hands out for the started process
OPEN_ACCESS = 0x0100 | 0x0001 | 0x1000  # PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION
KILL_ON_JOB_CLOSE = 0x2000

posix_only = pytest.mark.skipif(os.name == "nt", reason="the stand-in launcher is a POSIX shell script")


class FakeKernel32:
    """Stand-in for the kernel32 calls of the job-object code: records every call and hands out handles."""

    def __init__(self, *, open_ok: bool = True, in_job: bool = False, refuse_worker_job: bool = False,
                 refuse_own_job: bool = False, raise_on_open: bool = False) -> None:
        self.calls: list[tuple] = []
        self.open_ok, self.in_job = open_ok, in_job
        self.refuse_worker_job, self.refuse_own_job = refuse_worker_job, refuse_own_job
        self.raise_on_open = raise_on_open
        self.processes: dict[int, int] = {}      # process handle -> pid
        self.jobs: list[int] = []                # job handles created through CreateJobObjectW
        self.flags: dict[int, int] = {}          # job handle -> LimitFlags set on it
        self._next = itertools.count(0x500, 4)
        self._lock = threading.Lock()

    def _rec(self, *call) -> None:
        with self._lock:
            self.calls.append(call)

    def OpenProcess(self, access, inherit, pid):
        self._rec("OpenProcess", access, bool(inherit), pid)
        if self.raise_on_open:
            raise OSError("fake OpenProcess failure")
        if not self.open_ok:
            return None
        h = next(self._next)
        self.processes[h] = pid
        return h

    def IsProcessInJob(self, h, job, result):
        self._rec("IsProcessInJob", h, job)
        result.contents.value = 1 if self.in_job else 0
        return 1

    def AssignProcessToJobObject(self, job, h):
        self._rec("AssignProcessToJobObject", job, h)
        if job == WORKER_JOB:
            return 0 if self.refuse_worker_job else 1
        return 0 if self.refuse_own_job else 1

    def CreateJobObjectW(self, attrs, name):
        h = next(self._next)
        self.jobs.append(h)
        self._rec("CreateJobObjectW", h)
        return h

    def SetInformationJobObject(self, job, info_class, info, size):
        self.flags[job] = info._obj.BasicLimitInformation.LimitFlags
        self._rec("SetInformationJobObject", job, info_class)
        return 1

    def CloseHandle(self, h):
        self._rec("CloseHandle", h)
        return 1

    def last_error(self) -> int:
        return 5                                 # ERROR_ACCESS_DENIED

    def named(self, name: str) -> list[tuple]:
        with self._lock:
            return [c for c in self.calls if c[0] == name]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """Workers start through a shell script that runs the interpreter as its child (like a Windows venv launcher)."""
    script = tmp_path / "launcher.sh"
    script.write_text(f'#!/bin/sh\n"{sys.executable}" "$@"\nexit $?\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setattr(sys, "executable", str(script))
    return script


@pytest.fixture
def jobs(monkeypatch):
    """Turn the job-object code path on with a kernel32 fake; returns a configurator (keyword args of
    FakeKernel32) that yields the fake and the list of started-process pids put into WORKER_JOB."""
    started: list[int] = []
    state: dict = {"fake": FakeKernel32(), "job": WORKER_JOB}

    def assign(proc):
        started.append(proc.pid)
        return state["job"]

    monkeypatch.setattr(sched_mod, "_JOB_OBJECTS", True, raising=False)
    monkeypatch.setattr(sched_mod, "_assign_job_object", assign)
    monkeypatch.setattr(sched_mod, "_kernel32", lambda: state["fake"], raising=False)

    def configure(job=WORKER_JOB, **kw):
        state["fake"] = FakeKernel32(**kw)
        state["job"] = job
        return state["fake"], started
    return configure


def render(sched, job_id: str = "j"):
    job = fake_job(job_id, 0.2)
    fut = sched.submit(Request(job_id, "music", "numpy"), job, sink=MemorySink(job.n_out), cancel=threading.Event())
    return fut.result(timeout=60)


def the_worker(sched):
    with sched._lock:
        ws = list(sched._idle) + list(sched._busy)
    assert len(ws) == 1, ws
    return ws[0]


def assert_gone(pids) -> None:
    for p in pids:
        assert wait_until(lambda: pid_state(p) == "gone", 15) is not None, f"process {p} left behind"


# --- protocol ----------------------------------------------------------------------------------------------

def test_hello_is_the_first_frame(tmp_path):
    """K-S5b.1: the worker's first frame names the interpreter's pid, before the answer to the first request."""
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC}, read_hello=False)
    try:
        write_frame(p.stdin, {"req": "r", "op": "render", "job": fake_job("h", 0.05).to_json(), "kernel": "numpy",
                              "chunk": 48000})
        first, _ = read_frame(p.stdout)
        assert first == {"hello": 2, "pid": p.pid}, first
        while True:
            m, _ = read_frame(p.stdout)
            assert "hello" not in m, "one hello only"
            if m.get("done") or m.get("ok") is False:
                break
        assert m.get("done"), m
    finally:
        p.stdin.close()
        p.wait(timeout=10)


# --- interpreter behind a launcher -------------------------------------------------------------------------

@posix_only
def test_interpreter_behind_launcher_joins_the_job(fake_workers, make_scheduler, launcher, jobs):
    """K-S5b.2: reported pid != started pid -> OpenProcess + AssignProcessToJobObject with the reported pid, the
    process handle closed; the job handles are closed with the worker."""
    fake, started = jobs()
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    assert started == [w.pid]
    interp = stats.worker_pid
    assert interp != w.pid and w.interp_pid == interp, (interp, w.pid, w.interp_pid)
    opened = fake.named("OpenProcess")
    assert opened == [("OpenProcess", OPEN_ACCESS, False, interp)], fake.calls
    (h,) = [k for k, v in fake.processes.items() if v == interp]
    assert ("IsProcessInJob", h, WORKER_JOB) in fake.calls
    assert fake.named("AssignProcessToJobObject") == [("AssignProcessToJobObject", WORKER_JOB, h)]
    assert ("CloseHandle", h) in fake.calls, "the process handle is closed again"
    assert w.job_state == "assigned" and w.interp_job is None
    assert fake.jobs == [], "no extra job when the worker's job takes the interpreter"
    sched.shutdown()
    assert ("CloseHandle", WORKER_JOB) in fake.calls
    assert_gone([w.pid, interp])


@posix_only
def test_interpreter_already_in_job_is_not_assigned_again(fake_workers, make_scheduler, launcher, jobs):
    """K-S5b.3: an interpreter its launcher started inside the job is left as it is (no second assignment)."""
    fake, _ = jobs(in_job=True)
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    assert [c[3] for c in fake.named("OpenProcess")] == [stats.worker_pid]
    assert fake.named("AssignProcessToJobObject") == [], fake.calls
    (h,) = fake.processes
    assert ("CloseHandle", h) in fake.calls
    assert w.job_state == "inherited"


@posix_only
def test_refused_interpreter_gets_a_job_of_its_own(fake_workers, make_scheduler, launcher, jobs):
    """K-S5b.4: the worker's job refuses the interpreter (it is in its launcher's job already) -> a new kill-on-close
    job takes it; that job is closed with the worker."""
    fake, _ = jobs(refuse_worker_job=True)
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    (h,) = fake.processes
    assert fake.processes[h] == stats.worker_pid
    (own,) = fake.jobs
    assert fake.flags[own] == KILL_ON_JOB_CLOSE
    assert fake.named("AssignProcessToJobObject") == [("AssignProcessToJobObject", WORKER_JOB, h),
                                                     ("AssignProcessToJobObject", own, h)]
    assert ("CloseHandle", h) in fake.calls and ("CloseHandle", own) not in fake.calls
    assert w.job_state == "own job" and w.interp_job == own
    sched.shutdown()
    assert ("CloseHandle", own) in fake.calls and ("CloseHandle", WORKER_JOB) in fake.calls


@posix_only
@pytest.mark.parametrize("case", ["open fails", "both jobs refuse", "call raises"])
def test_join_failure_is_logged_not_raised(fake_workers, make_scheduler, launcher, jobs, caplog, case):
    """K-S5b.5: a failure to put the interpreter into a job is logged; the render goes on; no handle leaks."""
    kw = {"open fails": {"open_ok": False}, "both jobs refuse": {"refuse_worker_job": True, "refuse_own_job": True},
          "call raises": {"raise_on_open": True}}[case]
    fake, _ = jobs(**kw)
    caplog.set_level(logging.WARNING, logger="chimera_vocal.scheduler")
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    assert stats.n_out > 0 and w.job_state.startswith("failed"), w.job_state
    msgs = [r.getMessage() for r in caplog.records if r.name == "chimera_vocal.scheduler"]
    assert any("not in a job object" in m and str(stats.worker_pid) in m for m in msgs), msgs
    closed = {c[1] for c in fake.named("CloseHandle")}
    assert set(fake.processes) <= closed, "every opened process handle is closed"
    assert set(fake.jobs) <= closed, "a job of its own that could not take the interpreter is closed"
    assert None not in closed and 0 not in closed
    assert w.interp_job is None


@posix_only
def test_held_worker_joins_before_its_first_request(fake_workers, make_scheduler, launcher, jobs):
    """K-S5b.6: a worker kept warm (idle until a request comes) has its interpreter in the job already."""
    fake, _ = jobs()
    sched = make_scheduler(workers=2, idle_s=30)
    sched.hold_warm(True)
    assert wait_until(lambda: bool(sched._idle) and sched._idle[0].job_state is not None, 30) is not None, \
        (sched.status(), fake.calls)
    w = sched._idle[0]
    assert w.job_state == "assigned" and w.interp_pid != w.pid
    assert [c[3] for c in fake.named("OpenProcess")] == [w.interp_pid]
    stats = render(sched)
    assert stats.worker_pid == w.interp_pid and len(fake.named("OpenProcess")) == 1, "assigned once"
    sched.hold_warm(False)
    sched.shutdown()
    assert_gone([w.pid, w.interp_pid])


# --- nothing to do -----------------------------------------------------------------------------------------

def test_same_process_is_not_assigned_again(fake_workers, make_scheduler, jobs):
    """K-S5b.7: without a launcher the reported pid is the started process, which is in the job already."""
    fake, started = jobs()
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    assert stats.worker_pid == w.pid == started[0]
    assert fake.named("OpenProcess") == [] and fake.named("AssignProcessToJobObject") == []
    assert w.job_state == "same process"


@posix_only
def test_no_job_object_no_assignment(fake_workers, make_scheduler, launcher, jobs):
    """K-S5b.8: when no job object could be made for the started process there is nothing to join."""
    fake, _ = jobs(job=None)
    sched = make_scheduler(workers=2)
    stats = render(sched)
    w = the_worker(sched)
    assert stats.worker_pid != w.pid
    assert fake.calls == [] and w.job_state is None

