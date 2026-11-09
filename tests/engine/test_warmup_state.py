"""numba warm-up state machine of the RenderScheduler (REVIEW-2A R2A-01, R2A-07) and the life of its maintenance
thread (R2A-15).

The warm-up worker is replaced by a stub that answers the ``warm`` op the way ``dsp.backend.warm_numba`` does
(``WarmResult`` as JSON), and the marker check by a flag, so every outcome is reachable without numba.

* R2A-01: once the marker disappears (housekeeping removed the cache folder), the next request warms up again.
  Red proof: the one-warm-up-per-process flag of 9a616be (no second warm-up, state stays ``cold``).
* R2A-07: a warm-up whose self-test failed ends ``failed`` with the reason; one that only timed out waiting for
  another process's lock ends ``cold`` with the reason and is retried; an explicit warm-up retries a failure.
  Red proof: only ``ok: false`` counted as failure (state ``cold``, no reason).
* R2A-15: the maintenance thread ends once nothing is left to watch and comes back with the next job; cancellation
  still works afterwards. Red proof: a ``while not stop.wait(0.05)`` loop for the life of the process.
"""

from __future__ import annotations

import threading

import pytest

import chimera_vocal.scheduler as S
from chimera_vocal import RenderCancelled
from chimera_vocal.scheduler import MemorySink, Request
from support import REPO_ROOT
from support.fakes import fake_job, read_logs, wait_until

MARKER = {"content_hash": "0123456789abcdef", "selftest": {"passed": True}}


def _warm_reply(marker, *, compiled=True, waited=False, detail="compiled"):
    return {"req": "warm", "ok": True, "warm": {"compiled": compiled, "waited": waited, "marker": marker,
                                                "seconds": 0.01, "detail": detail}}


@pytest.fixture
def stub_warm(monkeypatch):
    """numba "installed", marker = state["marker"], warm-up worker = stub answering with state["reply"]()."""
    state = {"marker": False, "warms": 0, "reply": None}

    class StubWorker:
        def __init__(self, sched):
            state["warms"] += 1

        def send(self, msg):
            assert msg["op"] == "warm"

        def read(self):
            return state["reply"](), None

        def quit(self):
            pass

        def tail(self):
            return ""

    monkeypatch.setattr(S, "_numba_installed", lambda: True)
    monkeypatch.setattr(S, "_marker_ok", lambda base: dict(MARKER) if state["marker"] else None)
    monkeypatch.setattr(S, "_Worker", StubWorker)
    return state


@pytest.fixture
def sched(tmp_path):
    s = S.RenderScheduler(workers=2, cache_base=str(tmp_path / "numba"), root=str(REPO_ROOT))
    yield s
    s.shutdown()


def _settled(s):
    return wait_until(lambda: s.status()["numba_state"] != "warming", 5.0) is not None


def test_rewarm_after_marker_loss(stub_warm, sched):
    def ok():
        stub_warm["marker"] = True
        return _warm_reply(dict(MARKER))
    stub_warm["reply"] = ok
    assert sched.choose_kernel() == "numpy"
    assert _settled(sched) and sched.numba_state() == "ready"
    assert sched.choose_kernel() == "numba" and stub_warm["warms"] == 1
    stub_warm["marker"] = False                      # the cache folder was removed by housekeeping
    assert sched.numba_state() == "cold"
    assert sched.choose_kernel() == "numpy"
    assert wait_until(lambda: stub_warm["warms"] == 2 and sched.numba_state() == "ready", 5.0) is not None, \
        (stub_warm["warms"], sched.status())
    assert sched.choose_kernel() == "numba"


def test_selftest_failure_is_failed_with_reason(stub_warm, sched):
    detail = "selftest failed: max |delta| 9.900e-03 x peak over 4800 samples"
    stub_warm["reply"] = lambda: _warm_reply(None, detail=detail)
    assert sched.choose_kernel() == "numpy"
    assert _settled(sched)
    st = sched.status()
    assert st["numba_state"] == "failed", st
    assert st["numba_error"] and "selftest failed" in st["numba_error"], st
    assert sched.choose_kernel() == "numpy" and stub_warm["warms"] == 1, "a failed warm-up is not repeated by itself"


def test_lock_timeout_is_cold_with_reason_and_retried(stub_warm, sched):
    stub_warm["reply"] = lambda: _warm_reply(None, compiled=False, waited=True,
                                             detail="timed out waiting for the warm-up lock")
    sched.choose_kernel()
    assert _settled(sched)
    st = sched.status()
    assert st["numba_state"] == "cold" and "timed out" in (st["numba_error"] or ""), st
    sched.choose_kernel()
    assert wait_until(lambda: stub_warm["warms"] == 2, 5.0) is not None


def test_marker_not_visible_to_host_is_failed(stub_warm, sched):
    stub_warm["reply"] = lambda: _warm_reply(dict(MARKER))      # worker reports a marker, the host check fails
    sched.choose_kernel()
    assert _settled(sched)
    st = sched.status()
    assert st["numba_state"] == "failed" and "does not match" in (st["numba_error"] or ""), st
    sched.choose_kernel()
    assert stub_warm["warms"] == 1, "no warm-up loop"


def test_explicit_warmup_retries_a_failure(stub_warm, sched):
    stub_warm["reply"] = lambda: _warm_reply(None, detail="selftest failed: x")
    sched.choose_kernel()
    assert _settled(sched) and sched.numba_state() == "failed"
    assert sched.start_warmup() is False, "without the explicit request a failure stays"

    def ok():
        stub_warm["marker"] = True
        return _warm_reply(dict(MARKER))
    stub_warm["reply"] = ok
    assert sched.start_warmup(retry_failed=True) is True
    assert _settled(sched) and sched.numba_state() == "ready" and sched.status()["numba_error"] is None


def test_backend_numpy_never_warms(stub_warm, tmp_path):
    s = S.RenderScheduler(workers=2, cache_base=str(tmp_path), root=str(REPO_ROOT), backend="numpy")
    try:
        assert s.choose_kernel() == "numpy" and s.start_warmup(retry_failed=True) is False
        assert stub_warm["warms"] == 0
    finally:
        s.shutdown()


def test_engine_warmup_requests_a_retry(make_engine, monkeypatch):
    eng = make_engine(workers=2)
    seen = []
    monkeypatch.setattr(eng.scheduler, "start_warmup", lambda **kw: seen.append(kw) or False)
    monkeypatch.setattr(eng, "_warm_bg", lambda: None)
    eng.warmup()
    assert seen == [{"retry_failed": True}]


def _maint_alive(s) -> bool:
    m = s._maint
    return m is not None and m.is_alive()


def test_maintenance_thread_ends_when_idle(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    s = make_scheduler(workers=2, idle_s=0.3)
    job = fake_job("first", 0.2)
    s.submit(Request("r1", "music", "numpy"), job, sink=MemorySink(job.n_out), cancel=threading.Event()).result(30)
    assert wait_until(lambda: s.status()["workers_alive"] == 0, 10.0) is not None, s.status()
    assert wait_until(lambda: not _maint_alive(s), 5.0) is not None, "the maintenance thread keeps polling"
    # it comes back with the next job, and cancellation is still watched
    cancel = threading.Event()
    job = fake_job(f"long|slow=30|log={logs}", 5.0)
    fut = s.submit(Request("r2", "music", "numpy"), job, sink=MemorySink(job.n_out), cancel=cancel)
    assert wait_until(lambda: any(r[0] == "render" for r in read_logs(logs)), 30) is not None
    assert _maint_alive(s)
    cancel.set()
    with pytest.raises(RenderCancelled):
        fut.result(timeout=5)
    assert wait_until(lambda: s.status()["workers_alive"] == 0 and not _maint_alive(s), 10.0) is not None


def test_held_worker_keeps_maintenance(fake_workers, make_scheduler):
    s = make_scheduler(workers=2, idle_s=0.2)
    s.hold_warm(True)
    assert wait_until(lambda: s.status()["workers_alive"] == 1, 15.0) is not None
    threading.Event().wait(0.6)
    assert _maint_alive(s) and s.status()["workers_alive"] == 1
    s.hold_warm(False)
    assert wait_until(lambda: s.status()["workers_alive"] == 0 and not _maint_alive(s), 10.0) is not None
