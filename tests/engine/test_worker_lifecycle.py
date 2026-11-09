"""K-5.5 test_worker_lifecycle (TESTPLAN L5, INTERFACES §9.1/§9.3, P1-08): frames, 10-MB pieces, crash -> one
retry, two numba crashes => ``failed``, parent death => worker gone <= 2 s (EOF watch), cancel => worker ended and
reaped, slot free <= 2 s, closing the ``speak_pcm16`` generator cancels the remaining windows.
Red proof: EOF only checked at the next read (worker keeps rendering after its parent died)."""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from chimera_vocal import BackendError, ChimeraError, RenderCancelled
from chimera_vocal.scheduler import MemorySink, Request
from chimera_vocal.worker import MAGIC, ProtocolError, read_frame, write_frame
from support import REPO_ROOT, TESTS_DIR
from support.fakes import (FACTORY_SPEC, fake_job, pid_state, read_logs, reference_render, spawn_worker,
                           wait_until)

TEN_MIB = 10 * 1024 * 1024


# --- frames -------------------------------------------------------------------------------------------------

def test_frames_roundtrip():
    buf = io.BytesIO()
    raw = np.arange(TEN_MIB // 4, dtype="<f4").tobytes()
    write_frame(buf, {"req": "a", "op": "quit"})
    write_frame(buf, {"req": "a", "chunk": 0, "offset": 0, "n": len(raw) // 4}, raw)
    write_frame(buf, {"req": "a", "done": True, "seconds": 0.5})
    data = buf.getvalue()
    n = struct.unpack("<I", data[4:8])[0]
    assert data[:4] == MAGIC and json.loads(data[8:8 + n].decode("utf-8")) == {"req": "a", "op": "quit"}
    n2 = struct.unpack("<I", data[12 + n:16 + n])[0]
    second = json.loads(data[16 + n:16 + n + n2].decode("utf-8"))
    assert data[8 + n:12 + n] == MAGIC and second["chunk"] == 0
    raw_hdr = data[16 + n + n2:24 + n + n2]
    assert raw_hdr[:4] == MAGIC and struct.unpack("<I", raw_hdr[4:])[0] == len(raw), "raw frame: magic + nbytes"
    buf.seek(0)
    m1, r1 = read_frame(buf)
    m2, r2 = read_frame(buf)
    m3, r3 = read_frame(buf)
    assert m1 == {"req": "a", "op": "quit"} and r1 is None
    assert m2["chunk"] == 0 and m2["n"] == TEN_MIB // 4 and r2 == raw
    assert m3["done"] is True and r3 is None
    with pytest.raises(EOFError):
        read_frame(buf)


@pytest.mark.parametrize("data", [b"XXXX\x02\x00\x00\x00{}", b"CHW2\x10\x00\x00\x00{\"a\":", b"CHW", b"CHW2\x05"])
def test_frames_malformed(data):
    with pytest.raises((ProtocolError, EOFError)) as ei:
        read_frame(io.BytesIO(data))
    if data.startswith(b"XXXX"):
        assert ei.type is ProtocolError


def test_frame_chunk_needs_raw():
    buf = io.BytesIO()
    write_frame(buf, {"req": "a", "chunk": 0, "offset": 0, "n": 4})
    buf.seek(0)
    with pytest.raises((ProtocolError, EOFError)):
        read_frame(buf)


# --- worker process, 10-MB pieces ---------------------------------------------------------------------------

def test_worker_ten_mib_chunks(tmp_path):
    job = fake_job("big", n_out=3_000_000)
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC})
    try:
        write_frame(p.stdin, {"req": "r1", "op": "render", "job": job.to_json(), "kernel": "numpy",
                              "chunk": TEN_MIB // 4})
        got, msgs = [], []
        while True:
            m, raw = read_frame(p.stdout)
            msgs.append(m)
            assert m.get("ok", True) is not False, m
            if "chunk" in m:
                assert len(raw) == 4 * m["n"]
                got.append((m["offset"], np.frombuffer(raw, "<f4")))
            if m.get("done"):
                break
        assert [(o, len(a)) for o, a in got] == [(0, TEN_MIB // 4), (TEN_MIB // 4, 3_000_000 - TEN_MIB // 4)]
        assert np.array_equal(np.concatenate([a for _, a in got]), reference_render(job))
        assert all(m["req"] == "r1" for m in msgs)
        write_frame(p.stdin, {"req": "q", "op": "quit"})
        assert p.wait(timeout=10) == 0
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_worker_error_reply(tmp_path):
    job = fake_job("bad|error=boom", 0.2)
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC})
    try:
        write_frame(p.stdin, {"req": "e1", "op": "render", "job": job.to_json(), "kernel": "numpy", "chunk": 48000})
        m, _ = read_frame(p.stdout)
        assert m["req"] == "e1" and m["ok"] is False and "boom" in m["error"] and m["trace"]
        good = fake_job("good", 0.1)
        write_frame(p.stdin, {"req": "e2", "op": "render", "job": good.to_json(), "kernel": "numpy", "chunk": 48000})
        chunks = []
        while True:
            m, raw = read_frame(p.stdout)
            if "chunk" in m:
                chunks.append(np.frombuffer(raw, "<f4"))
            if m.get("done"):
                break
        assert np.array_equal(np.concatenate(chunks), reference_render(good)), "the worker survives an error reply"
    finally:
        p.stdin.close()
        p.wait(timeout=10)


def test_worker_exits_on_stdin_eof_when_idle(tmp_path):
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC})
    p.stdin.close()
    assert p.wait(timeout=5) == 0


# --- scheduler: retry, numba failure ------------------------------------------------------------------------

def run(sched, job, *, kernel="numpy", priority="music", timeout=30):
    sink = MemorySink(job.n_out)
    fut = sched.submit(Request(job.id, priority, kernel), job, sink=sink, cancel=threading.Event())
    stats = fut.result(timeout=timeout)
    return stats, sink.data


def test_crash_retried_once(fake_workers, make_scheduler, tmp_path):
    marker = tmp_path / "crashed-once"
    logs = tmp_path / "log"
    job = fake_job(f"c1|crash-once={marker}|log={logs}", 0.3)
    sched = make_scheduler(workers=2)
    stats, data = run(sched, job)
    assert marker.exists(), "the first attempt crashed"
    assert np.array_equal(data, reference_render(job))
    pids = {int(r[1]) for r in read_logs(logs) if r[0] == "factory"}
    assert len(pids) == 2 and stats.worker_pid in pids and int(marker.read_text()) != stats.worker_pid
    assert sched.status()["numba_state"] in ("absent",)


def test_crash_twice_fails_job(fake_workers, make_scheduler, tmp_path):
    sched = make_scheduler(workers=2)
    job = fake_job("c2|crash", 0.2)
    with pytest.raises(BackendError):
        run(sched, job)
    stats, data = run(sched, fake_job("after", 0.2))
    assert np.array_equal(data, reference_render(fake_job("after", 0.2))), "the scheduler recovers"


def test_two_numba_crashes_mark_failed(fake_workers, make_scheduler, monkeypatch, tmp_path):
    import chimera_vocal.scheduler as sched_mod
    monkeypatch.setattr(sched_mod, "_numba_installed", lambda: True)
    monkeypatch.setattr(sched_mod, "_marker_ok", lambda base: {"selftest": {"passed": True}})
    sched = make_scheduler(workers=2)
    assert sched.choose_kernel() == "numba"
    with pytest.raises(BackendError):
        run(sched, fake_job("n1|crash-numba", 0.2), kernel="numba")
    st = sched.status()
    assert st["numba_state"] == "failed" and st["numba_error"]
    assert sched.choose_kernel() == "numpy", "after two numba crashes new requests use numpy"
    stats, data = run(sched, fake_job("n2", 0.2), kernel=sched.choose_kernel())
    assert stats.kernel == "numpy" and np.array_equal(data, reference_render(fake_job("n2", 0.2), "numpy"))


# --- parent death, cancel ------------------------------------------------------------------------------------

PARENT = textwrap.dedent(r"""
    import os, sys, threading, time
    sys.path[:0] = [sys.argv[1], sys.argv[2]]
    os.environ["CHIMERA_WORKER_VOICE_FACTORY"] = "support.fakes:fake_voice_factory"
    import chimera_vocal.scheduler as s
    s._numba_installed = lambda: False
    s._marker_ok = lambda base: None
    from support.fakes import fake_job
    from chimera_vocal.scheduler import MemorySink, RenderScheduler, Request
    sched = RenderScheduler(workers=2, cache_base=sys.argv[3], root=sys.argv[1])
    job = fake_job("long|slow=30|log=" + sys.argv[4], 5.0)
    sched.submit(Request("long", "music", "numpy"), job, sink=MemorySink(job.n_out), cancel=threading.Event())
    print("submitted", flush=True)
    time.sleep(600)
""")


def _render_pids(logs) -> list[int]:
    return [int(r[1]) for r in read_logs(logs) if r[0] == "render"]


def test_parent_death_ends_worker(tmp_path):
    logs = tmp_path / "log"
    parent = subprocess.Popen([sys.executable, "-c", PARENT, str(REPO_ROOT), str(TESTS_DIR), str(tmp_path / "c"),
                               str(logs)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert wait_until(lambda: _render_pids(logs), 30) is not None, parent.stderr.read1(4000) \
            if parent.poll() is not None else "worker never started rendering"
        wpid = _render_pids(logs)[0]
        assert pid_state(wpid) == "alive"
        parent.kill()
        parent.wait()
        took = wait_until(lambda: pid_state(wpid) != "alive", 5.0)
        assert took is not None and took <= 2.0, f"worker {wpid} still alive {took} s after its parent died"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for pid in _render_pids(logs):
            if pid_state(pid) == "alive":
                os.kill(pid, 9)


def test_cancel_ends_and_reaps_worker(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=2)
    job = fake_job(f"long|slow=30|log={logs}", 5.0)
    cancel = threading.Event()
    fut = sched.submit(Request("long", "music", "numpy"), job, sink=MemorySink(job.n_out), cancel=cancel)
    assert wait_until(lambda: _render_pids(logs), 30) is not None
    wpid = _render_pids(logs)[0]
    t0 = time.monotonic()
    cancel.set()
    with pytest.raises(RenderCancelled):
        fut.result(timeout=5)
    assert time.monotonic() - t0 <= 2.0
    took = wait_until(lambda: pid_state(wpid) == "gone", 2.0)
    assert took is not None, f"worker {wpid} is {pid_state(wpid)} (zombie = not reaped)"
    t1 = time.monotonic()
    stats, data = run(sched, fake_job("next", 0.2), timeout=10)
    assert time.monotonic() - t1 <= 2.0 + 1.0, "the slot is free again"
    assert sched.status()["running"] == 0


def test_cancel_drops_queued_jobs(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=2)       # one music slot
    cancel = threading.Event()
    jobs = [fake_job(f"q{i}|slow=30|log={logs}", 2.0) for i in range(3)]
    futs = [sched.submit(Request("req", "music", "numpy"), j, sink=MemorySink(j.n_out), cancel=cancel) for j in jobs]
    assert wait_until(lambda: _render_pids(logs), 30) is not None
    cancel.set()
    for f in futs:
        with pytest.raises(RenderCancelled):
            f.result(timeout=5)
    st = sched.status()
    assert st["queued_music"] == 0 and st["running"] == 0
    assert len({r[2] for r in read_logs(logs) if r[0] == "render"}) == 1, "queued jobs never started"


def test_stream_close_cancels_remaining(stub_planner, make_engine, fake_workers, monkeypatch, tmp_path):
    logs = tmp_path / "log"
    monkeypatch.setenv("CHIMERA_FAKE", f"slow=0.4|log={logs}")
    eng = make_engine(workers=2)
    text = "One two three, four five six. Seven eight nine, ten eleven twelve. Thirteen fourteen, fifteen sixteen."
    gen = eng.speak_pcm16(text, "male")
    first = next(gen)
    assert isinstance(first, (bytes, bytearray)) and len(first) > 0 and len(first) % 2 == 0
    gen.close()
    t_close = time.monotonic()

    def quiet() -> bool:
        st = eng.scheduler.status()
        return st["running"] == 0 and st["queued_speech"] == 0
    took = wait_until(quiet, 3.0)
    assert took is not None and took <= 2.0, eng.scheduler.status()
    n_after = len(read_logs(logs))
    time.sleep(1.0)
    assert len(read_logs(logs)) == n_after, "no render continues after the generator was closed"
    windows = {r[2] for r in read_logs(logs) if r[0] == "render"}
    assert len(windows) < 6, "the remaining windows were not all rendered"
    assert time.monotonic() - t_close < 10
