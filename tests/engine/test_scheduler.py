"""K-5.4 test_scheduler: the reserved speech slot (a speech job starts at
once while music fills every music slot), the limit of concurrent renders, FIFO within a priority, one kernel per
request, an error inside a job. Red proof: one common queue for speech and music."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chimera_vocal import ChimeraError
from chimera_vocal.scheduler import MemorySink, RenderStats, Request, default_workers
from support.fakes import fake_job, read_logs, reference_render, wait_until


def submit(sched, job, *, priority="music", kernel="numpy", cancel=None, req_id=None):
    sink = MemorySink(job.n_out)
    fut = sched.submit(Request(req_id or job.id, priority, kernel), job, sink=sink, cancel=cancel or threading.Event())
    done = {}
    fut.add_done_callback(lambda f: done.setdefault("t", time.monotonic()))
    return fut, sink, done


@pytest.mark.parametrize("cpus,w", [(None, 2), (1, 2), (2, 2), (3, 2), (4, 3), (5, 4), (16, 4)])
def test_default_workers(cpus, w):
    assert default_workers(cpus) == w


def test_speech_slot_free_while_music_fills_all_slots(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=3)                       # music slots: 2
    music = [submit(sched, fake_job(f"m{i}|slow=2.5|log={logs}", 0.5)) for i in range(3)]
    assert wait_until(lambda: len({r[2] for r in read_logs(logs) if r[0] == "render"}) >= 2, 30) is not None
    t0 = time.monotonic()
    speech_job = fake_job(f"s0|log={logs}", 0.3)
    fut, sink, done = submit(sched, speech_job, priority="speech")
    stats = fut.result(timeout=30)
    assert isinstance(stats, RenderStats) and stats.n_out == speech_job.n_out
    assert done["t"] - t0 < 2.0, "speech waited for music"
    assert not any(f.done() for f, _, _ in music), "speech finished before any music job"
    assert np.array_equal(sink.data, reference_render(speech_job))
    for f, _, _ in music:
        f.result(timeout=60)


def test_single_worker_speech_gets_extra_worker(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=1)
    m, _, mdone = submit(sched, fake_job(f"m|slow=3|log={logs}", 0.5))
    assert wait_until(lambda: any(r[0] == "render" for r in read_logs(logs)), 30) is not None
    s, _, sdone = submit(sched, fake_job(f"s|log={logs}", 0.2), priority="speech")
    s.result(timeout=30)
    assert not m.done(), "with one worker, speech starts an additional worker instead of waiting"
    pids = {r[1] for r in read_logs(logs) if r[0] == "render"}
    assert len(pids) == 2
    m.result(timeout=60)


def test_concurrency_limit_and_fifo(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=2)                       # one music slot
    subs = [submit(sched, fake_job(f"f{i}|slow=0.3|log={logs}", 0.2)) for i in range(4)]
    for f, _, _ in subs:
        f.result(timeout=60)
    starts = {}
    for r in read_logs(logs):
        if r[0] == "render":
            starts.setdefault(r[2].split("|")[0], float(r[4]))
    order = sorted(starts, key=starts.get)
    assert order == ["f0", "f1", "f2", "f3"], "FIFO within the music priority"
    ends = {f"f{i}": d["t"] for i, (_, _, d) in enumerate(subs)}
    # wall clock of the log (time.time) vs monotonic: compare only orderings inside one clock
    for a, b in zip(order, order[1:]):
        assert ends[a] <= ends[b] + 1e-3
    assert sched.status()["running"] == 0


def test_music_never_exceeds_slots(fake_workers, make_scheduler, tmp_path):
    sched = make_scheduler(workers=3)                       # music slots: 2
    peak = {"n": 0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak["n"] = max(peak["n"], sched.status()["running"])
            time.sleep(0.01)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    subs = [submit(sched, fake_job(f"p{i}|slow=0.4", 0.2)) for i in range(5)]
    for f, _, _ in subs:
        f.result(timeout=60)
    stop.set()
    th.join()
    assert 1 <= peak["n"] <= 2


def test_kernel_per_request(fake_workers, make_scheduler):
    sched = make_scheduler(workers=2)
    job = fake_job("k1", 0.2)
    fut, sink, _ = submit(sched, job, kernel="numba")
    stats = fut.result(timeout=30)
    assert stats.kernel == "numba"
    assert np.array_equal(sink.data, reference_render(job, "numba")), "the worker rendered with the request's kernel"
    fut2, sink2, _ = submit(sched, job, kernel="numpy")
    assert fut2.result(timeout=30).kernel == "numpy" and np.array_equal(sink2.data, reference_render(job, "numpy"))


def test_error_in_job(fake_workers, make_scheduler, tmp_path):
    logs = tmp_path / "log"
    sched = make_scheduler(workers=2)
    fut, _, _ = submit(sched, fake_job(f"e|error=boom|log={logs}", 0.2))
    with pytest.raises(ChimeraError) as ei:
        fut.result(timeout=30)
    assert "boom" in str(ei.value)
    assert sum(1 for r in read_logs(logs) if r[0] == "factory") == 1, "an error reply is not retried"
    fut2, _, _ = submit(sched, fake_job(f"ok|log={logs}", 0.2))
    s2 = fut2.result(timeout=30)
    pids = [int(r[1]) for r in read_logs(logs) if r[0] == "factory"]
    assert s2.worker_pid == pids[0], "the worker survives an error reply"


def test_status_shape(fake_workers, make_scheduler):
    sched = make_scheduler(workers=3)
    st = sched.status()
    for key in ("workers", "workers_alive", "worker_pids", "running", "queued_speech", "queued_music", "numba_state",
                "numba_error", "rtf", "held_warm"):
        assert key in st, key
    assert st["workers"] == 3 and st["running"] == 0 and st["numba_state"] == "absent"
    assert set(st["rtf"]) == {"numba", "numpy"}


def test_hold_warm_keeps_a_worker(fake_workers, make_scheduler):
    sched = make_scheduler(workers=2, idle_s=0.5)
    sched.hold_warm(True)
    assert wait_until(lambda: sched.status()["workers_alive"] >= 1, 10) is not None
    time.sleep(1.5)
    assert sched.status()["workers_alive"] >= 1, "a held worker survives the idle timeout"
    sched.hold_warm(False)
    assert wait_until(lambda: sched.status()["workers_alive"] == 0, 10) is not None, "idle workers quit"


def test_shutdown_reaps_workers(fake_workers, make_scheduler):
    from support.fakes import pid_state
    sched = make_scheduler(workers=2)
    fut, _, _ = submit(sched, fake_job("x", 0.1))
    fut.result(timeout=30)
    pids = list(sched.status()["worker_pids"])
    sched.shutdown()
    for pid in pids:
        assert wait_until(lambda: pid_state(pid) == "gone", 5) is not None, f"worker {pid} not reaped"
