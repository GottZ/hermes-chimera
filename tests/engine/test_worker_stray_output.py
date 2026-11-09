"""K-5.8 test_worker_stray_output (TESTPLAN L5, INTERFACES §9.3): ``print()`` and raw writes to fd 1 inside the worker
(third-party code, numba diagnostics) do not corrupt the protocol - fd 1 is redirected to stderr, the protocol runs
on a duplicate of the original fd 1. Red proof: protocol on fd 1."""

from __future__ import annotations

import threading

import numpy as np

from chimera_vocal.scheduler import MemorySink, Request
from chimera_vocal.worker import read_frame, write_frame
from support.fakes import FACTORY_SPEC, fake_job, reference_render, spawn_worker


def test_stray_output_direct(tmp_path):
    job = fake_job("noisy|stray", 0.5)
    err = tmp_path / "stderr.txt"
    p = spawn_worker(err, env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC})
    try:
        write_frame(p.stdin, {"req": "r", "op": "render", "job": job.to_json(), "kernel": "numpy", "chunk": 4800})
        chunks = []
        while True:
            m, raw = read_frame(p.stdout)
            assert m.get("ok", True) is not False, m
            if "chunk" in m:
                chunks.append(np.frombuffer(raw, "<f4"))
            if m.get("done"):
                break
        assert np.array_equal(np.concatenate(chunks), reference_render(job))
        write_frame(p.stdin, {"req": "q", "op": "quit"})
        assert p.wait(timeout=10) == 0
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert b"stray print from the voice" in err.read_bytes(), "stray output goes to stderr"


def test_stray_output_scheduler(fake_workers, make_scheduler):
    sched = make_scheduler(workers=2)
    job = fake_job("noisy|stray", 1.0)
    sink = MemorySink(job.n_out)
    stats = sched.submit(Request("r", "speech", "numpy"), job, sink=sink, cancel=threading.Event()).result(timeout=30)
    assert stats.n_out == job.n_out and np.array_equal(sink.data, reference_render(job))
