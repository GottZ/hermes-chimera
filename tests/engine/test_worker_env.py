"""K-5.12 test_worker_env: the worker's ``sys.path`` is exactly
``[root] + parent sys.path`` without "" and without the parent's working directory, it starts isolated and without
``site`` (``-I -S``: no base site-packages, no sitecustomize/.pth side effects at start-up), its cwd is the plugin
root, and the numpy kernel never loads numba. Red proof: start without ``-S``."""

from __future__ import annotations

import json
import os
import sys
import threading

from chimera_vocal.scheduler import MemorySink, Request
from support import REPO_ROOT
from support.fakes import fake_job, wait_until


def test_worker_env(fake_workers, make_scheduler, tmp_path):
    probe = tmp_path / "probe.json"
    sched = make_scheduler(workers=2)
    cwd = os.getcwd()
    expected = [str(REPO_ROOT)] + [p for p in sys.path if p not in ("", cwd)]
    job = fake_job(f"env|probe={probe}", 0.1)
    sched.submit(Request("r", "speech", "numpy"), job, sink=MemorySink(job.n_out),
                 cancel=threading.Event()).result(timeout=30)
    info = json.loads(probe.read_text(encoding="utf-8"))
    assert info["sys_path"] == expected
    assert "" not in info["sys_path"]
    assert os.path.realpath(info["cwd"]) == os.path.realpath(REPO_ROOT)
    assert info["flags"]["isolated"] == 1, "started with -I"
    assert info["flags"]["no_site"] == 1, "started with -S (no base site-packages at start-up)"
    assert info["sitecustomize_loaded"] is False
    assert info["numba_loaded"] is False, "the numpy kernel never imports numba"
    assert info["executable"] == sys.executable
    assert info["pid"] != os.getpid()


def test_test_hook_needs_chimera_testing(tmp_path):
    """The voice-factory hook is ignored unless CHIMERA_TESTING=1: the worker then loads the real voice (without the
    dsp package that fails with an error reply; with it the worker renders something other than FakeVoice)."""
    from chimera_vocal.worker import read_frame, write_frame
    from support.fakes import FACTORY_SPEC, reference_render, spawn_worker
    import numpy as np
    job = fake_job("nohook", 0.05)
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_WORKER_VOICE_FACTORY": FACTORY_SPEC,
                                                   "CHIMERA_TESTING": "0"})
    try:
        write_frame(p.stdin, {"req": "h", "op": "render", "job": job.to_json(), "kernel": "numpy", "chunk": 48000})
        pieces = []
        while True:
            m, raw = read_frame(p.stdout)
            if m.get("ok") is False or m.get("done"):
                break
            if "chunk" in m:
                pieces.append(np.frombuffer(raw, "<f4"))
        fake_rendered = m.get("done") and np.array_equal(np.concatenate(pieces), reference_render(job))
        assert not fake_rendered, "the factory hook must not act without CHIMERA_TESTING=1"
    finally:
        p.stdin.close()
        p.wait(timeout=10)


def test_worker_threads_env(fake_workers, make_scheduler, tmp_path, monkeypatch):
    """OPENBLAS/OMP/MKL thread counts are pinned to 1 in the worker environment."""
    probe = tmp_path / "probe.json"
    monkeypatch.setenv("CHIMERA_FAKE", "")
    sched = make_scheduler(workers=2)
    job = fake_job(f"env|probe={probe}", 0.1)
    sched.submit(Request("r", "speech", "numpy"), job, sink=MemorySink(job.n_out),
                 cancel=threading.Event()).result(timeout=30)
    assert wait_until(probe.exists, 5) is not None
    env = json.loads(probe.read_text(encoding="utf-8"))["env"]
    assert env == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_worker_follows_host_bytecode_setting(fake_workers, make_scheduler, tmp_path, monkeypatch):
    """Workers start with -I, which ignores PYTHONDONTWRITEBYTECODE: the host hands its effective setting over, so a
    host that writes no bytecode does not get __pycache__ written into the plugin directory by its workers."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    probe = tmp_path / "probe.json"
    sched = make_scheduler(workers=1)
    job = fake_job(f"env|probe={probe}", 0.1)
    sched.submit(Request("r", "speech", "numpy"), job, sink=MemorySink(job.n_out),
                 cancel=threading.Event()).result(timeout=30)
    info = json.loads(probe.read_text(encoding="utf-8"))
    assert info["dont_write_bytecode"] is True
