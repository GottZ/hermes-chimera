"""K-5.12 test_worker_env (TESTPLAN L5, INTERFACES §9.3, P1-20): the worker's ``sys.path`` is exactly
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
