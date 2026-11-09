"""K-3.16 test_warmup_marker: two simultaneous `warm` workers — exactly one compiles under the
lock, the other waits for the marker; the marker carries hash, versions, CPU and the self-test; a stale lock (dead
pid, or older than 15 min) is broken; a marker of another numba version (or hash, or a failed self-test) is not
accepted by marker_ok; a failing self-test never leaves an accepted marker."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest

import dsp_numba
from chimera_vocal.dsp import backend

WARM = r"""
import time
from chimera_vocal.dsp import backend
if len(sys.argv) > 4 and sys.argv[4] == "falsify":
    real = backend.kernel_from_module
    def broken(mod):
        k = real(mod)
        inner = k.run_samples
        def run_samples(*a):
            inner(*a)
            a[-1][:] *= 1.01
        return k._replace(run_samples=run_samples)
    backend.kernel_from_module = broken
t0 = time.time()
r = backend.warm_numba(sys.argv[3])
print(json.dumps({"compiled": r.compiled, "waited": r.waited, "marker": r.marker, "seconds": r.seconds,
                  "detail": r.detail, "start": t0, "end": time.time(), "pid": os.getpid()}))
"""

pytestmark = pytest.mark.numba


def _popen(base, *extra):
    return subprocess.Popen([sys.executable, "-I", "-c", dsp_numba.PRELUDE + WARM, str(dsp_numba.REPO_ROOT),
                             str(dsp_numba.TESTS_DIR), str(base), *extra], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=str(dsp_numba.REPO_ROOT))


def _result(p, timeout=1800):
    out, err = p.communicate(timeout=timeout)
    assert p.returncode == 0, err[-5000:]
    return json.loads(out.strip().splitlines()[-1])


def _cache_dir(base) -> Path:
    return Path(dsp_numba.cache_dir_for(str(base)))


def _marker_ok(base):
    return dsp_numba.marker_ok_for(str(base))


def test_two_warm_workers_one_compiles(tmp_path):
    base = tmp_path / "cache"
    a, b = _popen(base), _popen(base)
    ra, rb = _result(a), _result(b)
    assert sorted([ra["compiled"], rb["compiled"]]) == [False, True], (ra, rb)
    waiter = ra if not ra["compiled"] else rb
    holder = rb if waiter is ra else ra
    assert waiter["waited"] and not holder["waited"]
    assert waiter["end"] >= holder["end"] - 1.0, "the waiter returned before the holder finished"
    m = _marker_ok(base)
    assert m is not None, "no valid marker after warm-up"
    assert holder["marker"] == m and waiter["marker"] == m
    for key in ("content_hash", "numba", "llvmlite", "numpy", "python", "platform", "cpu", "selftest", "created"):
        assert key in m, key
    assert m["selftest"]["passed"] is True and m["selftest"]["max_rel_err"] <= 1e-6
    assert m["python"] == platform.python_version()
    d = _cache_dir(base)
    assert (d / backend.MARKER_NAME).is_file() and not (d / backend.LOCK_NAME).exists()
    assert m["content_hash"].startswith(d.name)


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


@pytest.mark.parametrize("kind", ["dead-pid", "too-old"])
def test_stale_lock_is_broken(tmp_path, kind):
    base = tmp_path / "cache"
    d = _cache_dir(base)
    d.mkdir(parents=True)
    pid = _dead_pid() if kind == "dead-pid" else os.getpid()
    t = time.time() - (0 if kind == "dead-pid" else backend.STALE_LOCK_S + 300)
    (d / backend.LOCK_NAME).write_text(json.dumps({"pid": pid, "time": t}), encoding="utf-8")
    if kind == "too-old":
        os.utime(d / backend.LOCK_NAME, (t, t))
    r = _result(_popen(base), timeout=900)
    assert r["compiled"] and not r["waited"], r
    assert _marker_ok(base) is not None


def test_marker_mismatch_rejected(tmp_path):
    base = tmp_path / "cache"
    r = _result(_popen(base))
    good = _marker_ok(base)
    assert good is not None and r["compiled"]
    path = _cache_dir(base) / backend.MARKER_NAME
    raw = path.read_text(encoding="utf-8")
    for key, value in (("numba", "0.0.1"), ("llvmlite", "0.0.1"), ("numpy", "0.0.1"), ("python", "2.7.18"),
                       ("cpu", "not-this-cpu"), ("content_hash", "0" * 64)):
        bad = dict(good)
        bad[key] = value
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert _marker_ok(base) is None, key
    bad = dict(good)
    bad["selftest"] = dict(good["selftest"], passed=False)
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert _marker_ok(base) is None
    path.write_text("{not json", encoding="utf-8")
    assert _marker_ok(base) is None
    path.write_text(raw, encoding="utf-8")
    assert _marker_ok(base) == good
    assert "numba" not in sys.modules


def test_failed_selftest_leaves_no_marker(tmp_path):
    base = tmp_path / "cache"
    r = _result(_popen(base, "falsify"))
    assert r["compiled"]
    assert r["marker"] is None or r["marker"]["selftest"]["passed"] is False
    assert _marker_ok(base) is None, "a failed self-test left an accepted marker"
    assert not (_cache_dir(base) / backend.LOCK_NAME).exists(), "lock not released after a failed self-test"
