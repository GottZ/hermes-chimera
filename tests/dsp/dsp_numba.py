"""Run dsp_cases with the numba kernel in a fresh interpreter (the test process never imports numba), and the
worker-side loading those subprocesses use.

Standalone mode (a tree with the acoustics package but without the ``control`` package): the subprocess loads
only the core + dsp JIT files and the self-test replays the recorded /ɑ/ track instead of the real control model
(``install_standalone_overrides``). With ``control`` present the same tests load everything with
``backend.load_numba``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

import support

TESTS_DIR = support.TESTS_DIR
REPO_ROOT = support.REPO_ROOT

BOOT = r"""
import json, sys
root, tests = sys.argv[1], sys.argv[2]
sys.path[:0] = [root, tests, tests + "/dsp"]
import dsp_cases
sys.exit(dsp_cases.subprocess_main(json.loads(sys.stdin.read())))
"""

PRELUDE = r"""
import json, os, sys
root, tests = sys.argv[1], sys.argv[2]
sys.path[:0] = [root, tests, tests + "/dsp"]
import dsp_numba
dsp_numba.install_standalone_overrides()
"""
"""Start of an ad-hoc worker script (K-3.13, K-3.16): paths + standalone overrides; argv[1:3] = root, tests."""


def control_manifest_available() -> bool:
    try:
        return importlib.util.find_spec("chimera_vocal.control.jit_manifest") is not None
    except ModuleNotFoundError:
        return False


def control_model_available() -> bool:
    try:
        return importlib.util.find_spec("chimera_vocal.control.track") is not None
    except ModuleNotFoundError:
        return False


def install_standalone_overrides() -> None:
    """Without the control package: JIT files = core + dsp; the self-test's control model = recorded sustained /ɑ/
    (same scenario)."""
    from chimera_vocal.core.jit import CORE_JIT_FILES
    from chimera_vocal.dsp import backend, voice
    from chimera_vocal.dsp.jit_manifest import JIT_FILES

    if not control_manifest_available():
        backend.all_jit_files = lambda: CORE_JIT_FILES + JIT_FILES
    if not control_model_available():
        import recorded_control as rc
        voice.CONTROL_FACTORY = rc.recorded_factory("sustain-aa-male")


def load_for_tests(cache_dir: str) -> dict:
    """Inside a subprocess: {"numba": KernelImpl, "geometry": GeometryImpl | None}."""
    from chimera_vocal.dsp import backend

    install_standalone_overrides()
    if control_manifest_available():
        k, g = backend.load_numba(cache_dir)
        return {"numba": k, "geometry": g}
    mods = backend.load_jit_modules(backend.all_jit_files(), backend.numba_cache_dir(cache_dir))
    return {"numba": backend.kernel_from_module(mods["_chimera_kernel_numba"]), "geometry": None}


def dispatch_stats() -> dict:
    """Cache hits/misses of every compiled function of the loaded JIT modules (inside a subprocess)."""
    out = {}
    for modname in ("_chimera_nbmath", "_chimera_kernel_numba", "_chimera_geometry_ref"):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for name in dir(mod):
            f = getattr(mod, name)
            st = getattr(f, "stats", None)
            if st is not None and hasattr(st, "cache_hits"):
                out[f"{modname}.{name}"] = {"hits": sum(st.cache_hits.values()), "misses": sum(st.cache_misses.values())}
    return out


def run_subprocess(case: str, args: dict, *, cache_dir, tmp_path, timeout: float = 3600.0) -> dict:
    out = Path(tmp_path) / f"{case}-{uuid.uuid4().hex[:8]}.npz"
    req = {"case": case, "args": args, "out": str(out), "cache_dir": str(cache_dir)}
    r = subprocess.run([sys.executable, "-I", "-c", BOOT, str(REPO_ROOT), str(TESTS_DIR)], input=json.dumps(req),
                       capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
    assert r.returncode == 0, f"numba subprocess failed (rc {r.returncode}):\n{r.stderr[-6000:]}"
    with np.load(out) as z:
        res = {k: z[k] for k in z.files}
    res["_stats"] = json.loads(Path(str(out) + ".info.json").read_text(encoding="utf-8"))
    return res


def run_case(kernel_name: str, case: str, *, cache_dir, tmp_path, **args) -> dict:
    """numpy: in this process; numba: in a subprocess."""
    import dsp_cases

    if kernel_name == "numpy":
        return dsp_cases.run_local(case, kernel_name="numpy", **args)
    return run_subprocess(case, {"kernel_name": kernel_name, **args}, cache_dir=cache_dir, tmp_path=tmp_path)


def run_script(script: str, *args: str, timeout: float = 1800.0, env: dict | None = None, cwd=None):
    """Run PRELUDE + script in a fresh isolated interpreter; returns the CompletedProcess."""
    import os

    return subprocess.run([sys.executable, "-I", "-c", PRELUDE + script, str(REPO_ROOT), str(TESTS_DIR), *args],
                          capture_output=True, text=True, timeout=timeout, cwd=cwd or REPO_ROOT,
                          env=env if env is not None else dict(os.environ))


def standalone_jit_files():
    """ALL_JIT_FILES as the test workers load them (without the control package: core + dsp)."""
    from chimera_vocal.core.jit import CORE_JIT_FILES
    from chimera_vocal.dsp.jit_manifest import JIT_FILES
    if control_manifest_available():
        from chimera_vocal.control.jit_manifest import JIT_FILES as CJ
        return CORE_JIT_FILES + CJ + JIT_FILES
    return CORE_JIT_FILES + JIT_FILES


def _with_files(fn, *args):
    from chimera_vocal.dsp import backend
    if control_manifest_available():
        return fn(*args)
    old = backend.all_jit_files
    try:
        backend.all_jit_files = standalone_jit_files
        return fn(*args)
    finally:
        backend.all_jit_files = old


def cache_dir_for(base: str) -> str:
    """backend.numba_cache_dir(base) as the test workers compute it (host side, no numba import)."""
    from chimera_vocal.dsp import backend
    return _with_files(lambda b: backend.numba_cache_dir(b), base)


def marker_ok_for(base: str):
    """backend.marker_ok(base) for the test workers' file set (host side, no numba import)."""
    from chimera_vocal.dsp import backend
    return _with_files(lambda b: backend.marker_ok(b), base)
