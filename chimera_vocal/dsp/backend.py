"""Kernel choice support: numba loading only inside render worker processes, a content-addressed cache, the
warm-up lock and marker, and the self-test.

numba runs only in worker processes: compiled code is not bounds-checked, so an indexing error would corrupt or kill
the process that runs it, and compiling holds the GIL for seconds; neither may happen in the gateway. The host process
may import this module and call ``marker_ok``/``numba_cache_dir`` — neither imports numba. ``load_numba`` and
``warm_numba`` import numba and are called only from ``worker.py``.

JIT files are loaded under fixed module names because numba keys its on-disk cache by module name, and the cache
folder carries a hash of all JIT files because a cached caller does not notice a changed callee in another file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from ..core.jit import JitFile
from .kernel_numpy import KernelImpl

PACKAGE_DIR = Path(__file__).resolve().parent.parent
_LOAD_INFO: dict = {"cache_dir": None, "cached": False, "reason": "not loaded"}

if TYPE_CHECKING:
    from ..control.track import GeometryImpl

LOCK_NAME = ".warm.lock"
MARKER_NAME = "marker.json"
WAIT_POLL_S = 0.5
WAIT_MAX_S = 300.0
STALE_LOCK_S = 900.0
SELFTEST_FILE = "selftest.json"
SELFTEST_TOL = 1e-6


def all_jit_files() -> tuple[JitFile, ...]:
    """ALL_JIT_FILES = CORE_JIT_FILES + control.jit_manifest.JIT_FILES + dsp.jit_manifest.JIT_FILES (load order)."""
    from ..control.jit_manifest import JIT_FILES as CONTROL_FILES
    from ..core.jit import CORE_JIT_FILES
    from .jit_manifest import JIT_FILES as DSP_FILES
    return CORE_JIT_FILES + CONTROL_FILES + DSP_FILES


def content_hash(files: tuple[JitFile, ...]) -> str:
    """sha256 over the bytes of the files (paths relative to the chimera_vocal package), in order; hex."""
    h = hashlib.sha256()
    for f in files:
        h.update((PACKAGE_DIR / f.relpath).read_bytes())
    return h.hexdigest()


def numba_cache_dir(base: str) -> str:
    """<base>/<content_hash(all_jit_files())[:16]>; creates nothing."""
    return os.path.join(base, content_hash(all_jit_files())[:16])


def _writable(d: str) -> tuple[bool, str]:
    """os.makedirs + a temp-file probe, as numba's cache locators do (numba/core/caching.py:108-112)."""
    try:
        os.makedirs(d, exist_ok=True)
        with tempfile.TemporaryFile(dir=d):
            pass
        return True, ""
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


def load_jit_modules(files: tuple[JitFile, ...], cache_dir: str | None) -> dict[str, ModuleType]:
    """Import every file under its fixed name with CHIMERA_JIT = numba.njit(cache=<cache_dir is writable>, nogil=True,
    fastmath=False, error_model="numpy"); numba.config.CACHE_DIR is set only around the imports and restored.
    Returns {fixed_name: module}. Worker only."""
    import numba

    global _LOAD_INFO
    cached, reason = (False, "no cache directory") if cache_dir is None else _writable(cache_dir)
    old = numba.config.CACHE_DIR
    mods: dict[str, ModuleType] = {}
    try:
        if cached:
            numba.config.CACHE_DIR = cache_dir
        for f in files:
            path = PACKAGE_DIR / f.relpath
            spec = importlib.util.spec_from_file_location(f.fixed_name, path)
            mod = importlib.util.module_from_spec(spec)
            mod.CHIMERA_JIT = numba.njit(cache=cached, nogil=True, fastmath=False, error_model="numpy",
                                         boundscheck=False)
            sys.modules[f.fixed_name] = mod
            spec.loader.exec_module(mod)
            mods[f.fixed_name] = mod
    finally:
        numba.config.CACHE_DIR = old
    _LOAD_INFO = {"cache_dir": cache_dir if cached else None, "cached": cached, "reason": reason}
    return mods


def kernel_from_module(mod: ModuleType) -> KernelImpl:
    """KernelImpl("numba", ...) from a loaded ``_chimera_kernel_numba`` module."""
    return KernelImpl("numba", mod.run_samples, mod.output_chain, mod.flush_subnormals)


def load_numba(cache_base: str | None) -> "tuple[KernelImpl, GeometryImpl]":
    """ONLY called inside worker.py: load every JIT file (cache in numba_cache_dir(cache_base) if writable, else
    cache=False) and return the numba kernel and the "ref-numba" geometry."""
    files = all_jit_files()
    mods = load_jit_modules(files, numba_cache_dir(cache_base) if cache_base else None)
    kernel = kernel_from_module(mods["_chimera_kernel_numba"])
    geo_mod = mods.get("_chimera_geometry_ref")
    geometry = None
    if geo_mod is not None:
        from ..control.geometry import GEOMETRY_REF_CPYTHON
        geometry = GEOMETRY_REF_CPYTHON._replace(name="ref-numba", compute=geo_mod.geometry_compute,
                                                 resample=geo_mod.geometry_resample)
    return kernel, geometry


def load_info() -> dict:
    """What the last load_numba/load_jit_modules did: {"cache_dir": str | None, "cached": bool, "reason": str}."""
    return dict(_LOAD_INFO)


@dataclass(frozen=True)
class WarmResult:
    compiled: bool          # this worker held the lock and compiled
    waited: bool            # another worker held the lock; this one waited for its marker
    marker: dict | None     # the marker after the operation (None: no valid marker)
    seconds: float
    detail: str


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259                   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_stale(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        pid, t = int(info["pid"]), float(info["time"])
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError):
        try:
            return time.time() - os.path.getmtime(path) > 10.0      # half-written lock of a crashed writer
        except OSError:
            return False
    return (not _pid_alive(pid)) or time.time() - t > STALE_LOCK_S


def _fingerprint() -> dict:
    from importlib import metadata

    def ver(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    cpu = "|".join((platform.machine(), platform.processor(), os.environ.get("NUMBA_CPU_NAME", ""),
                    os.environ.get("NUMBA_CPU_FEATURES", "")))
    return {"content_hash": content_hash(all_jit_files()), "numba": ver("numba"), "llvmlite": ver("llvmlite"),
            "numpy": ver("numpy"), "python": platform.python_version(),
            "platform": f"{sys.platform}-{platform.machine()}", "cpu": cpu}


def _exercise(kernel: KernelImpl) -> None:
    """Call every entry point once on a tiny state (compiles them into the cache)."""
    import numpy as np

    from ..core.body import BODY_MALE
    from ..core.types import Seeds
    from .network import human_net
    from .voice import _no_rows, run
    st, idx = human_net(BODY_MALE, 96000.0, Seeds())
    from . import layout as L
    st.si[L.SI_CTRL_COUNTER] = 1     # one sample without a control step
    raw = np.zeros(1)
    run(kernel, st, idx, _no_rows(idx.n_ph + idx.n_or), 1, raw)
    kernel.output_chain(st.sc, st.si, st.dec_c, st.dec_buf, raw, 1, np.zeros(1, dtype=np.float32))
    kernel.flush_subnormals(st.nf, st.bf, st.rf, st.noise_f, st.glot, st.sc)


def warm_numba(cache_base: str) -> WarmResult:
    """Worker op "warm": take <dir>/.warm.lock (O_CREAT|O_EXCL, pid + time); if held, wait for the
    marker (poll 0.5 s, max 300 s); a stale lock (pid gone or older than 15 min) is removed and retried. The holder
    loads numba, calls every entry point once, runs selftest("numba") and only then writes marker.json atomically."""
    t0 = time.monotonic()
    d = numba_cache_dir(cache_base)
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, LOCK_NAME)
    marker_path = os.path.join(d, MARKER_NAME)
    waited = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_stale(lock):
                try:
                    os.remove(lock)
                except FileNotFoundError:
                    pass
                continue
            waited = True
            deadline = time.monotonic() + WAIT_MAX_S
            while time.monotonic() < deadline:
                m = marker_ok(cache_base)
                if m is not None:
                    return WarmResult(False, True, m, time.monotonic() - t0, "marker written by another worker")
                if not os.path.exists(lock) or _lock_stale(lock):
                    break
                time.sleep(WAIT_POLL_S)
            else:
                return WarmResult(False, True, None, time.monotonic() - t0, "timed out waiting for the warm-up lock")
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "time": time.time()}, fh)
        break
    try:
        kernel, geometry = load_numba(cache_base)
        _exercise(kernel)
        st = selftest(kernel, geometry)
        import numba
        try:
            llvm_cpu = numba.config.CPU_NAME or __import__("llvmlite.binding", fromlist=["x"]).get_host_cpu_name()
        except Exception:  # noqa: BLE001 - informational only
            llvm_cpu = None
        marker = _fingerprint() | {"llvm_cpu": llvm_cpu, "created": time.time(),
                                    "selftest": {"passed": st.passed, "max_rel_err": st.max_rel_err,
                                                 "seconds": st.seconds, "detail": st.detail}}
        tmp = marker_path + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(marker, fh)
        replace_retrying(tmp, marker_path)
        return WarmResult(True, waited, marker if st.passed else None, time.monotonic() - t0,
                          "compiled" if st.passed else f"selftest failed: {st.detail}")
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


REPLACE_TRIES = 20
REPLACE_DELAY_S = 0.05


def replace_retrying(src: str, dst: str) -> None:
    """os.replace that rides out a reader: on Windows replacing a file another process has open (the host reads the
    marker on every request, waiting workers poll it) fails with PermissionError until the reader closes it."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_DELAY_S * (attempt + 1))


def marker_ok(cache_base: str) -> dict | None:
    """The marker if it exists, its selftest passed and it matches this interpreter (content hash, numba, llvmlite and
    numpy versions from package metadata, Python, platform, CPU); else None. Never imports numba."""
    try:
        with open(os.path.join(numba_cache_dir(cache_base), MARKER_NAME), encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict):
        return None
    for k, v in _fingerprint().items():
        if m.get(k) != v:
            return None
    st = m.get("selftest")
    if not isinstance(st, dict) or st.get("passed") is not True:
        return None
    return m


@dataclass(frozen=True)
class SelfTestResult:
    backend: str
    passed: bool
    max_rel_err: float
    seconds: float
    detail: str


def selftest_scenario() -> dict:
    """The self-test input from selftest.json: {"body", "fs_out", "oversample", "source", "artic", "gate", "samples",
    "block"} (= R1 held_vowel("ɑ"): set_source(f0 110), set_articulation(vowel_artic("ɑ", male)), set_gate(1.0),
    4 800 samples in host blocks of 128)."""
    d = json.loads((PACKAGE_DIR / "dsp" / SELFTEST_FILE).read_text(encoding="utf-8"))
    sc = d["scenario"]
    return {"body": sc["body"], "fs_out": sc["fs_out"], "oversample": sc["oversample"], "source": {"f0": sc["f0"]},
            "artic": sc["artic"], "gate": sc["gate"]["level"], "samples": sc["samples"], "block": sc["host_block"]}


def selftest(kernel: KernelImpl, geometry: "GeometryImpl | None") -> SelfTestResult:
    """Render 4 800 samples of sustained /ɑ/ male 110 Hz (the scenario in selftest.json) and compare with the
    embedded expected samples; pass if max|Δ| ≤ 1e-6 × peak. A miscompiled kernel or an incompatible numba/numpy
    pair would otherwise only show as wrong sound."""
    import numpy as np

    from ..core.body import BodyMix
    from .voice import HumanVoice
    t0 = time.perf_counter()
    try:
        d = json.loads((PACKAGE_DIR / "dsp" / SELFTEST_FILE).read_text(encoding="utf-8"))
        ref = np.asarray(d["expected"], dtype=np.float64).astype(np.float32).astype(np.float64)
        sc = selftest_scenario()
        v = HumanVoice(float(sc["fs_out"]), BodyMix.parse(sc["body"]), int(sc["oversample"]), kernel=kernel,
                       geometry=geometry)
        v.set_source(sc["source"])
        v.set_articulation(sc["artic"], None)
        v.set_gate(sc["gate"])
        n, block = int(sc["samples"]), int(sc["block"])
        out = np.zeros(n, dtype=np.float32)
        for i in range(0, n, block):
            v.render(min(block, n - i), out[i:i + block])
        peak = float(np.max(np.abs(ref)))
        err = float(np.max(np.abs(out.astype(np.float64) - ref))) / peak
        if not np.isfinite(err):
            err = float("inf")
        passed = err <= SELFTEST_TOL
        detail = f"max |delta| {err:.3e} x peak over {n} samples"
    except Exception as e:  # noqa: BLE001 - a broken kernel must report, not raise
        passed, err, detail = False, float("inf"), f"{type(e).__name__}: {e}"
    return SelfTestResult(kernel.name, passed, err, time.perf_counter() - t0, detail)
