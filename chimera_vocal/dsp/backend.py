"""Kernel choice support: numba loading ONLY inside worker processes, content-addressed cache, warm-up lock and
marker, self-test (INTERFACES §6.4a, §6.6; DESIGN §2.6, §2.7).

The host (Hermes) process may import this module and call ``marker_ok``/``numba_cache_dir``: neither imports numba
(K-3.17). ``load_numba``/``warm_numba`` import numba and are called only from ``worker.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from ..core.jit import JitFile
from .kernel_numpy import KernelImpl

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
    raise NotImplementedError


def content_hash(files: tuple[JitFile, ...]) -> str:
    """sha256 over the bytes of the files (paths relative to the chimera_vocal package), in order; hex."""
    raise NotImplementedError


def numba_cache_dir(base: str) -> str:
    """<base>/<content_hash(all_jit_files())[:16]>; creates nothing."""
    raise NotImplementedError


def load_jit_modules(files: tuple[JitFile, ...], cache_dir: str | None) -> dict[str, ModuleType]:
    """Import every file under its fixed name with CHIMERA_JIT = numba.njit(cache=<cache_dir is writable>, nogil=True,
    fastmath=False, error_model="numpy"); numba.config.CACHE_DIR is set only around the imports and restored.
    Returns {fixed_name: module}. Worker only."""
    raise NotImplementedError


def kernel_from_module(mod: ModuleType) -> KernelImpl:
    """KernelImpl("numba", ...) from a loaded ``_chimera_kernel_numba`` module."""
    raise NotImplementedError


def load_numba(cache_base: str | None) -> "tuple[KernelImpl, GeometryImpl]":
    """ONLY called inside worker.py: load every JIT file (cache in numba_cache_dir(cache_base) if writable, else
    cache=False) and return the numba kernel and the "ref-numba" geometry."""
    raise NotImplementedError


def load_info() -> dict:
    """What the last load_numba/load_jit_modules did: {"cache_dir": str | None, "cached": bool, "reason": str}."""
    raise NotImplementedError


@dataclass(frozen=True)
class WarmResult:
    compiled: bool          # this worker held the lock and compiled
    waited: bool            # another worker held the lock; this one waited for its marker
    marker: dict | None     # the marker after the operation (None: no valid marker)
    seconds: float
    detail: str


def warm_numba(cache_base: str) -> WarmResult:
    """Worker op "warm" (INTERFACES §6.4a): take <dir>/.warm.lock (O_CREAT|O_EXCL, pid + time); if held, wait for the
    marker (poll 0.5 s, max 300 s); a stale lock (pid gone or older than 15 min) is removed and retried. The holder
    loads numba, calls every entry point once, runs selftest("numba") and only then writes marker.json atomically."""
    raise NotImplementedError


def marker_ok(cache_base: str) -> dict | None:
    """The marker if it exists, its selftest passed and it matches this interpreter (content hash, numba, llvmlite and
    numpy versions from package metadata, Python, platform, CPU); else None. Never imports numba."""
    raise NotImplementedError


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
    raise NotImplementedError


def selftest(kernel: KernelImpl, geometry: "GeometryImpl | None") -> SelfTestResult:
    """Render 4 800 samples of sustained /ɑ/ male 110 Hz (the scenario in selftest.json) and compare with the
    embedded expected samples; pass if max|Δ| ≤ 1e-6 × peak (INTERFACES §6.6)."""
    raise NotImplementedError
