"""JIT-file manifest type. No numba import here: the loader lives in dsp/backend.py
and runs only inside worker processes.

Rules for every JIT file: first statement after the imports is
``jit = globals().get("CHIMERA_JIT", lambda f: f)``; every compiled function carries ``@jit``; imports are
limited to ``math``, ``numpy`` and ``import _chimera_nbmath as nbm`` guarded by
``try/except ImportError: from ..core import nbmath as nbm``; entry points and all ``@jit`` functions take
only numpy arrays and Python scalars (no package classes); no class definitions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JitFile:
    relpath: str                   # path inside the chimera_vocal package, "/"-separated
    fixed_name: str                # module name used by the worker loader, e.g. "_chimera_geometry_ref"
    entry_points: tuple[str, ...]  # functions the loader returns (all others are internal)


CORE_JIT_FILES: tuple[JitFile, ...] = (JitFile("core/nbmath.py", "_chimera_nbmath", ()),)

JIT_PLACEHOLDER_LINE = 'jit = globals().get("CHIMERA_JIT", lambda f: f)'
