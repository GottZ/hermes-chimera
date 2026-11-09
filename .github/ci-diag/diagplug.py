"""Temporary CI diagnostic (pytest plugin, -p diagplug): DIAG_TOL overrides NUMBA_TOL of test_geometry_numba;
DIAG_LIBM_HYPOT=1 makes this CPython process use the C library's hypot - the one numba-compiled code calls
(numba/cpython/mathimpl.py: "hypot", "_hypot" on Windows) - instead of CPython's own math.hypot algorithm."""

import ctypes
import ctypes.util
import math
import os
import sys

if os.environ.get("DIAG_LIBM_HYPOT") == "1":
    if sys.platform == "win32":
        _lib = ctypes.CDLL("ucrtbase")
        _f = _lib._hypot
    else:
        _lib = ctypes.CDLL(ctypes.util.find_library("m") or ctypes.util.find_library("c"))
        _f = _lib.hypot
    _f.restype = ctypes.c_double
    _f.argtypes = [ctypes.c_double, ctypes.c_double]
    math.hypot = lambda x, y: _f(x, y)


def pytest_collection_modifyitems(items):
    for it in items:
        if hasattr(it.module, "NUMBA_TOL") and "DIAG_TOL" in os.environ:
            it.module.NUMBA_TOL = float(os.environ["DIAG_TOL"])
