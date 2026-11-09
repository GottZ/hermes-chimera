"""Fixtures of the acoustics tests (tests/dsp). Helpers live next to the tests and are imported by module name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(scope="session")
def numba_cache(tmp_path_factory) -> Path:
    """One numba cache base for the session: kernels compile once, later subprocesses load from it."""
    return tmp_path_factory.mktemp("numba-cache")


@pytest.fixture
def run(numba_cache, tmp_path):
    """run(kernel_name, case, **args) -> dict of arrays (numpy in-process, numba in a subprocess)."""
    import dsp_numba

    def _run(kernel_name: str, case: str, **args):
        return dsp_numba.run_case(kernel_name, case, cache_dir=numba_cache, tmp_path=tmp_path, **args)

    return _run
