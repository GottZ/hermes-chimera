"""Shared pytest configuration.

* markers (registered in pyproject.toml): ``slow``, ``numba``, ``lab``, ``hermes`` -- skipped automatically
  when their precondition is missing:
    - ``slow``   runs only with ``--run-slow`` or ``CHIMERA_RUN_SLOW=1``
    - ``numba``  runs only when numba is installed (checked with find_spec: this process never imports numba;
                 numba code runs in worker subprocesses)
    - ``lab``    runs only when ``CHIMERA_LAB_REFS`` names a directory (full-length references, not in the repository)
    - ``hermes`` runs only when ``HERMES_UPSTREAM`` names a Hermes checkout
* fixture ``kernel_name``: parametrizes a test over the two kernels ("numpy", "numba"); the numba case
  carries the ``numba`` marker.
* fixtures ``lab_refs``, ``golden``, ``vector``: see tests/support/__init__.py.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

import support


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")


def _numba_installed() -> bool:
    return importlib.util.find_spec("numba") is not None


def _hermes_upstream() -> Path | None:
    p = os.environ.get("HERMES_UPSTREAM")
    return Path(p) if p and Path(p).is_dir() else None


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow") or os.environ.get("CHIMERA_RUN_SLOW") == "1"
    skips = {
        "slow": None if run_slow else pytest.mark.skip(reason="slow: use --run-slow or CHIMERA_RUN_SLOW=1"),
        "numba": None if _numba_installed() else pytest.mark.skip(reason="numba not installed"),
        "lab": None if support.lab_refs() else pytest.mark.skip(reason="CHIMERA_LAB_REFS not set"),
        "hermes": None if _hermes_upstream() else pytest.mark.skip(reason="HERMES_UPSTREAM not set"),
    }
    for item in items:
        for name, mark in skips.items():
            if mark is not None and item.get_closest_marker(name) is not None:
                item.add_marker(mark)


KERNEL_PARAMS = [pytest.param("numpy", id="numpy"), pytest.param("numba", id="numba", marks=pytest.mark.numba)]


@pytest.fixture(params=KERNEL_PARAMS)
def kernel_name(request) -> str:
    """"numpy" or "numba" (the numba case is marked ``numba`` and skipped without numba)."""
    return request.param


@pytest.fixture(scope="session")
def lab_refs() -> Path:
    p = support.lab_refs()
    if p is None:
        pytest.skip("CHIMERA_LAB_REFS not set")
    return p


@pytest.fixture(scope="session")
def golden():
    return support.golden


@pytest.fixture(scope="session")
def vector():
    return support.vector


@pytest.fixture(scope="session")
def hermes_upstream() -> Path:
    p = _hermes_upstream()
    if p is None:
        pytest.skip("HERMES_UPSTREAM not set")
    return p
