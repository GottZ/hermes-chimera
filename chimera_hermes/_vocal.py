"""The one place where the Hermes binding reaches the engine package ``chimera_vocal``.

Call these as ``_vocal.api()`` / ``_vocal.engine()`` (module attribute lookup, never ``from ._vocal import api``) so
tests can replace ``api`` with a fake engine. Imports are relative because Hermes loads the plugin under a synthetic
package name (``hermes_plugins.<name>``), and lazy so that registering the plugin never loads numpy.
"""

from __future__ import annotations

import importlib.util
from typing import Any

NUMPY_MISSING = ("numpy is not available in this Hermes process - if chimera was just installed, restart Hermes; "
                 "otherwise reinstall chimera with its dependencies (hermes plugins install <source> --yes-deps)")

_CTX: Any = None


def bind(ctx: Any) -> None:
    """Remember the PluginContext of this plugin copy (tool handlers and the CLI get no ctx argument)."""
    global _CTX
    _CTX = ctx


def ctx() -> Any:
    return _CTX


def numpy_ok() -> bool:
    """numpy importable in this process; looks it up without importing it."""
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ValueError):
        return False


def numba_installed() -> bool:
    """numba installed. Looked up, never imported: numba runs only in render worker processes, so a fault in
    compiled code cannot take the Hermes process down."""
    try:
        return importlib.util.find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


def api() -> Any:
    """The ``chimera_vocal`` package with its ``presets`` and ``runtime`` submodules loaded."""
    from .. import chimera_vocal
    from ..chimera_vocal import presets, runtime  # noqa: F401  (loads the submodules as attributes)

    return chimera_vocal


def engine(c: Any = None) -> Any:
    """The process-wide Engine, configured from this plugin's settings."""
    from . import settings

    return api().get_engine(settings.engine_config(c if c is not None else _CTX))


def jobs() -> Any:
    """The process-wide JobRegistry. It lives on the engine's runtime registry so every copy of the plugin in one
    process (Hermes loads one per profile scope) sees the same jobs; it is created here because the engine package
    never imports the Hermes binding."""
    from .jobs import JobRegistry

    rt = api().runtime.runtime()
    with rt.lock:
        if rt.jobs is None:
            rt.jobs = JobRegistry()
        return rt.jobs
