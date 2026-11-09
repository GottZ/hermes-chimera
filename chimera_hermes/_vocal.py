"""C5-internal seam between the Hermes binding and the engine package (not part of INTERFACES).

Every engine access of ``chimera_hermes`` goes through this module, always as ``_vocal.api()`` /
``_vocal.engine()`` (module attribute lookup, never ``from ._vocal import api``) so the tests can replace ``api``
with a fake engine namespace (tests/hermes/conftest.py). Imports are lazy and relative (INTERFACES §1 rules 2, 3).

S1: signatures only.
"""

from __future__ import annotations

from typing import Any

NUMPY_MISSING = ("numpy is not available in this Hermes process - if chimera was just installed, restart Hermes; "
                 "otherwise reinstall chimera with its dependencies (hermes plugins install <source> --yes-deps)")


def bind(ctx: Any) -> None:
    """Remember the PluginContext of this plugin copy (tool handlers and the CLI get no ctx argument)."""
    raise NotImplementedError


def ctx() -> Any:
    """The bound PluginContext (None before ``register``)."""
    raise NotImplementedError


def numpy_ok() -> bool:
    """numpy importable in this process (``importlib.util.find_spec`` only, never imports it)."""
    raise NotImplementedError


def numba_installed() -> bool:
    """numba installed (find_spec only: the Hermes process never imports numba, INTERFACES §6.4a rule 5)."""
    raise NotImplementedError


def api() -> Any:
    """The ``chimera_vocal`` package (relative import) with its ``presets`` and ``runtime`` submodules loaded."""
    raise NotImplementedError


def engine(c: Any = None) -> Any:
    """The process-wide Engine: ``api().get_engine(settings.engine_config(ctx))``."""
    raise NotImplementedError


def jobs() -> Any:
    """The process-wide JobRegistry, kept on the runtime registry (INTERFACES §9.4; request C5-1)."""
    raise NotImplementedError
