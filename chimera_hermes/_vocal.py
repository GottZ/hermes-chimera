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
    """The process-wide Engine with this plugin's current settings applied: a changed ``backend``, ``workers``,
    ``text_prep`` or ``max_voices`` takes effect with the next render, without restarting Hermes."""
    return engine_and_config(c)[0]


def engine_and_config(c: Any = None) -> tuple[Any, Any]:
    """(process-wide Engine, EngineConfig from the settings of the profile asking now). Every profile of a Hermes
    process shares the one engine, whose own settings are those applied last; a request therefore carries its
    profile's settings itself (``speech_options``/``score_options``), resolved here on the caller's thread."""
    from . import settings

    cfg = settings.engine_config(c if c is not None else _CTX)
    eng = api().get_engine(cfg)
    apply = getattr(eng, "apply_settings", None)
    if apply is not None:
        apply(cfg)
    return eng, cfg


def speech_options(cfg: Any) -> dict:
    """Per-request engine settings for speech, frozen when the request is made."""
    return {"text_prep": bool(cfg.text_prep), "backend": cfg.backend}


def score_options(cfg: Any) -> dict:
    """Per-request engine settings for a score, frozen when the request is made."""
    return {"max_voices": int(cfg.max_voices), "backend": cfg.backend}


def is_error(e: BaseException, name: str) -> bool:
    """``e`` is the engine error class ``name`` (ScoreError, TextError, ...). Matched by class name, not identity:
    every copy of the plugin in one process (Hermes loads one per profile) shares the engine of the first copy, so
    its exceptions are instances of that copy's classes."""
    return any(c.__name__ == name for c in type(e).__mro__)


def jobs_if_any() -> Any:
    """The process-wide JobRegistry if a copy of the plugin in this process has created it, else None; creates no
    engine (for session hooks and ``/chimera cancel``)."""
    try:
        rt = api().runtime.existing()
    except Exception:  # noqa: BLE001 - engine package unavailable: no jobs either
        return None
    return getattr(rt, "jobs", None) if rt is not None else None


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
