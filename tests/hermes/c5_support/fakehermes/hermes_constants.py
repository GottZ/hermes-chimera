"""Fake ``hermes_constants`` (hermes_constants.py:18-49, 194, 498-510): HERMES_HOME from a context-local override,
else the module attribute HOME."""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path

HOME: Path = Path("/nonexistent-hermes-home")
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar = ContextVar("_HERMES_HOME_OVERRIDE", default=_UNSET)   # hermes_constants.py:18


def set_hermes_home_override(path):
    """Context-local HERMES_HOME (hermes_constants.py:32-44): how one process serves several profiles."""
    return _HERMES_HOME_OVERRIDE.set(str(path) if path else _UNSET)


def reset_hermes_home_override(token) -> None:
    _HERMES_HOME_OVERRIDE.reset(token)


def get_hermes_home() -> Path:
    override = _HERMES_HOME_OVERRIDE.get()
    return Path(override) if override is not _UNSET else Path(HOME)


def _legacy_path_has_content(p: Path) -> bool:
    try:
        return p.is_dir() and any(p.iterdir())
    except OSError:
        return False


def get_hermes_dir(new_subpath: str, old_name: str, *, home: Path | None = None) -> Path:
    home = Path(home) if home is not None else get_hermes_home()
    old_path = home / old_name
    return old_path if _legacy_path_has_content(old_path) else home / new_subpath
