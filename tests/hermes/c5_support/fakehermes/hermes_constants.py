"""Fake ``hermes_constants`` (hermes_constants.py:194, 498-510): HERMES_HOME from the module attribute HOME."""

from __future__ import annotations

from pathlib import Path

HOME: Path = Path("/nonexistent-hermes-home")


def get_hermes_home() -> Path:
    return Path(HOME)


def _legacy_path_has_content(p: Path) -> bool:
    try:
        return p.is_dir() and any(p.iterdir())
    except OSError:
        return False


def get_hermes_dir(new_subpath: str, old_name: str, *, home: Path | None = None) -> Path:
    home = Path(home) if home is not None else get_hermes_home()
    old_path = home / old_name
    return old_path if _legacy_path_has_content(old_path) else home / new_subpath
