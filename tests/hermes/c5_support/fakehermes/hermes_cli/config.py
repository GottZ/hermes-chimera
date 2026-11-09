"""Fake ``hermes_cli.config``: ``load_config_readonly()`` returns the module attribute CONFIG (hermes_cli/config.py:1985).
Write helpers record into WRITES so tests can assert that the plugin never writes core configuration."""

from __future__ import annotations

from typing import Any

CONFIG: dict = {}
WRITES: list = []


def load_config_readonly() -> dict[str, Any]:
    return CONFIG


def load_config() -> dict[str, Any]:
    import copy
    return copy.deepcopy(CONFIG)


def set_config_value(key: str, value: Any) -> None:
    WRITES.append(("set_config_value", key, value))


def save_config(config: dict) -> None:
    WRITES.append(("save_config", config))
