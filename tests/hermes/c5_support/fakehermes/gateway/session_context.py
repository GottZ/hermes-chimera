"""Fake ``gateway.session_context.get_session_env`` (gateway/session_context.py:188-194): session value from ENV
(stands in for the ContextVar), else os.environ, else default."""

from __future__ import annotations

import os

ENV: dict = {}


def get_session_env(name: str, default: str = "") -> str:
    if name in ENV:
        return ENV[name]
    return os.getenv(name, default)
