"""Process-wide registry under the fixed module name ``_chimera_runtime`` (INTERFACES §9.4, P1-25).

Every copy of the plugin in one process (Hermes loads one per profile scope) with the same key - package version and
hash of the package's code - shares one Engine/RenderScheduler/JobRegistry; another key gets its own. No numpy import.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from . import Engine, EngineConfig
    from .scheduler import RenderScheduler

REGISTRY_MODULE = "_chimera_runtime"


@dataclass
class Runtime:
    engine: "Engine"
    scheduler: "RenderScheduler"
    jobs: Any                       # chimera_hermes JobRegistry, attached by C5 under ``lock`` (request C4-3); None first
    lock: threading.Lock = field(default_factory=threading.Lock)


def package_key() -> str:
    """``<__version__>-<sha256 over all .py files of this chimera_vocal package>[:16]``."""
    raise NotImplementedError


def runtime() -> Runtime:
    raise NotImplementedError


def _runtime(config: "EngineConfig | None") -> Runtime:
    """runtime() with the config used if the Runtime is created now (get_engine: first config wins)."""
    raise NotImplementedError


def _replace(config: "EngineConfig") -> Runtime:
    """reconfigure(): shut the registered engine down and register a new Runtime for this key."""
    raise NotImplementedError
