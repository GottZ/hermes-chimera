"""Process-wide registry under the fixed module name ``_chimera_runtime``.

Every copy of the plugin in one process (Hermes loads one per profile scope) with the same key - package version and
hash of the package's code - shares one Engine/RenderScheduler/JobRegistry; another key gets its own. No numpy import.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from . import Engine, EngineConfig
    from .scheduler import RenderScheduler

REGISTRY_MODULE = "_chimera_runtime"
_KEY: str | None = None


@dataclass
class Runtime:
    engine: "Engine"
    scheduler: "RenderScheduler"
    lock: threading.Lock = field(default_factory=threading.Lock)
    jobs: Any = None                # the plugin's job registry; the Hermes adapter attaches it once under ``lock``


def _registry() -> types.ModuleType:
    mod = sys.modules.get(REGISTRY_MODULE)
    if mod is None:
        new = types.ModuleType(REGISTRY_MODULE, "chimera runtime registry (shared by all plugin copies)")
        new.lock = threading.RLock()
        new.runtimes = {}
        mod = sys.modules.setdefault(REGISTRY_MODULE, new)
    return mod


def package_key() -> str:
    """``<__version__>-<sha256 over all .py files of this chimera_vocal package>[:16]``."""
    global _KEY
    if _KEY is None:
        from . import __version__
        root = os.path.dirname(os.path.abspath(__file__))
        files = []
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for n in names:
                if n.endswith(".py"):
                    p = os.path.join(d, n)
                    files.append((os.path.relpath(p, root).replace(os.sep, "/"), p))
        h = hashlib.sha256()
        for rel, p in sorted(files):
            h.update(rel.encode("utf-8") + b"\0")
            with open(p, "rb") as f:
                h.update(f.read())
            h.update(b"\0")
        _KEY = f"{__version__}-{h.hexdigest()[:16]}"
    return _KEY


def runtime() -> Runtime:
    return _runtime(None)


def existing() -> Runtime | None:
    """The registered Runtime of this package version, or None when nothing in this process has used the engine yet.
    Creates nothing (and hashes nothing when no copy has registered anything)."""
    reg = sys.modules.get(REGISTRY_MODULE)
    if reg is None:
        return None
    key = package_key()
    with reg.lock:
        return reg.runtimes.get(key)


def _runtime(config: "EngineConfig | None") -> Runtime:
    """runtime() with the config used if the Runtime is created now (get_engine: first config wins)."""
    reg = _registry()
    key = package_key()
    with reg.lock:
        rt = reg.runtimes.get(key)
        if rt is None:
            from . import Engine
            eng = Engine(config)
            rt = Runtime(engine=eng, scheduler=eng.scheduler, lock=threading.Lock(), jobs=None)
            reg.runtimes[key] = rt
        return rt


def _replace(config: "EngineConfig") -> Runtime:
    """reconfigure(): shut the registered engine down and register a new Runtime for this key."""
    from . import Engine
    reg = _registry()
    key = package_key()
    with reg.lock:
        old = reg.runtimes.get(key)
        eng = Engine(config)
        rt = Runtime(engine=eng, scheduler=eng.scheduler, lock=old.lock if old else threading.Lock(),
                     jobs=old.jobs if old else None)
        reg.runtimes[key] = rt
    if old is not None:
        old.engine.shutdown()
    return rt
