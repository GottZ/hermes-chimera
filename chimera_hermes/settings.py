"""Configuration: voice, speaking rate, engine settings and Hermes' tool time limit.

The plugin reads Hermes' own configuration but never writes it; its only writes go to its own settings
(``plugins.entries.chimera.settings``) through the PluginContext.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from typing import Any

from . import _vocal

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "numba", "numpy")
DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0           # Hermes' default tool deadline
_WINDOWS_MAX_WAIT_S = (0xFFFFFFFF - 1000) / 1000.0  # Hermes caps "unbounded" at a DWORD of milliseconds on Windows
MAX_SAFE_TIMEOUT_S = _WINDOWS_MAX_WAIT_S if sys.platform == "win32" else 31_536_000.0

_warned: set = set()
_warned_lock = threading.Lock()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(msg, *args)


def setting(ctx: Any, key: str, default: Any = None) -> Any:
    """Plugin setting ``plugins.entries.chimera.settings.<key>``; any failure gives ``default``."""
    if ctx is None:
        return default
    try:
        value = ctx.get_config(key, default)
    except Exception:  # Hermes rejects odd keys with ValueError; a broken config must not break speech
        return default
    return default if value is None else value


def core_config() -> dict:
    """Read-only view of Hermes' config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def core_tts() -> dict:
    tts = core_config().get("tts")
    return tts if isinstance(tts, dict) else {}


def resolve_voice(ctx: Any, hermes_voice: str | None) -> Any:
    """Plugin setting ``voice``, else ``hermes_voice`` (Hermes passes ``tts.voice``), else the default preset.

    The specific setting wins; ``tts.voice`` may hold another provider's voice ("alloy") and is then skipped with one
    warning per process and value."""
    return resolve_voice_spec(ctx, hermes_voice)[1]


def resolve_voice_spec(ctx: Any, hermes_voice: str | None) -> tuple[str, Any]:
    """(voice spec text, preset) for ``resolve_voice``. Hand the engine the text, not the preset: Hermes imports the
    plugin once per profile and every copy uses the engine of the first one, which only knows its own preset class."""
    presets = _vocal.api().presets
    for source, value in (("plugin setting 'voice'", setting(ctx, "voice")), ("tts.voice", hermes_voice)):
        if value is None or value == "":
            continue
        if isinstance(value, str) and value.strip():
            preset = presets.parse_voice_spec(value.strip())
            if preset is not None:
                return value.strip(), preset
        _warn_once(f"voice:{source}:{value!r}",
                   "chimera: ignoring voice %r from %s (not a chimera voice; see /chimera voices)", value, source)
    return presets.DEFAULT_VOICE, presets.PRESETS[presets.DEFAULT_VOICE]


def _finite_positive(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) and x > 0 else None


def resolve_speed(hermes_speed: float | None, preset: Any) -> float:
    """Absolute speaking rate: clamp(speed × preset rate, 0.5, 1.8); a missing or invalid speed counts as 1."""
    rate = _finite_positive(getattr(getattr(preset, "speech", None), "rate", 1.0)) or 1.0
    speed = _finite_positive(hermes_speed) or 1.0
    return min(1.8, max(0.5, speed * rate))


def faithful(ctx: Any) -> bool:
    """Plugin setting ``faithful``: speak like the template's ``chimera render text`` command (see Engine.speak)."""
    v = setting(ctx, "faithful", False)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return v is True


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        v = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return v if lo <= v <= hi else default


def engine_config(ctx: Any) -> Any:
    """EngineConfig from the plugin settings; CHIMERA_BACKEND / CHIMERA_WORKERS in the environment win; the numba
    cache lives under <HERMES_HOME>/cache/chimera/numba."""
    from . import paths

    api = _vocal.api()
    backend = setting(ctx, "backend", "auto")
    if backend not in BACKENDS:
        _warn_once(f"backend:{backend!r}", "chimera: unknown backend %r in the plugin settings; using auto", backend)
        backend = "auto"
    env_backend = os.environ.get("CHIMERA_BACKEND", "").strip().lower()
    if env_backend in BACKENDS:
        backend = env_backend
    workers = _int(setting(ctx, "workers", 0), 0, 0, 64)
    env_workers = os.environ.get("CHIMERA_WORKERS", "").strip()
    if env_workers:
        workers = _int(env_workers, workers, 0, 64)
    text_prep = setting(ctx, "text_prep", True)
    max_voices = _int(setting(ctx, "max_voices", 8), 8, 1, 64)
    return api.EngineConfig(backend=backend, workers=workers,
                            text_prep=text_prep if isinstance(text_prep, bool) else True,
                            max_voices=max_voices, cache_dir=str(paths.numba_cache_dir()))


# Hermes' tool deadline, resolved exactly like hermes-agent agent/deadline.py resolve_timeout (not a public module):
# config value (a number, not bool/NaN) > environment variable > default; <= 0 or NaN means unbounded.

def _clamp(timeout: Any) -> float | None:
    if timeout is None:
        return None
    try:
        v = float(timeout)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return None if v <= 0 else min(v, MAX_SAFE_TIMEOUT_S)


def _lookup(section: Any, dotted: str) -> Any:
    node = section
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve(section: dict, key: str, *, default: float | None, env_var: str | None = None) -> float | None:
    raw = _lookup(section, key)
    if raw is not None and not isinstance(raw, bool):
        try:
            v = float(raw)
            if v == v:
                return _clamp(v)
        except (TypeError, ValueError):
            pass
    if env_var:
        env_raw = os.getenv(env_var, "").strip()
        if env_raw:
            try:
                return _clamp(float(env_raw))
            except ValueError:
                pass
    return _clamp(default)


def tool_deadline_s() -> float | None:
    """The tool time limit this call runs under: the smaller of Hermes' concurrent-batch and sequential-call limits
    (a plugin cannot tell which path called it); None = unbounded."""
    section = core_config().get("timeouts")
    section = section if isinstance(section, dict) else {}
    concurrent = _resolve(section, "tools.concurrent_batch", default=DEFAULT_CONCURRENT_TOOL_TIMEOUT_S,
                          env_var="HERMES_CONCURRENT_TOOL_TIMEOUT_S")
    sequential = _resolve(section, "tools.sequential_call", default=concurrent)
    vals = [v for v in (concurrent, sequential) if v is not None]
    return min(vals) if vals else None
