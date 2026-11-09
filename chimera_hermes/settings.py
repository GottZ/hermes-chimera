"""Configuration resolution (INTERFACES §10.3): voice, speed, engine config, tool deadline.

S1: signatures only.
"""

from __future__ import annotations

from typing import Any

BACKENDS = ("auto", "numba", "numpy")
DEFAULT_CONCURRENT_TOOL_TIMEOUT_S = 420.0          # agent/tool_executor.py:139


def setting(ctx: Any, key: str, default: Any = None) -> Any:
    """Plugin setting ``plugins.entries.chimera.settings.<key>`` via ``ctx.get_config``; any failure -> default."""
    raise NotImplementedError


def core_config() -> dict:
    """Read-only view of Hermes' config (``hermes_cli.config.load_config_readonly``); never written (§12)."""
    raise NotImplementedError


def core_tts() -> dict:
    """The ``tts`` section of ``core_config()`` (dict, possibly empty)."""
    raise NotImplementedError


def resolve_voice(ctx: Any, hermes_voice: str | None) -> Any:
    """1. plugin setting ``voice`` if parse_voice_spec accepts it; 2. ``hermes_voice`` (= tts.voice) likewise;
    3. PRESETS[DEFAULT_VOICE]. Every rejected non-empty value logs one warning per process and value."""
    raise NotImplementedError


def resolve_speed(hermes_speed: float | None, preset: Any) -> float:
    """clamp(speed · preset.speech.rate, 0.5, 1.8); missing/invalid speed counts as 1."""
    raise NotImplementedError


def engine_config(ctx: Any) -> Any:
    """EngineConfig from the plugin settings backend/workers/text_prep/max_voices; env CHIMERA_BACKEND /
    CHIMERA_WORKERS win; cache_dir = <HERMES_HOME>/cache/chimera/numba."""
    raise NotImplementedError


def tool_deadline_s() -> float | None:
    """Re-implements agent.deadline.resolve_timeout (agent/deadline.py:125-137, 184-209; tool_executor.py:190-199,
    839-846): min(concurrent batch, sequential call) of the non-None values; None = unbounded (P1-18)."""
    raise NotImplementedError
