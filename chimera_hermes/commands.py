"""``/chimera`` (slash) and ``hermes chimera`` (CLI) (INTERFACES §12). Read-only towards Hermes' own
configuration: the only write is the plugin's own setting ``voice`` through ``ctx.set_config``.

S1: signatures only.
"""

from __future__ import annotations

from typing import Any, Callable

SLASH_DESCRIPTION = "Chimera voices: status, voices, voice <id>, warmup, test"
SLASH_ARGS_HINT = "[status|voices|voice <id>|warmup|test]"
CLI_HELP = "Chimera vocal synth"
COMPANION = "chimera-numba"


def install_source() -> str:
    """Repository chimera was installed from (companion id ``<source>#accel``)."""
    raise NotImplementedError


def companion_hint() -> str:
    """How to install chimera-numba (quoted ``'<source>#accel'``, --yes-deps, Desktop path, restart)."""
    raise NotImplementedError


def status_text(ctx: Any) -> str:
    """numpy in this process, numba installed/marker/state, ffmpeg, tts.provider (read-only), resolved voice,
    workers, cache dir; restart vs. reinstall hints (P1-19); companion id while numba is absent."""
    raise NotImplementedError


def voices_text() -> str:
    raise NotImplementedError


def handle_slash(raw_args: str, *, ctx: Any = None) -> str:
    """``/chimera [status|voices|voice <id>|warmup|test]``; never raises."""
    raise NotImplementedError


def handle_slash_factory(ctx: Any) -> Callable[[str], str]:
    """The handler registered with ``ctx.register_command`` (one positional ``raw_args``)."""
    raise NotImplementedError


def setup_cli(subparser: Any) -> None:
    """``hermes chimera status | voices | warmup | selftest | say TEXT [--voice] [--out] | sing SCORE.json [--out]
    | check SCORE.json``."""
    raise NotImplementedError


def run_cli(args: Any) -> int:
    raise NotImplementedError
