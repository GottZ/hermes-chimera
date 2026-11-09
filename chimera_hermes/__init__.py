"""Hermes binding of the chimera vocal synth: TTS provider, music tools, /chimera command, music skill.

``register`` runs in every Hermes process (gateway, CLI, desktop backend, plugin host), so it imports only light
modules, starts no threads and does no I/O beyond resolving paths inside the plugin directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SKILL_DESCRIPTION = ("Write scores for chimera_render: voices, ranges, lyrics per note, canon/choir examples, "
                     "MEDIA delivery.")
# Hermes does not read the frontmatter of a plugin skill file; it only keeps what register_skill receives.
SKILL_FRONTMATTER = {"name": "music", "description": SKILL_DESCRIPTION, "version": "0.1.0",
                     "metadata": {"hermes": {"tags": ["music", "singing", "audio"],
                                             "requires_tools": ["chimera_render"]}}}


def skill_path() -> Path:
    """``<plugin root>/skills/music/SKILL.md``."""
    return Path(__file__).resolve().parent.parent / "skills" / "music" / "SKILL.md"


def register(ctx: Any) -> None:
    from . import _vocal
    from .commands import CLI_HELP, SLASH_ARGS_HINT, SLASH_DESCRIPTION, handle_slash_factory, run_cli, setup_cli
    from .provider import build_provider
    from .tools import HANDLERS, SCHEMAS, check_available

    _vocal.bind(ctx)
    ctx.register_tts_provider(build_provider(ctx))
    for name, schema in SCHEMAS.items():
        ctx.register_tool(name=name, toolset="chimera", schema=schema, handler=HANDLERS[name],
                          check_fn=check_available)
    ctx.register_command("chimera", handle_slash_factory(ctx), description=SLASH_DESCRIPTION,
                         args_hint=SLASH_ARGS_HINT)
    ctx.register_cli_command("chimera", help=CLI_HELP, setup_fn=setup_cli, handler_fn=run_cli)
    ctx.register_skill("music", str(skill_path()), description=SKILL_DESCRIPTION, frontmatter=SKILL_FRONTMATTER)
