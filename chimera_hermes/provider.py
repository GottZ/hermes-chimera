"""TTS provider ``chimera`` (INTERFACES §10; DESIGN §5).

``agent.tts_provider`` is imported inside ``build_provider`` (INTERFACES §1 rule 2); the registered class combines the
mixin ``ChimeraTTSProvider`` with Hermes' ``TTSProvider`` there.

S1: signatures only.
"""

from __future__ import annotations

from typing import Any, Iterator

VOICE_BUBBLE_PLATFORMS = frozenset({"telegram", "discord", "matrix", "feishu", "whatsapp", "signal"})
HOST_PROCESS_ENV = "HERMES_PLUGIN_HOST_PROCESS"      # hermes_cli/plugin_isolation.py:97
STREAM_SAMPLE_RATE = 48000


class ChimeraTTSProvider:
    stream_sample_rate = STREAM_SAMPLE_RATE

    def __init__(self, ctx: Any) -> None:
        raise NotImplementedError

    @property
    def streams_pcm(self) -> bool:
        """False when os.environ["HERMES_PLUGIN_HOST_PROCESS"] == "1" (plugins.isolation: host, P1-10)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        raise NotImplementedError

    def is_available(self) -> bool:
        """find_spec("numpy") is not None; cached; never raises."""
        raise NotImplementedError

    def get_setup_schema(self) -> dict:
        raise NotImplementedError

    def list_voices(self) -> list[dict]:
        raise NotImplementedError

    def default_voice(self) -> str:
        raise NotImplementedError

    @property
    def voice_compatible(self) -> bool:
        """get_session_env("HERMES_SESSION_PLATFORM", "").lower() in VOICE_BUBBLE_PLATFORMS."""
        raise NotImplementedError

    def synthesize(self, text: str, output_path: str, *, voice: str | None = None, model: str | None = None,
                   speed: float | None = None, format: str = "mp3", **extra: Any) -> str:
        """WAV PCM16 mono 48 kHz at output_path if it ends with .ogg, else at <stem>.wav; returns that path.
        numpy missing -> FileNotFoundError; everything else -> RuntimeError (never ValueError)."""
        raise NotImplementedError

    def stream(self, text: str, *, voice: str | None = None, model: str | None = None, format: str = "opus",
               **extra: Any) -> Iterator[bytes]:
        """format != "pcm" -> NotImplementedError; int16 LE mono @ 48 kHz; thread-safe; the generator's finally sets
        the cancel event of its jobs and closes the engine stream (P1-08)."""
        raise NotImplementedError

    def warm(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


def build_provider(ctx: Any) -> "ChimeraTTSProvider":
    raise NotImplementedError
