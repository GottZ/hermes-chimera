"""Text-to-speech provider ``chimera``.

Output contract with Hermes: WAV PCM16 mono 48 kHz. A path ending in ``.ogg`` gets the WAV bytes as given (Hermes
turns them into Ogg/Opus with ffmpeg, or renames honestly to ``.wav`` without it); any other path is replaced by
``<stem>.wav`` and that path is returned - WAV bytes never sit under ``.mp3``. Streaming yields raw int16 PCM at the
engine's native 48 kHz. ``agent.tts_provider`` is imported inside ``build_provider`` so that importing the plugin
never pulls Hermes internals; the registered class combines the ``ChimeraTTSProvider`` methods with Hermes' base.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterator

from . import _vocal, settings

logger = logging.getLogger(__name__)

VOICE_BUBBLE_PLATFORMS = frozenset({"telegram", "discord", "matrix", "feishu", "whatsapp", "signal"})
HOST_PROCESS_ENV = "HERMES_PLUGIN_HOST_PROCESS"      # set by Hermes in its plugin host process
STREAM_SAMPLE_RATE = 48000
STREAM_WARN_BYTES = 170 * STREAM_SAMPLE_RATE * 2     # Hermes caps one stream at 16 MiB = 174 s at 48 kHz
NOTHING_SPEAKABLE = "chimera reads English text; nothing speakable was left after cleanup"


class ChimeraTTSProvider:
    """Methods of the chimera TTS provider; ``build_provider`` puts Hermes' TTSProvider behind it."""

    stream_sample_rate = STREAM_SAMPLE_RATE

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._available: bool | None = None

    @property
    def streams_pcm(self) -> bool:
        # Under plugins.isolation: host a generator cannot cross the host connection; Hermes then speaks sentence by
        # sentence through synthesize(). Read per reply, hence a property.
        return os.environ.get(HOST_PROCESS_ENV) != "1"

    @property
    def name(self) -> str:
        return "chimera"

    @property
    def display_name(self) -> str:
        return "Chimera (local vocal synth)"

    def is_available(self) -> bool:
        if self._available is None:
            self._available = _vocal.numpy_ok()
        return self._available

    def get_setup_schema(self) -> dict:
        return {"name": self.display_name, "badge": "local · free",
                "tag": "Physically modelled English voices on your CPU", "env_vars": []}

    def list_voices(self) -> list[dict]:
        try:
            return list(_vocal.api().presets.list_voices())
        except Exception:
            return []

    def default_voice(self) -> str:
        try:
            tts_voice = settings.core_tts().get("voice")
            return settings.resolve_voice(self._ctx, tts_voice if isinstance(tts_voice, str) else None).id
        except Exception:
            return "male"

    @property
    def voice_compatible(self) -> bool:
        # True only on platforms with voice bubbles. Elsewhere Hermes would convert to Ogg/Opus with ffmpeg for
        # nothing (desktop and the Windows CLI play WAV directly). In the plugin host process the session is unknown.
        try:
            from gateway.session_context import get_session_env

            return str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower() in VOICE_BUBBLE_PLATFORMS
        except Exception:
            return False

    def _voice_and_speed(self, voice: Any, speed: Any) -> tuple[Any, float]:
        preset = settings.resolve_voice(self._ctx, voice if isinstance(voice, str) else None)
        return preset, settings.resolve_speed(speed, preset)

    def synthesize(self, text: str, output_path: str, *, voice: str | None = None, model: str | None = None,
                   speed: float | None = None, format: str = "mp3", **extra: Any) -> str:
        if not self.is_available():
            raise FileNotFoundError(_vocal.NUMPY_MISSING)
        out = str(output_path)
        path = out if out.lower().endswith(".ogg") else os.path.splitext(out)[0] + ".wav"
        try:
            api = _vocal.api()
            preset, spd = self._voice_and_speed(voice, speed)
            try:
                audio, _kernel = _vocal.engine(self._ctx).speak(text, preset, speed=spd)
            except api.TextError as e:
                raise RuntimeError(str(e) or NOTHING_SPEAKABLE) from e
            audio.to_wav(path, bits=16)
        except RuntimeError:
            raise
        except Exception as e:  # never ValueError: Hermes would report a configuration error
            raise RuntimeError(f"chimera synthesis failed: {type(e).__name__}: {e}") from e
        return path

    def stream(self, text: str, *, voice: str | None = None, model: str | None = None, format: str = "opus",
               **extra: Any) -> Iterator[bytes]:
        if format != "pcm":
            raise NotImplementedError("chimera streams raw PCM only (format='pcm'); use synthesize() for files")
        if not self.is_available():
            raise FileNotFoundError(_vocal.NUMPY_MISSING)
        preset, spd = self._voice_and_speed(voice, extra.get("speed"))
        return self._pcm(text, preset, spd)

    def _pcm(self, text: str, preset: Any, spd: float) -> Iterator[bytes]:
        # Closing this generator (the listener stopped) sets the cancel event and closes the engine stream, which
        # frees its render slot; nothing is shared between concurrent calls (Hermes prefetches up to 3 sentences).
        cancel = threading.Event()
        gen = None
        api = None
        total = 0
        try:
            try:
                api = _vocal.api()
                gen = iter(_vocal.engine(self._ctx).speak_pcm16(text, preset, speed=spd, cancel=cancel))
                for chunk in gen:
                    total += len(chunk)
                    if total > STREAM_WARN_BYTES and total - len(chunk) <= STREAM_WARN_BYTES:
                        logger.warning("chimera: one sentence longer than 170 s; Hermes cuts streams at 16 MiB")
                    yield chunk
            except RuntimeError:
                raise
            except Exception as e:
                text_error = getattr(api, "TextError", None)
                if isinstance(text_error, type) and isinstance(e, text_error):
                    raise RuntimeError(str(e) or NOTHING_SPEAKABLE) from e
                raise RuntimeError(f"chimera streaming failed: {type(e).__name__}: {e}") from e
        finally:
            cancel.set()
            close = getattr(gen, "close", None)
            if close is not None:
                close()

    def warm(self) -> None:
        if not self.is_available():
            return
        try:
            eng = _vocal.engine(self._ctx)
            eng.hold_warm(True)
            eng.warmup()
        except Exception as e:  # best effort
            logger.warning("chimera: warm-up failed: %s", e)

    def release(self) -> None:
        if not self.is_available():
            return
        try:
            _vocal.engine(self._ctx).hold_warm(False)
        except Exception as e:
            logger.warning("chimera: release failed: %s", e)


_CLASSES: dict = {}


def build_provider(ctx: Any) -> "ChimeraTTSProvider":
    from agent.tts_provider import TTSProvider

    cls = _CLASSES.get(TTSProvider)
    if cls is None:
        cls = type("ChimeraTTSProvider", (ChimeraTTSProvider, TTSProvider), {"__module__": __name__})
        _CLASSES[TTSProvider] = cls
    return cls(ctx)
