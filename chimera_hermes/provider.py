"""Text-to-speech provider ``chimera``.

Output contract with Hermes: WAV PCM16 mono 48 kHz. A path ending in ``.ogg`` or ``.wav`` gets the WAV bytes as given
(for ``.ogg`` Hermes turns them into Ogg/Opus with ffmpeg, or renames honestly to ``.wav`` without it); any other path
is replaced by ``<stem>.wav`` and that path is returned - WAV bytes never sit under ``.mp3``. An existing
``<stem>.wav`` that Hermes did not name is never overwritten: chimera then writes ``<stem>.<random>.wav``. Audio is
rendered to disk window by window, so memory does not grow with the text. Streaming yields raw int16 PCM at the
engine's native 48 kHz. ``agent.tts_provider`` is imported inside ``build_provider`` so that importing the plugin
never pulls Hermes internals; the registered class combines the ``ChimeraTTSProvider`` methods with Hermes' base.
"""

from __future__ import annotations

import logging
import os
import secrets
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

    def _voice_and_speed(self, voice: Any, speed: Any) -> tuple[str, float]:
        """(voice spec text for the engine, absolute speaking rate)."""
        spec, preset = settings.resolve_voice_spec(self._ctx, voice if isinstance(voice, str) else None)
        return spec, settings.resolve_speed(speed, preset)

    def synthesize(self, text: str, output_path: str, *, voice: str | None = None, model: str | None = None,
                   speed: float | None = None, format: str = "mp3", **extra: Any) -> str:
        if not self.is_available():
            raise FileNotFoundError(_vocal.NUMPY_MISSING)
        path, reserved = _target(str(output_path))
        try:
            spec, spd = self._voice_and_speed(voice, speed)
            eng, cfg = _vocal.engine_and_config(self._ctx)
            eng.speak_to_wav(text, path, spec, speed=spd, faithful=settings.faithful(self._ctx),
                             tmp_dir=_tmp_dir(), **_vocal.speech_options(cfg))
        except Exception as e:  # never ValueError: Hermes would report a configuration error
            if reserved:
                _remove_empty(path)
            if _vocal.is_error(e, "TextError"):
                raise RuntimeError(str(e) or NOTHING_SPEAKABLE) from e
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"chimera synthesis failed: {type(e).__name__}: {e}") from e
        return path

    def stream(self, text: str, *, voice: str | None = None, model: str | None = None, format: str = "opus",
               **extra: Any) -> Iterator[bytes]:
        if format != "pcm":
            raise NotImplementedError("chimera streams raw PCM only (format='pcm'); use synthesize() for files")
        if not self.is_available():
            raise FileNotFoundError(_vocal.NUMPY_MISSING)
        spec, spd = self._voice_and_speed(voice, extra.get("speed"))
        return self._pcm(text, spec, spd)

    def _pcm(self, text: str, spec: str, spd: float) -> Iterator[bytes]:
        # Closing this generator (the listener stopped) sets the cancel event and closes the engine stream, which
        # frees its render slot; nothing is shared between concurrent calls (Hermes prefetches up to 3 sentences).
        cancel = threading.Event()
        gen = None
        total = 0
        try:
            try:
                eng, cfg = _vocal.engine_and_config(self._ctx)
                gen = iter(eng.speak_pcm16(text, spec, speed=spd, faithful=settings.faithful(self._ctx),
                                           tmp_dir=_tmp_dir(), cancel=cancel, **_vocal.speech_options(cfg)))
                for chunk in gen:
                    total += len(chunk)
                    if total > STREAM_WARN_BYTES and total - len(chunk) <= STREAM_WARN_BYTES:
                        logger.warning("chimera: one sentence longer than 170 s; Hermes cuts streams at 16 MiB")
                    yield chunk
            except Exception as e:
                if _vocal.is_error(e, "TextError"):
                    raise RuntimeError(str(e) or NOTHING_SPEAKABLE) from e
                if isinstance(e, RuntimeError):
                    raise
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


def _target(out: str) -> tuple[str, bool]:
    """(path to write, reserved by us). ``.ogg``/``.wav`` paths are Hermes' own choice and used as given. Otherwise
    ``<stem>.wav`` - reserved exclusively so an existing file of the user is never overwritten; if it exists,
    ``<stem>.<8 hex>.wav``."""
    low = out.lower()
    if low.endswith(".ogg") or low.endswith(".wav"):
        return out, False
    stem = os.path.splitext(out)[0]
    candidates = [stem + ".wav"] + [f"{stem}.{secrets.token_hex(4)}.wav" for _ in range(20)]
    for cand in candidates:
        try:
            fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return cand, True
    raise RuntimeError(f"chimera synthesis failed: no free file name next to {out}")


def _remove_empty(path: str) -> None:
    try:
        if os.path.getsize(path) == 0:
            os.remove(path)
    except OSError:
        pass


def _tmp_dir() -> str | None:
    """Hermes' cache folder for chimera's temporary files (cleaned of leftovers after a day); None if unknown."""
    try:
        from . import paths

        return str(paths.tmp_dir(create=True))
    except Exception:  # noqa: BLE001 - the engine then uses the output folder / system temp
        return None


_CLASSES: dict = {}


def build_provider(ctx: Any) -> "ChimeraTTSProvider":
    from agent.tts_provider import TTSProvider

    cls = _CLASSES.get(TTSProvider)
    if cls is None:
        cls = type("ChimeraTTSProvider", (ChimeraTTSProvider, TTSProvider), {"__module__": __name__})
        _CLASSES[TTSProvider] = cls
    return cls(ctx)
