"""Fake ``agent.tts_provider`` with the semantics of hermes-agent 0f0e96fc8e (agent/tts_provider.py:26-106,
agent/provider_base.py:16-66): abstract ``name`` and ``synthesize``; ``streams_pcm`` False and
``stream_sample_rate`` None as class attributes; ``stream`` raises NotImplementedError; ``voice_compatible`` False."""

from __future__ import annotations

import abc
from typing import Any, Iterator, Optional

DEFAULT_OUTPUT_FORMAT = "mp3"


class TTSProvider(abc.ABC):
    streams_pcm: bool = False
    stream_sample_rate: Optional[int] = None

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def display_name(self) -> str:
        return self.name.title()

    def get_setup_schema(self) -> dict:
        return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list:
        return []

    def default_model(self) -> Optional[str]:
        return None

    def list_voices(self) -> list:
        return []

    def default_voice(self) -> Optional[str]:
        voices = self.list_voices()
        return voices[0].get("id") if voices else None

    @abc.abstractmethod
    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None, model: Optional[str] = None,
                   speed: Optional[float] = None, format: str = DEFAULT_OUTPUT_FORMAT, **extra: Any) -> str: ...

    def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None, format: str = "opus",
               **extra: Any) -> Iterator[bytes]:
        raise NotImplementedError(f"TTS provider {self.name!r} does not implement streaming synthesis.")

    def warm(self) -> None:
        pass

    def release(self) -> None:
        pass

    @property
    def voice_compatible(self) -> bool:
        return False
