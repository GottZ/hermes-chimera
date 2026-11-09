"""Voice presets and the voice-spec parser (C4, INTERFACES §7). No numpy import.

Presets are body + speaking style + source deviations; ids are stable forever (configuration values depend on them).
``male``/``female``/``child`` reproduce ``chimera render text … <body>`` of the template (f0 112/205/250 Hz,
``render.rs:135-141``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .core.body import BodyMix
from .core.types import SourceParams


@dataclass(frozen=True)
class SpeechStyle:
    rate: float = 1.0
    f0: float | None = None          # speaking f0 (Hz); None -> default_f0(body)
    accent: float | None = None      # PlanOptions.accent


@dataclass(frozen=True)
class VoicePreset:
    id: str
    display: str
    body: BodyMix
    source: SourceParams
    speech: SpeechStyle
    gender: Literal["male", "female", "neutral"]
    description: str
    seed: int | None = None          # request C4-1: spec option "seed=" (None -> template seeds, Seeds.for_voice(0, seed))


class _Pending(Mapping):
    """Placeholder mapping of S1: every access raises NotImplementedError."""

    def __getitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


PRESETS: Mapping[str, VoicePreset] = _Pending()   # ordered; table INTERFACES §7
DEFAULT_VOICE: str = "male"


def parse_voice_spec(spec: str | None) -> VoicePreset | None:
    """``preset(;option)*`` -> preset with options applied; None/"" -> None; invalid -> None (never raises)."""
    raise NotImplementedError


def list_voices() -> list[dict]:
    """``[{"id", "display", "language": "en", "gender"}]`` in PRESETS order (TTSProvider.list_voices)."""
    raise NotImplementedError
