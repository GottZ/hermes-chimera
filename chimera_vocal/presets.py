"""Voice presets and the voice-spec parser. No numpy import.

Presets are body + speaking style + source deviations; ids are stable forever (configuration values depend on them).
``male``/``female``/``child`` reproduce ``chimera render text … <body>`` of the template (f0 112/205/250 Hz,
``render.rs:135-141``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping

from .core.body import BodyMix, default_f0
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
    seed: int | None = None          # spec option "seed=": None/0 -> template seeds, n -> Seeds.for_voice(0, n)


_D = SourceParams()
_MALE, _FEMALE, _CHILD = BodyMix(0.0, 0.0), BodyMix(1.0, 0.0), BodyMix(0.0, 1.0)


def _p(pid: str, display: str, body: BodyMix, f0: float | None, rate: float, accent: float | None,
       gender: str, description: str, **source: float | bool) -> VoicePreset:
    return VoicePreset(pid, display, body, _D.patched(**source) if source else _D, SpeechStyle(rate, f0, accent),
                       gender, description)  # type: ignore[arg-type]


PRESETS: Mapping[str, VoicePreset] = {p.id: p for p in (
    _p("male", "Male", _MALE, 112.0, 1.0, None, "male", "Adult male, the template's default voice"),
    _p("female", "Female", _FEMALE, 205.0, 1.0, None, "female", "Adult female"),
    _p("child", "Child", _CHILD, 250.0, 1.0, None, "neutral", "Child"),
    _p("androgynous", "Androgynous", BodyMix(0.5, 0.0), None, 1.0, None, "neutral",
       "Between the adult male and female bodies"),
    _p("teen", "Teen", BodyMix(0.5, 0.5), None, 1.05, None, "neutral", "Between adult and child, a little faster"),
    _p("male-deep", "Male (deep)", _MALE, 92.0, 0.95, None, "male", "Low, darker male speaker",
       tension=0.4, brightness=0.15),
    _p("female-bright", "Female (bright)", _FEMALE, 225.0, 1.0, None, "female", "Higher, brighter female speaker",
       tension=0.55, brightness=0.45),
    _p("narrator", "Narrator", _MALE, 104.0, 0.92, 0.12, "male", "Calm, slower male narrator"),
    _p("whisper", "Whisper", _FEMALE, 205.0, 0.95, None, "female", "Whispered female voice", voiced=False),
    _p("male-whisper", "Whisper (male)", _MALE, 112.0, 0.95, None, "male", "Whispered male voice", voiced=False),
    _p("robot", "Robot", _MALE, 110.0, 1.0, 0.0, "male", "Monotone: no accents, no drift", drift=0.0, roughness=0.0),
    _p("elder", "Elder", _MALE, 118.0, 0.9, None, "male", "Rough, breathy older male",
       roughness=0.35, breathiness=0.3, tension=0.35),
)}
DEFAULT_VOICE: str = "male"

_UNIT_SOURCE = ("breathiness", "tension", "brightness", "roughness")


def _num(v: str) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(v)
    return x


def _in(x: float, lo: float, hi: float) -> float:
    if not lo <= x <= hi:
        raise ValueError(x)
    return x


def parse_voice_spec(spec: str | None) -> VoicePreset | None:
    """``preset(;option)*`` -> preset with options applied; None/"" -> None; invalid -> None (never raises)."""
    if spec is None or not isinstance(spec, str) or not spec.strip():
        return None
    parts = [p.strip() for p in spec.split(";")]
    base = PRESETS.get(parts[0].lower())
    if base is None:
        return None
    try:
        opts: dict[str, str] = {}
        for part in parts[1:]:
            if not part:
                continue
            k, sep, v = part.partition("=")
            k, v = k.strip().lower(), v.strip()
            if not sep or not v:
                return None
            opts[k] = v
        body = base.body
        source = base.source
        speech = base.speech
        seed = base.seed
        pitch: float | None = None
        if "body" in opts:
            fc = opts.pop("body").split(",")
            if len(fc) != 2:
                return None
            body = BodyMix(_in(_num(fc[0]), 0.0, 1.0), _in(_num(fc[1]), 0.0, 1.0))
            speech = replace(speech, f0=None)
        if "pitch" in opts:
            pitch = _in(_num(opts.pop("pitch")), 50.0, 600.0)
            speech = replace(speech, f0=pitch)
        if "semitones" in opts:
            st = _in(_num(opts.pop("semitones")), -24.0, 24.0)
            f0 = speech.f0 if speech.f0 is not None else default_f0(body)
            speech = replace(speech, f0=_in(f0 * 2.0 ** (st / 12.0), 50.0, 600.0))
        if "rate" in opts:
            speech = replace(speech, rate=_in(_num(opts.pop("rate")), 0.5, 1.8))
        if "accent" in opts:
            speech = replace(speech, accent=_in(_num(opts.pop("accent")), 0.0, 0.4))
        changes: dict[str, float | bool] = {}
        for k in _UNIT_SOURCE:
            if k in opts:
                changes[k] = _in(_num(opts.pop(k)), 0.0, 1.0)
        if "whisper" in opts:
            w = opts.pop("whisper").lower()
            if w not in ("true", "false"):
                return None
            changes["voiced"] = w == "false"
        if "seed" in opts:
            s = opts.pop("seed")
            if not s.isdigit():
                return None
            seed = int(s)
            if seed > 0xFFFFFFFF:
                return None
        if opts:
            return None
        if changes:
            source = source.patched(**changes)
        return VoicePreset(base.id, base.display, body, source, speech, base.gender, base.description, seed)
    except (ValueError, TypeError):
        return None


def list_voices() -> list[dict]:
    """``[{"id", "display", "language": "en", "gender"}]`` in PRESETS order (TTSProvider.list_voices)."""
    return [{"id": p.id, "display": p.display, "language": "en", "gender": p.gender} for p in PRESETS.values()]


def speaking_f0(preset: VoicePreset) -> float:
    """The preset's speaking f0: explicit, else default_f0 of its body."""
    return preset.speech.f0 if preset.speech.f0 is not None else default_f0(preset.body)
