"""Score format (INTERFACES §8.1) and validation (§8.2): precise issues with a JSON path and the fix. No numpy import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

SCORE_SCHEMA: dict = {}          # JSON schema of §8.1 (documentation/tooling; validate_score is the truth) - S2

NATURAL_RANGE_MIDI: Mapping[str, tuple[int, int]] = {"male": (38, 69), "female": (53, 84), "child": (57, 81)}
    # D2–A4, F3–C6, A3–A5: convention, warnings only; blends interpolate the bounds by BodyMix weights

MIDI_MIN, MIDI_MAX = 28, 100     # measured range of the engine (E1-work/pitch-range)
DYN_LEVELS: tuple[str, ...] = ("pp", "p", "mp", "mf", "f", "ff")


@dataclass(frozen=True)
class ScoreIssue:
    path: str                      # JSON path, e.g. "voices[1].notes[7].pitch"
    severity: Literal["error", "warning"]
    message: str                   # one sentence, says what is wrong AND what to write instead


@dataclass(frozen=True)
class VoiceSummary:
    name: str
    body: str
    items: int
    sounding: int
    spoken: int
    first_beat: float
    end_beat: float
    seconds: float
    lowest: str
    highest: str
    syllables: list[str]           # per sounding note as the engine will sing it ("~" = melisma)


@dataclass(frozen=True)
class ScoreReport:
    ok: bool
    issues: list[ScoreIssue]
    voices: list[VoiceSummary]
    seconds: float                 # mix length incl. lead_in and release tail
    voice_seconds: float           # Σ RENDERED voice lengths (start_block … end + 0.4 s)
    lead_in: float                 # effective (possibly raised) lead_in
    memory_mb: float
    disk_mb: float
    estimate_s: dict[str, float]
    normalized: dict               # score with all defaults filled, pitches as MIDI floats


def parse_pitch(value: Any) -> float:
    """Note name ``<letter>[#|b|##|bb]<octave>[±cents]`` (C4 = 60) or a MIDI number -> MIDI float; ValueError if invalid."""
    raise NotImplementedError


def midi_name(midi: float) -> str:
    """MIDI float -> note name with sharps and cents if fractional, e.g. 61 -> "C#4", 69.15 -> "A4+15"."""
    raise NotImplementedError


def parse_body(spec: Any) -> "Any":
    """Score body: "male"|"female"|"child"|{"female","child"}|"female=0.5,child=0.2" -> BodyMix; ValueError if invalid."""
    raise NotImplementedError


def validate_score(score: "dict | str", *, g2p: Any, max_voices: int = 8,
                   max_seconds: float = 300.0, max_voice_seconds: float = 1200.0) -> ScoreReport:
    """Check types, ranges, syllable counts, spoken slots and size; raise lead_in when an onset would start before 0.005 s."""
    raise NotImplementedError
