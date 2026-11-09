"""Plain data types shared by text, planning, control and the engine (INTERFACES §4).

Field names are snake_case; ``to_json()``/``from_json()`` use the template's serde form (camelCase,
optional fields omitted when None), so goldens and exported vectors load without translation tables.
No numpy import.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any, Mapping

from .body import BodyMix
from .keys import CONTROL_INDEX, CONTROL_KEYS
from .nbmath import rng_next

LABEL_CAP = 24  # control/score.rs Label::CAP (bytes, cut at a char boundary)


def _opt(d: dict, key: str, value: Any) -> None:
    if value is not None:
        d[key] = value


def _label(s: str | None) -> str | None:
    """Label::new: at most 24 UTF-8 bytes, cut back to a char boundary (score.rs:157-170)."""
    if s is None:
        return None
    b = s.encode("utf-8")
    if len(b) <= LABEL_CAP:
        return s
    return b[:LABEL_CAP].decode("utf-8", errors="ignore")


@dataclass
class Segment:
    ipa: str
    dur: float | None = None
    f0: float | None = None
    stress: int | None = None
    word_end: bool | None = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"ipa": self.ipa}
        _opt(d, "dur", self.dur)
        _opt(d, "stress", self.stress)
        _opt(d, "f0", self.f0)
        _opt(d, "wordEnd", self.word_end)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Segment":
        return cls(ipa=d["ipa"], dur=d.get("dur"), f0=d.get("f0"), stress=d.get("stress"), word_end=d.get("wordEnd"))


class Intonation(enum.Enum):
    STATEMENT = "statement"   # final fall (default)
    QUESTION = "question"     # final rise
    FLAT = "flat"


@dataclass
class Phrase:
    segments: list[Segment]
    intonation: Intonation
    pause: float
    words: list[str]

    def to_json(self) -> dict:
        return {"segments": [s.to_json() for s in self.segments], "intonation": self.intonation.value,
                "pause": self.pause, "words": list(self.words)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Phrase":
        return cls([Segment.from_json(s) for s in d["segments"]], Intonation(d["intonation"]), d["pause"],
                   list(d["words"]))


@dataclass
class PhraseInput:
    segments: list[Segment]
    intonation: Intonation | None = None
    pause: float | None = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"segments": [s.to_json() for s in self.segments]}
        _opt(d, "intonation", self.intonation.value if self.intonation is not None else None)
        _opt(d, "pause", self.pause)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PhraseInput":
        i = d.get("intonation")
        return cls([Segment.from_json(s) for s in d["segments"]], Intonation(i) if i is not None else None,
                   d.get("pause"))


@dataclass
class Syllable:
    text: str
    onset: list[Segment]
    nucleus: Segment
    coda: list[Segment]
    melisma: bool | None
    word_end: bool
    phrase_end: bool

    def to_json(self) -> dict:
        d: dict[str, Any] = {"text": self.text, "onset": [s.to_json() for s in self.onset],
                             "nucleus": self.nucleus.to_json(), "coda": [s.to_json() for s in self.coda]}
        _opt(d, "melisma", self.melisma)
        d["wordEnd"] = self.word_end
        d["phraseEnd"] = self.phrase_end
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Syllable":
        return cls(d["text"], [Segment.from_json(s) for s in d["onset"]], Segment.from_json(d["nucleus"]),
                   [Segment.from_json(s) for s in d["coda"]], d.get("melisma"), d["wordEnd"], d["phraseEnd"])


@dataclass
class SungNote:
    midi: float
    beats: float
    syl: Syllable

    def to_json(self) -> dict:
        return {"midi": self.midi, "beats": self.beats, "syl": self.syl.to_json()}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SungNote":
        return cls(d["midi"], d["beats"], Syllable.from_json(d["syl"]))


@dataclass
class PlanOptions:
    body: BodyMix
    t0: float
    rate: float | None = None
    f0: float | None = None
    intonation: Intonation | None = None
    accent: float | None = None
    open_end: bool | None = None
    sung: bool | None = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"body": self.body.to_json(), "t0": self.t0}
        _opt(d, "rate", self.rate)
        _opt(d, "f0", self.f0)
        _opt(d, "intonation", self.intonation.value if self.intonation is not None else None)
        _opt(d, "accent", self.accent)
        _opt(d, "openEnd", self.open_end)
        _opt(d, "sung", self.sung)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PlanOptions":
        i = d.get("intonation")
        return cls(BodyMix.parse(d.get("body", "male")), d.get("t0", 0.0), d.get("rate"), d.get("f0"),
                   Intonation(i) if i is not None else None, d.get("accent"), d.get("openEnd"), d.get("sung"))


@dataclass
class SongOptions:
    body: BodyMix
    bpm: float
    t0: float
    vibrato: float

    def to_json(self) -> dict:
        return {"body": self.body.to_json(), "bpm": self.bpm, "t0": self.t0, "vibrato": self.vibrato}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SongOptions":
        return cls(BodyMix.parse(d.get("body", "male")), d.get("bpm", 0.0), d.get("t0", 0.0), d.get("vibrato", 0.0))


@dataclass
class Gesture:
    t0: float
    t1: float
    ramp_in: float
    ramp_out: float
    strength: float
    targets: dict[str, float]            # only present keys; key in CONTROL_KEYS
    speed: float | None = None
    release_hold: float | None = None
    label: str | None = None
    tag: float | None = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"t0": self.t0, "t1": self.t1, "rampIn": self.ramp_in, "rampOut": self.ramp_out,
                             "strength": self.strength,
                             "targets": {k: self.targets[k] for k in CONTROL_KEYS if k in self.targets}}
        _opt(d, "speed", self.speed)
        _opt(d, "releaseHold", self.release_hold)
        _opt(d, "label", self.label)
        _opt(d, "tag", self.tag)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Gesture":
        # Targets deserialize: unknown keys dropped, order = ControlKey index (score.rs:135-146)
        raw = d["targets"]
        targets = {k: raw[k] for k in CONTROL_KEYS if k in raw}
        return cls(d["t0"], d["t1"], d["rampIn"], d["rampOut"], d["strength"], targets, d.get("speed"),
                   d.get("releaseHold"), _label(d.get("label")), d.get("tag"))

    def target_index(self) -> list[tuple[int, float]]:
        """(ControlKey index, value) pairs in index order."""
        return [(CONTROL_INDEX[k], v) for k, v in self.targets.items()]


@dataclass
class PlannedSegment:
    ipa: str
    kind: str
    t0: float
    t1: float
    stress: int | None = None   # serde Option<u8> (phonetics/mod.rs:58-66); added to INTERFACES §4 in S0

    def to_json(self) -> dict:
        d: dict[str, Any] = {"ipa": self.ipa, "t0": self.t0, "t1": self.t1}
        _opt(d, "stress", self.stress)
        d["kind"] = self.kind
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PlannedSegment":
        return cls(d["ipa"], d["kind"], d["t0"], d["t1"], d.get("stress"))


@dataclass
class Plan:
    gestures: list[Gesture]
    segments: list[PlannedSegment]
    end: float

    def to_json(self) -> dict:
        return {"gestures": [g.to_json() for g in self.gestures], "segments": [s.to_json() for s in self.segments],
                "end": self.end}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Plan":
        return cls([Gesture.from_json(g) for g in d["gestures"]],
                   [PlannedSegment.from_json(s) for s in d["segments"]], d["end"])


ArticPatch = dict  # dict[str, float], keys subset of ARTIC_KEYS

_SOURCE_JSON = (("f0", "f0"), ("glide", "glide"), ("pressure", "pressure"), ("breathiness", "breathiness"),
                ("tension", "tension"), ("fry", "fry"), ("roughness", "roughness"),
                ("vibrato_depth", "vibratoDepth"), ("vibrato_rate", "vibratoRate"), ("drift", "drift"),
                ("voiced", "voiced"), ("brightness", "brightness"))


@dataclass(frozen=True)
class SourceParams:
    """VoiceSource / DEFAULT_SOURCE (voice/human_voice.rs:15-49)."""

    f0: float = 110.0
    glide: float = 0.06
    pressure: float = 8.0
    breathiness: float = 0.15
    tension: float = 0.5
    fry: float = 0.0
    roughness: float = 0.1
    vibrato_depth: float = 0.0
    vibrato_rate: float = 6.0
    drift: float = 5.0
    voiced: bool = True
    brightness: float = 0.25

    def patched(self, **changes: float | bool) -> "SourceParams":
        """SourcePatch::apply: replace the given fields (unknown names raise TypeError)."""
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict:
        return {j: getattr(self, p) for p, j in _SOURCE_JSON}

    @classmethod
    def from_json(cls, d: Mapping[str, Any], base: "SourceParams | None" = None) -> "SourceParams":
        """Serde camelCase; a partial mapping patches ``base`` (default DEFAULT_SOURCE), like SourcePatch."""
        changes = {p: d[j] for p, j in _SOURCE_JSON if j in d}
        return (base or cls()).patched(**changes)


DEFAULT_SOURCE = SourceParams()

_GOLDEN = 0x9E3779B9


def _derive(base: int, n: int) -> int:
    """mulberry32 output of state (base ^ (0x9E3779B9 * n mod 2^32)); never 0 (INTERFACES §4)."""
    x = (base ^ ((_GOLDEN * n) & 0xFFFFFFFF)) & 0xFFFFFFFF
    _, v = rng_next(x)
    s = int(v * 4294967296.0)
    return s if s != 0 else 1


@dataclass(frozen=True)
class Seeds:
    """Per-instance seeds of the template: net 12345 (network.rs:415), glottis 777, drift 4242 (glottis.rs:118-119)."""

    net: int = 12345
    glottis: int = 777
    drift: int = 4242

    @classmethod
    def for_voice(cls, index: int, seed: int | None = None) -> "Seeds":
        """Voice ``index`` of a score. ``seed`` overrides: 0 = template seeds (reproduction); None -> index.

        n = 0 gives the template seeds; n >= 1 derives all three (D§2.8, D§8.3).
        """
        n = index if seed is None else (int(seed) & 0xFFFFFFFF)
        if n < 0:
            raise ValueError("voice index must be >= 0")
        base = cls()
        if n == 0:
            return base
        return cls(_derive(base.net, n), _derive(base.glottis, n), _derive(base.drift, n))

    @classmethod
    def for_window(cls, base: "Seeds", k: int) -> "Seeds":
        """Phrase window ``k`` of one utterance: k = 0 -> ``base``; k >= 1 derived (D§3.3, P1-13)."""
        if k < 0:
            raise ValueError("window index must be >= 0")
        if k == 0:
            return base
        return cls(_derive(base.net, k), _derive(base.glottis, k), _derive(base.drift, k))
