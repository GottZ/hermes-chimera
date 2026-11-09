"""Singing: syllables on notes through the speech planner; live notes (port of phonetics/sing.rs).

The option types below mirror the template's serde structs (sing.rs:194-236); INTERFACES §14 says
"Signaturen = Rust" for the live-note planners.

S1 stub (INTERFACES §14): types and signatures only; the planners follow in S2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.body import BodyMix
from ..core.types import Gesture, Plan, SongOptions, SungNote, Syllable


@dataclass
class LegatoFrom:
    """The previous note of a legato pair (sing.rs:194-200)."""

    syl: Syllable
    midi: float

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "LegatoFrom":
        return cls(Syllable.from_json(d["syl"]), d["midi"])

    def to_json(self) -> dict:
        return {"syl": self.syl.to_json(), "midi": self.midi}


@dataclass
class NoteOnOptions:
    """sing.rs:201-210."""

    body: BodyMix
    vibrato: float
    tag: float
    legato_from: LegatoFrom | None = None

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "NoteOnOptions":
        lf = d.get("legatoFrom")
        return cls(BodyMix.parse(d.get("body", "male")), d["vibrato"], d["tag"],
                   LegatoFrom.from_json(lf) if lf is not None else None)

    def to_json(self) -> dict:
        out: dict[str, Any] = {"body": self.body.to_json(), "vibrato": self.vibrato, "tag": self.tag}
        if self.legato_from is not None:
            out["legatoFrom"] = self.legato_from.to_json()
        return out


@dataclass(frozen=True)
class MelismaOptions:
    """sing.rs:230-236."""

    vibrato: float
    tag: float

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "MelismaOptions":
        return cls(d["vibrato"], d["tag"])

    def to_json(self) -> dict:
        return {"vibrato": self.vibrato, "tag": self.tag}


def plan_song(notes: Sequence[SungNote], opts: SongOptions) -> Plan:
    """A whole melody, one utterance per phrase (sing.rs:101-177)."""
    raise NotImplementedError


def plan_note_on(syl: Syllable, midi: float, o: NoteOnOptions) -> list[Gesture]:
    """Key pressed (sing.rs:214-228)."""
    raise NotImplementedError


def plan_melisma(midi: float, o: MelismaOptions) -> list[Gesture]:
    """Held vowel to a new pitch (sing.rs:238-244)."""
    raise NotImplementedError


def plan_note_off(syl: Syllable, midi: float, body: BodyMix) -> list[Gesture]:
    """Key released: glide and coda (sing.rs:247-254)."""
    raise NotImplementedError
