"""Phoneme inventory: articulatory definitions per IPA symbol, calibrated consonant targets per body
(port of phonetics/inventory.rs).

S1 stub (INTERFACES §14): types and signatures only; the implementation follows in S2.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ..core.body import BodyMix
from ..core.types import ArticPatch


class Manner(enum.Enum):
    """inventory.rs:16-44 (serde lowercase)."""

    VOWEL = "vowel"
    STOP = "stop"
    NASAL = "nasal"
    FRICATIVE = "fricative"
    AFFRICATE = "affricate"
    APPROXIMANT = "approximant"
    LATERAL = "lateral"
    TAP = "tap"
    TRILL = "trill"
    GLOTTAL = "glottal"


class Voicing(enum.Enum):
    VOICED = "voiced"
    VOICELESS = "voiceless"


class Glottal(enum.Enum):
    OPEN = "open"
    STOP = "stop"
    BREATHY = "breathy"


@dataclass(frozen=True)
class PhonemeDef:
    """inventory.rs:62-89."""

    ipa: str
    name: str
    manner: Manner
    voicing: Voicing
    place: str
    targets: ArticPatch                 # key order = ARTIC_KEYS
    dominant: tuple[str, ...]           # articulators this segment controls strongly
    nasal: bool | None
    dur: float                          # default duration (s)
    area: float | None = None
    cal_key: str | None = None
    release: str | None = None          # affricate: the fricative release
    glottal: Glottal | None = None


def phoneme(ipa: str, body: BodyMix) -> PhonemeDef | None:
    """Definition with calibrated targets for ``body``; blended between presets (inventory.rs:247-263)."""
    raise NotImplementedError


def consonants() -> tuple[PhonemeDef, ...]:
    """All 64 definitions in table order (inventory.rs:265-267)."""
    raise NotImplementedError
