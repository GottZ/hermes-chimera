"""Segment sequence -> gestural plan (port of phonetics/planner.rs).

S1 stub (INTERFACES §14): signatures only; the implementation follows in S2.
"""

from __future__ import annotations

from typing import Sequence

from ..core.body import BodyMix
from ..core.types import ArticPatch, Plan, PhraseInput, PlanOptions, Segment

EXHALE_LABEL = "exhale"   # planner.rs:19


def expand(ipa: str) -> tuple[str, ...] | None:
    """Diphthongs and r-coloured vowels -> vowel targets (planner.rs:47-58)."""
    raise NotImplementedError


def diphthong_parts(ipa: str) -> tuple[str, str] | None:
    """planner.rs:60-65."""
    raise NotImplementedError


def is_vowel(ipa: str, body: BodyMix) -> bool:
    """planner.rs:67-70."""
    raise NotImplementedError


def vowel_artic(ipa: str, body: BodyMix) -> ArticPatch | None:
    """Calibrated vowel articulation; blended between presets (planner.rs:74-108)."""
    raise NotImplementedError


def plan_utterance(segs: Sequence[Segment], opts: PlanOptions) -> Plan:
    """planner.rs:181-399."""
    raise NotImplementedError


def plan_phrases(phrases: Sequence[PhraseInput], opts: PlanOptions) -> Plan:
    """planner.rs:519-530."""
    raise NotImplementedError
