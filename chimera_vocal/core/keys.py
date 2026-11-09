"""Articulator, source and control keys in the template's index order.

ArticKey: anatomy/human.rs:83-105 (NAMES); SourceKey: control/score.rs (SOURCE_NAMES);
ControlKey = articulators then source channels, index = ControlKey::index() (score.rs:31-49).
"""

from __future__ import annotations

ARTIC_KEYS: tuple[str, ...] = (
    "jaw", "tbX", "tbY", "ttX", "ttY", "ttCurl", "root", "lipAp", "lipProt", "lipWidth",
    "labiodental", "velum", "larynx", "epilarynx", "ventricular", "lateral", "dorsum",
)
SOURCE_KEYS: tuple[str, ...] = ("abduction", "closure", "breath", "f0", "vibrato")
CONTROL_KEYS: tuple[str, ...] = ARTIC_KEYS + SOURCE_KEYS

N_ARTIC: int = 17
N_SOURCE: int = 5
N_CONTROL: int = 22

ARTIC_INDEX: dict[str, int] = {k: i for i, k in enumerate(ARTIC_KEYS)}
CONTROL_INDEX: dict[str, int] = {k: i for i, k in enumerate(CONTROL_KEYS)}

assert len(ARTIC_KEYS) == N_ARTIC and len(SOURCE_KEYS) == N_SOURCE and len(CONTROL_KEYS) == N_CONTROL


def control_index(name: str) -> int | None:
    """ControlKey::from_name(name).index(), or None for an unknown name (score.rs:71-76)."""
    return CONTROL_INDEX.get(name)
