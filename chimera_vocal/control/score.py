"""Gestural score (control/score.rs): raised-cosine activations, strength-weighted blending of the active
gestures per control key in insertion order, release of tagged (sung) gestures, pruning.

Gestures are ``core.types.Gesture``; the score stores them internally as a struct of arrays ("GestureArray",
INTERFACES §1) so blends can be vectorised; ``gestures()`` gives them back as Gesture objects in score order.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from ..core.types import Gesture


class Blend(NamedTuple):
    """Blended control keys at one time (score.rs:257-269), each float64[N_CONTROL] in CONTROL_KEYS order."""
    value: np.ndarray    # strength-weighted mean target (0 where no gesture is active)
    weight: np.ndarray   # total active weight (0 = no gesture controls the key)
    speed: np.ndarray    # articulator speed multiplier (1 = none)


CAPACITY = 16384         # score.rs:305 GestureScore::CAPACITY


def activation(g: Gesture, t: float) -> float:
    """Smooth activation 0..1..0 with cosine ramps (score.rs:271-286)."""
    raise NotImplementedError


class GestureScore:
    """score.rs:288-437. ``end``: end of the latest gesture (s); ``dropped``: gestures refused at CAPACITY."""

    end: float
    dropped: int

    def __init__(self) -> None:
        raise NotImplementedError

    def add(self, g: Gesture) -> None:
        raise NotImplementedError

    def add_all(self, gs: Sequence[Gesture]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def release(self, tag: float, t: float) -> None:
        """End every gesture with this tag at time t (score.rs:331-355): not yet begun -> dropped; ramping in ->
        turns round and ramps out from its present activation; at full strength -> ramp-out starts now; already
        ending -> unchanged. Recomputes ``end``."""
        raise NotImplementedError

    def prune(self, t: float) -> None:
        """Drop gestures whose end() <= t (score.rs:357-363)."""
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def gestures(self) -> list[Gesture]:
        """The gestures in score order (copies)."""
        raise NotImplementedError

    def blend(self, key: int, t: float) -> tuple[float, float, float]:
        """(value, weight, speed) of control key index ``key`` at time t (score.rs:378-401)."""
        raise NotImplementedError

    def blend_all(self, t: float) -> Blend:
        """All keys in one pass, same sums in the same order as ``blend`` per key (score.rs:403-436)."""
        raise NotImplementedError
