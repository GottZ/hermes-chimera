"""Articulator dynamics (control/articulators.rs): each articulator is a critically damped second-order system
tracking its target, stepped once per control block with semi-implicit Euler (articulators.rs:141-146).
Also the neutral posture and the resting tongue tip (anatomy/human.rs:421-451).

Arrays are float64[17] in ``core.keys.ARTIC_KEYS`` order.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np

from .score import Blend


class ArticSpec(NamedTuple):
    min: float
    max: float
    hz: float      # natural frequency of the tracking dynamics (omega = 2*pi*hz*speed)


# articulators.rs:24-46, ARTIC_KEYS order
ARTIC_SPECS: tuple[ArticSpec, ...] = (
    ArticSpec(-0.1, 2.6, 10.0),   # jaw
    ArticSpec(1.0, 4.8, 12.0),    # tbX
    ArticSpec(4.8, 9.2, 12.0),    # tbY
    ArticSpec(3.5, 8.0, 13.0),    # ttX
    ArticSpec(5.0, 10.8, 13.0),   # ttY
    ArticSpec(0.0, 1.0, 9.0),     # ttCurl
    ArticSpec(-1.0, 1.0, 9.0),    # root
    ArticSpec(-0.4, 2.4, 11.0),   # lipAp
    ArticSpec(-0.2, 1.5, 11.0),   # lipProt
    ArticSpec(0.0, 1.0, 11.0),    # lipWidth
    ArticSpec(0.0, 1.0, 10.0),    # labiodental
    ArticSpec(-0.25, 1.4, 12.0),  # velum
    ArticSpec(-1.6, 1.6, 8.0),    # larynx
    ArticSpec(0.0, 1.0, 6.0),     # epilarynx
    ArticSpec(0.0, 1.0, 6.0),     # ventricular
    ArticSpec(0.0, 1.0, 10.0),    # lateral
    ArticSpec(0.0, 1.0, 9.0),     # dorsum
)

# human.rs:421-439 (template coordinates), ARTIC_KEYS order
NEUTRAL: tuple[float, ...] = (0.6, 2.9, 6.85, 6.45, 7.3, 0.0, 0.0, 0.7, 0.2, 0.5, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0,
                              0.0)


def rest_tip(jaw: float, tb_x: float, tb_y: float) -> tuple[float, float]:
    """Resting tongue-tip position behind the lower incisors (human.rs:443-451), template coordinates."""
    raise NotImplementedError


def artic_array(patch: Mapping[str, float] | None = None, base: "np.ndarray | tuple[float, ...]" = NEUTRAL
                ) -> np.ndarray:
    """``{...base, ...patch}`` as float64[17] (ArticPatch::apply_to, human.rs:203-209); keys in ARTIC_KEYS."""
    raise NotImplementedError


class ArticulatorBank:
    """articulators.rs:58-148. Public state: ``x``, ``v``, ``target``, ``speed``, ``effective`` (float64[17]),
    ``tip_free`` (bool). A new bank rests at NEUTRAL with the tip at rest_tip(NEUTRAL) (articulators.rs:79-87)."""

    x: np.ndarray
    v: np.ndarray
    target: np.ndarray
    speed: np.ndarray
    effective: np.ndarray
    tip_free: bool

    def __init__(self) -> None:
        raise NotImplementedError

    def set_target(self, patch: Mapping[str, float]) -> None:
        """Clamp each finite value to its ARTIC_SPECS range and set it as target (articulators.rs:89-97)."""
        raise NotImplementedError

    def snap(self, patch: Mapping[str, float]) -> None:
        """set_target, then jump: x = target, v = 0 (articulators.rs:100-104)."""
        raise NotImplementedError

    def step(self, dt: float, blend: Blend | None, base_weight: float) -> None:
        """One control step (articulators.rs:109-148): tip follows rest_tip when tip_free; per key the target is
        blended with the gestures (velum: bounded blend; others: weighted mean with ``base_weight``), speed
        multiplied, clamped to the range, then semi-implicit Euler with omega = 2*pi*hz*speed."""
        raise NotImplementedError
