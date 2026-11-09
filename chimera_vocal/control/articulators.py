"""Articulator dynamics (control/articulators.rs): each articulator is a critically damped second-order system
tracking its target, stepped once per control block with semi-implicit Euler (articulators.rs:141-146).
Also the neutral posture and the resting tongue tip (anatomy/human.rs:421-451).

Arrays are float64[17] in ``core.keys.ARTIC_KEYS`` order.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import numpy as np

from ..core.keys import ARTIC_INDEX, ARTIC_KEYS, N_ARTIC
from ..core.nbmath import js_max, js_min
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

_JAW, _TBX, _TBY, _TTX, _TTY = (ARTIC_INDEX[k] for k in ("jaw", "tbX", "tbY", "ttX", "ttY"))
_VELUM = ARTIC_INDEX["velum"]
_OMEGA = tuple(2.0 * math.pi * s.hz for s in ARTIC_SPECS)


def rest_tip(jaw: float, tb_x: float, tb_y: float) -> tuple[float, float]:
    """Resting tongue-tip position behind the lower incisors (human.rs:443-451), template coordinates."""
    cx, cy = -2.5, 10.5
    px, py = 6.45, 7.35
    ang = -jaw / 9.84
    dx, dy = px - cx, py - cy
    x = cx + dx * math.cos(ang) - dy * math.sin(ang)
    y = cy + dx * math.sin(ang) + dy * math.cos(ang)
    return (x + 0.3 * (tb_x - NEUTRAL[_TBX]), y + 0.35 * (tb_y - NEUTRAL[_TBY]))


def artic_array(patch: Mapping[str, float] | None = None, base: "np.ndarray | tuple[float, ...]" = NEUTRAL
                ) -> np.ndarray:
    """``{...base, ...patch}`` as float64[17] (ArticPatch::apply_to, human.rs:203-209); keys in ARTIC_KEYS."""
    out = np.array(base, dtype=np.float64)
    for k, v in (patch or {}).items():
        out[ARTIC_INDEX[k]] = float(v)
    return out


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
        self.x = np.array(NEUTRAL, dtype=np.float64)
        self.v = np.zeros(N_ARTIC)
        self.target = np.array(NEUTRAL, dtype=np.float64)
        self.tip_free = True
        self.speed = np.ones(N_ARTIC)
        self.effective = np.array(NEUTRAL, dtype=np.float64)
        tx, ty = rest_tip(NEUTRAL[_JAW], NEUTRAL[_TBX], NEUTRAL[_TBY])
        self.x[_TTX] = tx
        self.target[_TTX] = tx
        self.x[_TTY] = ty
        self.target[_TTY] = ty

    def set_target(self, patch: Mapping[str, float]) -> None:
        """Clamp each finite value to its ARTIC_SPECS range and set it as target (articulators.rs:89-97)."""
        for i in sorted(ARTIC_INDEX[k] for k in patch):
            val = float(patch[ARTIC_KEYS[i]])
            if not math.isfinite(val):
                continue
            s = ARTIC_SPECS[i]
            self.target[i] = js_min(s.max, js_max(s.min, val))

    def snap(self, patch: Mapping[str, float]) -> None:
        """set_target, then jump: x = target, v = 0 (articulators.rs:100-104)."""
        self.set_target(patch)
        self.x = self.target.copy()
        self.v = np.zeros(N_ARTIC)

    def step(self, dt: float, blend: Blend | None, base_weight: float) -> None:
        """One control step (articulators.rs:109-148)."""
        x, v, target, speed = self.x, self.v, self.target, self.speed
        if self.tip_free:
            tx, ty = rest_tip(float(x[_JAW]), float(x[_TBX]), float(x[_TBY]))
            target[_TTX] = tx
            target[_TTY] = ty
        if blend is not None:
            bval, bw, bsp = blend.value, blend.weight, blend.speed
        for i in range(N_ARTIC):
            tgt = float(target[i])
            sp = float(speed[i])
            if blend is not None:
                w = float(bw[i])
                if w > 0.0:
                    if i == _VELUM:
                        tgt += js_min(1.0, w / 2.0) * (float(bval[i]) - tgt)
                    else:
                        tgt = (tgt * base_weight + float(bval[i]) * w) / (base_weight + w)
                    sp *= float(bsp[i])
            s = ARTIC_SPECS[i]
            if tgt < s.min:
                tgt = s.min
            elif tgt > s.max:
                tgt = s.max
            self.effective[i] = tgt
            om = _OMEGA[i] * sp
            xi = float(x[i])
            acc = om * om * (tgt - xi) - 2.0 * om * float(v[i])
            v[i] = float(v[i]) + acc * dt
            x[i] = xi + float(v[i]) * dt

