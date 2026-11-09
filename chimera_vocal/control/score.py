"""Gestural score (control/score.rs): raised-cosine activations, strength-weighted blending of the active
gestures per control key in insertion order, release of tagged (sung) gestures, pruning.

Gestures are ``core.types.Gesture``; the score keeps mutable copies (a release moves t0/t1) with their targets as
(key index, value) pairs in key order; ``gestures()`` gives them back as Gesture objects in score order.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

from ..core.keys import CONTROL_INDEX, N_CONTROL
from ..core.nbmath import js_max, js_min
from ..core.types import Gesture


class Blend(NamedTuple):
    """Blended control keys at one time (score.rs:257-269), each float64[N_CONTROL] in CONTROL_KEYS order."""
    value: np.ndarray    # strength-weighted mean target (0 where no gesture is active)
    weight: np.ndarray   # total active weight (0 = no gesture controls the key)
    speed: np.ndarray    # articulator speed multiplier (1 = none)


CAPACITY = 16384         # score.rs:305 GestureScore::CAPACITY
_PI = math.pi


def activation(g: Gesture, t: float) -> float:
    """Smooth activation 0..1..0 with cosine ramps (score.rs:271-286)."""
    if t <= g.t0 - g.ramp_in or t >= g.t1 + g.ramp_out:
        return 0.0
    if t < g.t0:
        u = (t - (g.t0 - g.ramp_in)) / js_max(1e-6, g.ramp_in)
        return 0.5 - 0.5 * math.cos(_PI * u)
    if t > g.t1:
        u = (t - g.t1) / js_max(1e-6, g.ramp_out)
        return 0.5 + 0.5 * math.cos(_PI * u)
    return 1.0


def _end(g: Gesture) -> float:
    return g.t1 + g.ramp_out + (g.release_hold if g.release_hold is not None else 0.0)


class _G:
    """Mutable copy of a Gesture with its targets as (index, value) pairs in key order."""
    __slots__ = ("t0", "t1", "ramp_in", "ramp_out", "strength", "targets", "speed", "release_hold", "label", "tag",
                 "pairs")

    def __init__(self, g: Gesture) -> None:
        self.t0, self.t1, self.ramp_in, self.ramp_out = g.t0, g.t1, g.ramp_in, g.ramp_out
        self.strength, self.targets, self.speed = g.strength, dict(g.targets), g.speed
        self.release_hold, self.label, self.tag = g.release_hold, g.label, g.tag
        self.pairs = sorted((CONTROL_INDEX[k], float(v)) for k, v in g.targets.items()
                            if k in CONTROL_INDEX and v == v)

    def gesture(self) -> Gesture:
        return Gesture(self.t0, self.t1, self.ramp_in, self.ramp_out, self.strength, dict(self.targets), self.speed,
                       self.release_hold, self.label, self.tag)


class GestureScore:
    """score.rs:288-437. ``end``: end of the latest gesture (s); ``dropped``: gestures refused at CAPACITY."""

    end: float
    dropped: int

    def __init__(self) -> None:
        self._g: list[_G] = []
        self.end = 0.0
        self.dropped = 0

    def add(self, g: Gesture) -> None:
        if len(self._g) >= CAPACITY:
            self.dropped += 1
            return
        self._g.append(_G(g))
        self.end = js_max(self.end, _end(g))

    def add_all(self, gs: Sequence[Gesture]) -> None:
        for g in gs:
            self.add(g)

    def clear(self) -> None:
        self._g.clear()
        self.end = 0.0

    def release(self, tag: float, t: float) -> None:
        self._g = [g for g in self._g if g.tag != tag or t > g.t0 - g.ramp_in]
        for g in self._g:
            if g.tag != tag or g.t1 <= t:
                continue
            if t < g.t0:
                a = activation(g, t)
                u = math.acos(js_min(1.0, js_max(-1.0, 2.0 * a - 1.0))) / _PI
                g.t1 = t - u * js_max(1e-6, g.ramp_out)
                g.t0 = g.t1
            else:
                g.t1 = t
        end = 0.0
        for g in self._g:
            end = js_max(end, _end(g))
        self.end = end

    def prune(self, t: float) -> None:
        if not self._g:
            return
        self._g = [g for g in self._g if _end(g) > t]

    def size(self) -> int:
        return len(self._g)

    def gestures(self) -> list[Gesture]:
        return [g.gesture() for g in self._g]

    def blend(self, key: int, t: float) -> tuple[float, float, float]:
        wsum, vsum, speed = 0.0, 0.0, 1.0
        for g in self._g:
            target = None
            for i, v in g.pairs:
                if i == key:
                    target = v
                    break
            if target is None:
                continue
            act = activation(g, t)
            gs = g.speed
            if gs is not None and gs != 0.0 and gs > speed and t >= g.t0 - g.ramp_in and t <= _end(g):
                speed = gs
            if act <= 0.0:
                continue
            w = act * g.strength
            wsum += w
            vsum += w * target
        return (vsum / wsum if wsum > 0.0 else 0.0, wsum, speed)

    def blend_all(self, t: float) -> Blend:
        wsum = [0.0] * N_CONTROL
        vsum = [0.0] * N_CONTROL
        speed = [1.0] * N_CONTROL
        for g in self._g:
            lo = g.t0 - g.ramp_in
            gs0 = g.speed
            in_speed = gs0 is not None and gs0 != 0.0 and t >= lo and t <= _end(g)
            act = activation(g, t)
            if act <= 0.0 and not in_speed:
                continue
            w = act * g.strength
            gs = gs0 if gs0 is not None else 0.0
            for i, target in g.pairs:
                if in_speed and gs > speed[i]:
                    speed[i] = gs
                if act <= 0.0:
                    continue
                wsum[i] += w
                vsum[i] += w * target
        value = [vsum[i] / wsum[i] if wsum[i] > 0.0 else 0.0 for i in range(N_CONTROL)]
        return Blend(np.array(value), np.array(wsum), np.array(speed))
