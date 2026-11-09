"""JavaScript number semantics of the template's math.rs, for plain Python callers.

The scalar helpers come from :mod:`nbmath` (single implementation, also compiled by numba in workers);
this module adds the stateful Rng/Drift classes, the note/pitch conversions and IEEE-safe wrappers of
``math`` functions that raise in CPython where Rust returns inf/NaN. No numpy import.
"""

from __future__ import annotations

import math
from typing import Iterable

from .nbmath import (  # noqa: F401 - re-exported API
    clamp,
    fdiv,
    js_max,
    js_min,
    js_round,
    js_sign,
    js_trunc_i32,
    rng_gauss,
    rng_next,
    smoothstep,
)

PI = math.pi
TAU = PI * 2.0  # math.rs:11


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def fmod(a: float, b: float) -> float:
    """Rust ``a % b`` on f64 (C fmod: sign of ``a``); NaN instead of ValueError. Not in nbmath because numba has
    no ``math.fmod`` -- JIT files use ``numpy.fmod`` (same semantics, including NaN for a zero divisor)."""
    if b == 0.0 or a != a or b != b or math.isinf(a):
        return math.nan
    if math.isinf(b):
        return a
    return math.fmod(a, b)


def exp(x: float) -> float:
    """math.exp without OverflowError (Rust/libm returns +inf)."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def pow(x: float, y: float) -> float:  # noqa: A001 - mirrors math.rs `pow`
    """math.pow with IEEE results instead of OverflowError/ValueError (libm ``pow``)."""
    try:
        return math.pow(x, y)
    except OverflowError:
        odd = math.isfinite(y) and y == math.floor(y) and math.fmod(y, 2.0) != 0.0
        return -math.inf if (x < 0.0 and odd) else math.inf
    except ValueError:
        if x == 0.0:  # 0 ** negative
            odd = y == math.floor(y) and math.fmod(y, 2.0) != 0.0
            return math.copysign(math.inf, x) if odd else math.inf
        return math.nan


def _log(f, x: float) -> float:
    if x != x:
        return x
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return math.nan
    return f(x)


def ln(x: float) -> float:
    return _log(math.log, x)


def log2(x: float) -> float:
    return _log(math.log2, x)


def log10(x: float) -> float:
    return _log(math.log10, x)


def midi_to_hz(m: float) -> float:
    """440 * pow(2, (m - 69) / 12) (math.rs midi_to_hz)."""
    return 440.0 * pow(2.0, (m - 69.0) / 12.0)


def hz_to_midi(f: float) -> float:
    """69 + 12 * log2(max(1e-6, f) / 440) (math.rs hz_to_midi)."""
    return 69.0 + 12.0 * log2(js_max(1e-6, f) / 440.0)


def cents_to_ratio(c: float) -> float:
    return pow(2.0, c / 1200.0)


def db_to_gain(db: float) -> float:
    return pow(10.0, db / 20.0)


def gain_to_db(g: float) -> float:
    return 20.0 * log10(js_max(1e-12, g))


def fold_sum(xs: Iterable[float]) -> float:
    """Naive left fold from 0.0, as the template sums (never ``sum()``: compensated since 3.12)."""
    s = 0.0
    for x in xs:
        s += x
    return s


class Rng:
    """mulberry32 with 32-bit state, uniform in [0, 1) (math.rs:175-213)."""

    __slots__ = ("state",)

    def __init__(self, seed: int = 0x9E3779B9) -> None:
        s = int(seed) & 0xFFFFFFFF
        self.state = s if s != 0 else 1  # public: kernels copy it into int64 state slots

    def next(self) -> float:
        self.state, v = rng_next(self.state)
        return v

    def bipolar(self) -> float:
        """Uniform in [-1, 1)."""
        return self.next() * 2.0 - 1.0

    def gauss(self) -> float:
        """Approximately standard normal: (a+b+c+d-2.0)*1.7320508, drawn left to right."""
        self.state, v = rng_gauss(self.state)
        return v


class Drift:
    """1/f-ish slow random drift: three filtered noises at octave-spaced rates (math.rs:233-260)."""

    __slots__ = ("y", "a", "rng")

    def __init__(self, control_rate: float, base_hz: float, seed: int) -> None:
        f = (base_hz, base_hz * 3.0, base_hz * 9.0)
        self.y = [0.0, 0.0, 0.0]
        self.a = [exp((-TAU * fi) / control_rate) for fi in f]
        self.rng = Rng(seed)

    def step(self) -> float:
        s = 0.0
        for i in range(3):
            a = self.a[i]
            self.y[i] = a * self.y[i] + math.sqrt(1.0 - a * a) * self.rng.gauss()
            s += self.y[i] / (i + 1.0)
        return s / 1.4
