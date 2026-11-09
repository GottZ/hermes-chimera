"""Kaiser-windowed sinc decimator coefficients (airway/decimator.rs). The push/output runs in the kernels'
``output_chain``.

Ported from chimera-vocal-synth crates/chimera/src/airway/decimator.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.nbmath import js_max

TAPS_PER_PHASE = 24
PI = math.pi


def bessel_i0(x: float) -> float:
    """Series for I0 (decimator.rs:6-18)."""
    total = 1.0
    term = 1.0
    for k in range(1, 30):
        q = x / (2.0 * k)
        term *= q * q
        total += term
        if term < 1e-12 * total:
            break
    return total


def design_lowpass(taps: int, cutoff: float, beta: float) -> np.ndarray:
    """Kaiser-windowed sinc low-pass normalised to sum 1 (decimator.rs:21-38); cutoff as a fraction of the input rate."""
    h = np.zeros(taps, dtype=np.float64)
    mid = (taps - 1.0) / 2.0
    i0b = bessel_i0(beta)
    total = 0.0
    for i in range(taps):
        x = i - mid
        sinc = 2.0 * cutoff if x == 0.0 else math.sin(2.0 * PI * cutoff * x) / (PI * x)
        r = (2.0 * i) / (taps - 1.0) - 1.0
        w = bessel_i0(beta * math.sqrt(js_max(0.0, 1.0 - r * r))) / i0b
        h[i] = sinc * w
        total += h[i]
    for i in range(taps):
        h[i] /= total
    return h


def coefficients(m: int, taps_per_phase: int = TAPS_PER_PHASE) -> np.ndarray:
    """Decimator::with_taps(m, 24): m = 1 -> [1.0]; else design_lowpass(m·24 + 1, 0.46 / m, 8.5)."""
    if m == 1:
        return np.ones(1, dtype=np.float64)
    return design_lowpass(m * taps_per_phase + 1, 0.46 / m, 8.5)


def new_buffer(coeffs: np.ndarray) -> np.ndarray:
    """Zeroed ring buffer of 2·taps (decimator.rs:58)."""
    return np.zeros(2 * coeffs.shape[0], dtype=np.float64)
