"""Kaiser-windowed sinc decimator coefficients (airway/decimator.rs). The push/output runs in the kernels'
``output_chain``.

Ported from chimera-vocal-synth crates/chimera/src/airway/decimator.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import numpy as np

TAPS_PER_PHASE = 24


def bessel_i0(x: float) -> float:
    """Series for I0 (decimator.rs:6-18)."""
    raise NotImplementedError


def design_lowpass(taps: int, cutoff: float, beta: float) -> np.ndarray:
    """Kaiser-windowed sinc low-pass normalised to sum 1 (decimator.rs:21-38); cutoff as a fraction of the input rate."""
    raise NotImplementedError


def coefficients(m: int, taps_per_phase: int = TAPS_PER_PHASE) -> np.ndarray:
    """Decimator::with_taps(m, 24): m = 1 -> [1.0]; else design_lowpass(m·24 + 1, 0.46 / m, 8.5)."""
    raise NotImplementedError


def new_buffer(coeffs: np.ndarray) -> np.ndarray:
    """Zeroed ring buffer of 2·taps (decimator.rs:58)."""
    raise NotImplementedError
