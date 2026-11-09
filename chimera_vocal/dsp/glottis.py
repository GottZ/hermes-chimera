"""Glottis state and constants (sources/glottis.rs:71-136). The per-sample step lives in the kernels
(``kernel_numba.glottis_block``); the source -> parameter formula in ``core/glottis_params``.

Ported from chimera-vocal-synth crates/chimera/src/sources/glottis.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import numpy as np

from ..core.types import Seeds

STATE_FIELDS: tuple[str, ...] = (
    "phase", "cycle_amp", "cycle_period_scale", "cycle_index", "chaos_x", "vib_phase", "rng_state",
    "drift_y", "drift_a", "drift_rng_state", "drift_value", "drift_counter", "pitch_factor", "voicing", "voicing_sm",
    "dp_sm", "dp_coef", "voice_coef", "area", "f0_now", "threshold", "static_sm", "closure_sm", "amp_sm", "sm_coef",
)
"""Private state of ParametricGlottis by the names of the vector exports (V2 ``glottis_state``, V5 full)."""


def init_glottis(fs: float, seeds: Seeds) -> tuple[np.ndarray, np.ndarray]:
    """ParametricGlottis::new(fs) with the seeds' glottis (jitter) and drift rng: (glot float64[G], glot_i int64[GI])."""
    raise NotImplementedError


def glottis_state(glot: np.ndarray, glot_i: np.ndarray) -> dict:
    """The private state as a dict keyed like STATE_FIELDS (drift_y/drift_a as 3-lists, ints as int)."""
    raise NotImplementedError


def set_params(glot: np.ndarray, values) -> None:
    """Write the 17 GlottisParams (G_* order) into glot."""
    raise NotImplementedError
