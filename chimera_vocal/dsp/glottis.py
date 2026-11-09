"""Glottis state and constants (sources/glottis.rs:71-136). The per-sample step lives in the kernels
(``kernel_numba.glottis_block``); the source -> parameter formula in ``core/glottis_params``.

Ported from chimera-vocal-synth crates/chimera/src/sources/glottis.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import math

import numpy as np

from ..core import glottis_params as gp
from ..core.physics import CM_H2O
from ..core.types import Seeds
from . import layout as L

TAU = math.pi * 2.0

STATE_FIELDS: tuple[str, ...] = (
    "phase", "cycle_amp", "cycle_period_scale", "cycle_index", "chaos_x", "vib_phase", "rng_state",
    "drift_y", "drift_a", "drift_rng_state", "drift_value", "drift_counter", "pitch_factor", "voicing", "voicing_sm",
    "dp_sm", "dp_coef", "voice_coef", "area", "f0_now", "threshold", "static_sm", "closure_sm", "amp_sm", "sm_coef",
)
"""Private state of ParametricGlottis by the names of the vector exports (V2 ``glottis_state``, V5 full)."""


def init_glottis(fs: float, seeds: Seeds) -> tuple[np.ndarray, np.ndarray]:
    """ParametricGlottis::new(fs) with the seeds' glottis (jitter) and drift rng: (glot float64[G], glot_i int64[GI])."""
    g = np.zeros(L.G, dtype=np.float64)
    g[: gp.N_GLOTTIS] = gp.DEFAULT_GLOTTIS
    g[L.G_PHASE] = 0.0
    g[L.G_CYCLE_AMP] = 1.0
    g[L.G_CYCLE_PERIOD_SCALE] = 1.0
    g[L.G_CHAOS_X] = 0.37
    g[L.G_PITCH_FACTOR] = 1.0
    g[L.G_DP_COEF] = 1.0 - math.exp(-TAU * 30.0 / fs)
    g[L.G_VOICE_COEF] = 1.0 - math.exp(-TAU * 25.0 / fs)
    g[L.G_F0_NOW] = 110.0
    g[L.G_THRESHOLD] = 2.5 * CM_H2O
    g[L.G_SM_COEF] = 1.0 - math.exp(-1.0 / (0.004 * fs))
    g[L.G_FS] = fs
    control_rate = fs / 32.0
    base = 0.3
    for i, f in enumerate((base, base * 3.0, base * 9.0)):
        g[L.G_DRIFT_A + i] = math.exp((-TAU * f) / control_rate)
    gi = np.zeros(L.GI, dtype=np.int64)
    gi[L.GI_RNG] = (seeds.glottis & 0xFFFFFFFF) or 1
    gi[L.GI_DRIFT_RNG] = (seeds.drift & 0xFFFFFFFF) or 1
    return g, gi


def glottis_state(glot: np.ndarray, glot_i: np.ndarray) -> dict:
    """The private state as a dict keyed like STATE_FIELDS (drift_y/drift_a as 3-lists, ints as int)."""
    f = {"phase": L.G_PHASE, "cycle_amp": L.G_CYCLE_AMP, "cycle_period_scale": L.G_CYCLE_PERIOD_SCALE,
         "chaos_x": L.G_CHAOS_X, "vib_phase": L.G_VIB_PHASE, "drift_value": L.G_DRIFT_VALUE,
         "pitch_factor": L.G_PITCH_FACTOR, "voicing": L.G_VOICING, "voicing_sm": L.G_VOICING_SM, "dp_sm": L.G_DP_SM,
         "dp_coef": L.G_DP_COEF, "voice_coef": L.G_VOICE_COEF, "area": L.G_AREA, "f0_now": L.G_F0_NOW,
         "threshold": L.G_THRESHOLD, "static_sm": L.G_STATIC_SM, "closure_sm": L.G_CLOSURE_SM, "amp_sm": L.G_AMP_SM,
         "sm_coef": L.G_SM_COEF}
    out: dict = {k: float(glot[i]) for k, i in f.items()}
    out["drift_y"] = [float(glot[L.G_DRIFT_Y + i]) for i in range(3)]
    out["drift_a"] = [float(glot[L.G_DRIFT_A + i]) for i in range(3)]
    out["cycle_index"] = int(glot_i[L.GI_CYCLE_INDEX])
    out["drift_counter"] = int(glot_i[L.GI_DRIFT_COUNTER])
    out["rng_state"] = int(glot_i[L.GI_RNG])
    out["drift_rng_state"] = int(glot_i[L.GI_DRIFT_RNG])
    return out


def set_params(glot: np.ndarray, values) -> None:
    """Write the 17 GlottisParams (G_* order) into glot."""
    glot[: gp.N_GLOTTIS] = np.asarray(values, dtype=np.float64)
