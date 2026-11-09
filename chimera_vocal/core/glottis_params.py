"""Source parameters -> glottis parameters: ONE formula for C2 (track rows) and C3 (immediate set_source).

Port of HumanVoice::apply_source (voice/human_voice.rs:308-341). Field order = GlottisParams
(sources/glottis.rs:13-49). Imports numpy (load on demand only).
"""

from __future__ import annotations

import numpy as np

from .nbmath import clamp, js_max
from .types import DEFAULT_SOURCE, SourceParams

(G_F0, G_AMPLITUDE, G_ABDUCTION, G_CHINK, G_OPEN_QUOTIENT, G_SKEW, G_CLOSURE_SHARPNESS, G_EFFORT, G_JITTER,
 G_SHIMMER, G_VIBRATO_DEPTH, G_VIBRATO_RATE, G_DRIFT, G_FRY, G_DIPLOPHONIA, G_CLOSURE, G_CHAOS) = range(17)
N_GLOTTIS = 17

GLOTTIS_FIELDS: tuple[str, ...] = (
    "f0", "amplitude", "abduction", "chink", "open_quotient", "skew", "closure_sharpness", "effort", "jitter",
    "shimmer", "vibrato_depth", "vibrato_rate", "drift", "fry", "diplophonia", "closure", "chaos",
)

# sources/glottis.rs DEFAULT_GLOTTIS
DEFAULT_GLOTTIS: tuple[float, ...] = (
    110.0, 0.16, 0.0, 0.006, 0.62, 0.62, 0.6, 0.7, 0.004, 0.03, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0,
)

# the fields apply_source() writes; f0, closure and chaos stay untouched
WRITTEN: tuple[int, ...] = (
    G_AMPLITUDE, G_ABDUCTION, G_CHINK, G_OPEN_QUOTIENT, G_CLOSURE_SHARPNESS, G_SKEW, G_FRY, G_JITTER, G_SHIMMER,
    G_DIPLOPHONIA, G_VIBRATO_DEPTH, G_VIBRATO_RATE, G_DRIFT, G_EFFORT,
)


def apply_source_params(src: SourceParams, body_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (values float64[17], set_mask bool[17], base_abduction).

    Exactly the fields apply_source() writes are set and flagged in the mask; unwritten slots hold NaN so
    a caller that ignores the mask fails loudly. Arithmetic in the template's operation order.
    """
    s = src
    k2 = body_scale * body_scale
    v = np.full(N_GLOTTIS, np.nan, dtype=np.float64)
    v[G_AMPLITUDE] = 0.17 * k2 * (1.0 if s.voiced else 0.0)
    v[G_ABDUCTION] = (0.1 * s.breathiness * s.breathiness if s.voiced else 0.2 + 0.1 * s.breathiness) * k2
    base_abduction = float(v[G_ABDUCTION])
    v[G_CHINK] = (0.008 + 0.03 * s.breathiness) * k2
    v[G_OPEN_QUOTIENT] = 0.78 - 0.36 * s.tension
    bright = s.brightness - DEFAULT_SOURCE.brightness
    by_tension = 0.4 * s.tension + 1.375 * js_max(0.0, s.tension - 0.6)
    v[G_CLOSURE_SHARPNESS] = clamp(by_tension + 0.8 * bright, 0.0, 1.0)
    v[G_SKEW] = 0.56 + 0.16 * s.tension + 0.2 * bright
    v[G_FRY] = s.fry
    v[G_JITTER] = 0.0021 + 0.0245 * s.roughness
    v[G_SHIMMER] = 0.02 + 0.22 * s.roughness
    v[G_DIPLOPHONIA] = (s.roughness - 0.5) * 1.4 if s.roughness > 0.5 else 0.0
    v[G_VIBRATO_DEPTH] = s.vibrato_depth
    v[G_VIBRATO_RATE] = s.vibrato_rate
    v[G_DRIFT] = s.drift
    v[G_EFFORT] = clamp((abs(s.pressure) - 3.0) / 10.0, 0.0, 1.0)
    mask = np.zeros(N_GLOTTIS, dtype=np.bool_)
    mask[list(WRITTEN)] = True
    return v, mask, base_abduction
