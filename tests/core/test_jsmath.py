"""K-0.1, K-0.2, K-0.5 and the remaining scalar helpers of core.jsmath / core.nbmath."""

import math

import pytest

from chimera_vocal.core import jsmath as jm
from chimera_vocal.core import nbmath as nb

NAN, INF = math.nan, math.inf


def bits_equal(a: float, b: float) -> bool:
    if a != a and b != b:
        return True
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


# K-0.1 -- math.rs tests: halves toward +inf, floor-based
@pytest.mark.parametrize("x, want", [
    (2.5, 3.0), (-2.5, -2.0), (0.49999999999999994, 0.0), (-0.6, -1.0), (0.5, 1.0), (-0.5, -0.0),
    (1.4999999999999998, 1.0), (3.0, 3.0), (-3.5, -3.0), (4503599627370497.0, 4503599627370497.0),
    (1e300, 1e300), (INF, INF), (-INF, -INF),
])
def test_js_round(x, want):
    got = jm.js_round(x)
    assert got == want and isinstance(got, float), (x, got, want)


def test_js_round_nan_and_negative_zero():
    assert math.isnan(jm.js_round(NAN))
    # floor(-0.4) = -1, -0.4 - (-1) = 0.6 >= 0.5 -> -1 + 1 = 0.0 (positive zero, like Rust)
    assert bits_equal(jm.js_round(-0.4), 0.0)
    assert bits_equal(jm.js_round(-0.0), -0.0)


# K-0.2 -- NaN propagation in both argument positions, -0 < +0
@pytest.mark.parametrize("f", [jm.js_min, jm.js_max])
@pytest.mark.parametrize("a, b", [(NAN, 1.0), (1.0, NAN), (NAN, NAN), (NAN, -INF), (INF, NAN)])
def test_js_minmax_nan(f, a, b):
    assert math.isnan(f(a, b))


def test_js_minmax_signed_zero():
    assert bits_equal(jm.js_min(0.0, -0.0), -0.0)
    assert bits_equal(jm.js_min(-0.0, 0.0), -0.0)
    assert bits_equal(jm.js_max(0.0, -0.0), 0.0)
    assert bits_equal(jm.js_max(-0.0, 0.0), 0.0)
    assert jm.js_min(1.0, 2.0) == 1.0 and jm.js_max(1.0, 2.0) == 2.0
    assert jm.js_min(-INF, 3.0) == -INF and jm.js_max(-INF, 3.0) == 3.0


# K-0.5 -- naive left fold (CPython >= 3.12 sum() is compensated)
def test_fold_sum():
    assert jm.fold_sum([1e16, 1.0, -1e16]) == 0.0
    assert jm.fold_sum([0.1] * 10) == 0.9999999999999999
    assert jm.fold_sum([]) == 0.0


def test_js_trunc_i32():
    assert jm.js_trunc_i32(3.9) == 3 and jm.js_trunc_i32(-3.9) == -3
    assert jm.js_trunc_i32(2147483648.0) == -2147483648      # i64 then wrap to i32
    assert jm.js_trunc_i32(4294967296.0 + 5.0) == 5
    assert jm.js_trunc_i32(NAN) == 0 and jm.js_trunc_i32(INF) == 0 and jm.js_trunc_i32(-INF) == 0
    assert jm.js_trunc_i32(1e30) == -1                     # saturates to i64::MAX, low 32 bits = 0xFFFFFFFF


def test_js_sign():
    assert jm.js_sign(5.0) == 1.0 and jm.js_sign(-0.1) == -1.0
    assert bits_equal(jm.js_sign(-0.0), -0.0) and bits_equal(jm.js_sign(0.0), 0.0)
    assert math.isnan(jm.js_sign(NAN))


def test_clamp_and_smoothstep():
    assert math.isnan(jm.clamp(NAN, 0.0, 1.0))
    assert jm.clamp(-1.0, 0.0, 1.0) == 0.0 and jm.clamp(2.0, 0.0, 1.0) == 1.0
    assert jm.smoothstep(0.0, 1.0, 0.5) == 0.5
    assert jm.smoothstep(0.0, 1.0, -1.0) == 0.0 and jm.smoothstep(0.0, 1.0, 9.0) == 1.0
    assert jm.smoothstep(1.0, 1.0, 2.0) == 1.0     # (x-e0)/0 = +inf -> clamp 1, no ZeroDivisionError
    assert jm.smoothstep(1.0, 1.0, 0.0) == 0.0     # -inf -> 0
    assert math.isnan(jm.smoothstep(1.0, 1.0, 1.0))  # 0/0 = NaN passes the clamp, as in Rust


def test_ieee_division_and_fmod():
    assert jm.fdiv(1.0, 0.0) == INF and jm.fdiv(-1.0, 0.0) == -INF and jm.fdiv(1.0, -0.0) == -INF
    assert math.isnan(jm.fdiv(0.0, 0.0)) and math.isnan(jm.fdiv(NAN, 0.0))
    assert jm.fmod(-7.0, 3.0) == -1.0 and jm.fmod(7.0, -3.0) == 1.0   # Rust %, not Python %
    assert math.isnan(jm.fmod(1.0, 0.0)) and jm.fmod(2.5, INF) == 2.5


def test_ieee_wrappers():
    assert jm.exp(1000.0) == INF
    assert jm.pow(10.0, 400.0) == INF and jm.pow(-10.0, 401.0) == -INF
    assert jm.pow(0.0, -1.0) == INF and jm.pow(-0.0, -1.0) == -INF
    assert math.isnan(jm.pow(-8.0, 1.0 / 3.0))
    assert jm.ln(0.0) == -INF and math.isnan(jm.log2(-1.0)) and jm.log10(100.0) == 2.0


def test_pitch_conversions():
    assert jm.midi_to_hz(69.0) == 440.0
    assert jm.midi_to_hz(57.0) == 220.0
    assert jm.hz_to_midi(440.0) == 69.0
    assert jm.hz_to_midi(0.0) == 69.0 + 12.0 * math.log2(1e-6 / 440.0)
    assert jm.cents_to_ratio(1200.0) == 2.0
    assert jm.lerp(1.0, 3.0, 0.25) == 1.5


def test_nbmath_is_the_implementation():
    """jsmath re-exports the nbmath functions (single implementation)."""
    for name in ("js_min", "js_max", "js_round", "js_sign", "js_trunc_i32", "clamp", "smoothstep", "rng_next"):
        assert getattr(jm, name) is getattr(nb, name), name
