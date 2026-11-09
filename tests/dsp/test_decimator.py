"""K-3.3 test_decimator: coefficients = V2d (≤ 1e-15), impulse response of the first outputs = V2d through the
kernels' output chain (first output at the m-th push); m = 1 is a pass-through."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chimera_vocal.dsp import decimator
from support import rel, vector

V2D = vector("decimator")


def test_coefficients_m2():
    m2 = V2D["m2"]
    c = decimator.coefficients(2)
    assert c.dtype == np.float64 and c.shape == (m2["taps"],)
    worst = max(rel(float(a), b, 1e-300) for a, b in zip(c, m2["coeffs"]))
    assert worst <= 1e-15, worst
    assert decimator.new_buffer(c).shape == (m2["buf_len"],)
    d = decimator.design_lowpass(49, 0.46 / 2.0, 8.5)
    assert d.tolist() == c.tolist()


def test_coefficients_m1():
    c = decimator.coefficients(1)
    assert c.tolist() == V2D["m1"]["coeffs"] == [1.0]
    assert decimator.new_buffer(c).shape == (2,)


def test_bessel_i0():
    assert decimator.bessel_i0(0.0) == 1.0
    assert rel(decimator.bessel_i0(8.5), 683.1619269901156, 1e-300) < 1e-12   # exact series in rationals


def _chain(vals, gain, dc_coef, lim_hold_n, lim_release):
    """Oracle: human_voice.rs:368-393 applied to decimator outputs `vals` (same operation order)."""
    out = []
    dc_x = dc_y = env = 0.0
    hold = 0
    for x in vals:
        v = x * gain
        hp = v - dc_x + dc_coef * dc_y
        dc_x, dc_y = v, hp
        v = hp
        av = abs(v)
        if av >= env:
            env, hold = av, lim_hold_n
        elif hold > 0:
            hold -= 1
        else:
            env *= lim_release
        if env > 0.75:
            v *= 0.75 / env
        al = abs(v)
        if al > 0.85:
            v = math.copysign(1.0, v) * (0.85 + 0.15 * math.tanh((al - 0.85) / 0.15))
        out.append(np.float32(v))
    return out


@pytest.mark.parametrize("m", [2, 1])
def test_impulse_response(kernel_name, m, run):
    case = V2D[f"m{m}"]
    n_push = 2 * len(case["impulse_out"]) * m
    raw = [1.0] + [0.0] * (n_push - 1)
    res = run(kernel_name, "output_chain", raw=raw, m=m)
    counts = res["counts"].tolist()
    pushes = [i for i, c in enumerate(counts) if c]
    assert pushes[: len(case["impulse_push_index"])] == case["impulse_push_index"], "output at the m-th push"
    assert all(c in (0, 1) for c in counts)
    fs_out = 96000.0 / m
    assert rel(float(res["dc_coef"]), math.exp((-2.0 * math.pi * 25.0) / fs_out), 1e-300) <= 1e-15
    assert rel(float(res["lim_release"]), math.exp(-1.0 / (0.2 * fs_out)), 1e-300) <= 1e-15
    assert int(res["lim_hold_n"]) == math.floor(0.025 * fs_out + 0.5)
    want = _chain(case["impulse_out"], 1.0, float(res["dc_coef"]), int(res["lim_hold_n"]), float(res["lim_release"]))
    got = res["out"][: len(want)].tolist()
    assert got == [float(x) for x in want], (got, want)
