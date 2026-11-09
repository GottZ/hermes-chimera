"""K-2.8: gesture score (score.rs): activation shape, blend_all == per-key blend (the template's own unit test),
release of tagged gestures (a released note never rises again; the ramp-out starts from the present activation,
continuously), pruning, capacity -- and the control model's prune schedule: only when the clock crosses a 0.1-s
boundary, prune(t - 0.2) (human_voice.rs:458-460).
"""

import math

import numpy as np
import pytest

from chimera_vocal.core.types import Gesture


def G(t0, t1, ri, ro, strength, targets, speed=None, hold=None, tag=None, label=None):
    return Gesture(t0, t1, ri, ro, strength, dict(targets), speed, hold, label, tag)


def test_activation_shape():
    from chimera_vocal.control.score import activation

    g = G(0.2, 0.5, 0.1, 0.2, 1.0, {"jaw": 1.0})
    assert activation(g, 0.1) == 0.0 and activation(g, 0.05) == 0.0
    assert activation(g, 0.15) == 0.5 - 0.5 * math.cos(math.pi * ((0.15 - (0.2 - 0.1)) / 0.1))
    assert activation(g, 0.2) == 1.0 and activation(g, 0.35) == 1.0 and activation(g, 0.5) == 1.0
    assert activation(g, 0.6) == 0.5 + 0.5 * math.cos(math.pi * ((0.6 - 0.5) / 0.2))
    assert activation(g, 0.7) == 0.0 and activation(g, 0.9) == 0.0
    z = G(0.2, 0.3, 0.0, 0.0, 1.0, {"jaw": 1.0})            # zero ramps: the window is open (t0 - 0, t1 + 0)
    assert activation(z, 0.2) == 0.0 and activation(z, 0.25) == 1.0 and activation(z, 0.3) == 0.0


def test_blend_all_equals_per_key_blend():
    """Port of score.rs:443-460 (exact)."""
    from chimera_vocal.control.score import GestureScore
    from chimera_vocal.core.keys import N_CONTROL

    s = GestureScore()
    s.add(G(0.1, 0.3, 0.05, 0.05, 1.0, {"tbX": 3.0, "f0": 120.0}, speed=2.0))
    s.add(G(0.2, 0.5, 0.1, 0.1, 3.0, {"tbX": 4.0, "jaw": 1.0}, speed=1.5, hold=0.2))
    s.add(G(0.25, 0.6, 0.05, 0.1, 4.0, {"velum": 0.45, "breath": 1.0}))
    for i in range(80):
        t = i * 0.01
        b = s.blend_all(t)
        for k in range(N_CONTROL):
            assert (b.value[k], b.weight[k], b.speed[k]) == s.blend(k, t), (k, t)


def test_blend_values():
    from chimera_vocal.control.score import GestureScore
    from chimera_vocal.core.keys import CONTROL_INDEX

    s = GestureScore()
    s.add(G(0.0, 1.0, 0.1, 0.1, 2.0, {"jaw": 1.0}))
    s.add(G(0.0, 1.0, 0.1, 0.1, 6.0, {"jaw": 2.0}))
    b = s.blend_all(0.5)
    j = CONTROL_INDEX["jaw"]
    assert b.weight[j] == 8.0 and b.value[j] == (2.0 * 1.0 + 6.0 * 2.0) / 8.0
    assert b.weight[CONTROL_INDEX["tbX"]] == 0.0 and b.value[CONTROL_INDEX["tbX"]] == 0.0
    assert set(np.asarray(b.speed).tolist()) == {1.0}


def _release_case(t_rel):
    from chimera_vocal.control.score import GestureScore, activation

    s = GestureScore()
    note = G(0.30, 1.00, 0.08, 0.12, 1.0, {"f0": 220.0}, tag=7.0)
    other = G(0.30, 1.00, 0.08, 0.12, 1.0, {"f0": 330.0}, tag=8.0)
    s.add_all([note, other])
    a_before = activation(s.gestures()[0], t_rel)
    s.release(7.0, t_rel)
    return s, a_before


@pytest.mark.parametrize("t_rel", [0.23, 0.25, 0.27, 0.29])
def test_release_while_ramping_in_never_rises_again(t_rel):
    """K-2.8: released during the ramp-in, the note turns round: continuous at the release, non-increasing after,
    0 after its ramp-out; the other tag is untouched."""
    from chimera_vocal.control.score import activation

    s, a_before = _release_case(t_rel)
    g, other = s.gestures()
    assert other.t0 == 0.30 and other.t1 == 1.00
    a_at = activation(g, t_rel)
    assert abs(a_at - a_before) < 1e-9, (a_at, a_before)
    prev = a_at
    for i in range(1, 400):
        a = activation(g, t_rel + i * 0.0005)
        assert a <= prev + 1e-15, (i, a, prev)
        prev = a
    assert activation(g, t_rel + 0.12 + 1e-9) == 0.0
    assert s.end == max(g.t1 + g.ramp_out, other.t1 + other.ramp_out)


def test_release_cases():
    from chimera_vocal.control.score import GestureScore

    s = GestureScore()
    s.add_all([G(0.5, 0.9, 0.1, 0.1, 1.0, {"f0": 200.0}, tag=1.0),     # not begun at 0.3 -> dropped
               G(0.1, 0.9, 0.1, 0.1, 1.0, {"f0": 200.0}, tag=1.0),     # full strength -> t1 = t
               G(0.0, 0.2, 0.05, 0.3, 1.0, {"f0": 200.0}, tag=1.0),    # already ending (t1 <= t) -> unchanged
               G(0.1, 0.9, 0.1, 0.1, 1.0, {"f0": 200.0}, tag=2.0)])    # other tag
    s.release(1.0, 0.3)
    gs = s.gestures()
    assert len(gs) == 3
    assert (gs[0].t0, gs[0].t1) == (0.1, 0.3)
    assert (gs[1].t0, gs[1].t1) == (0.0, 0.2)
    assert (gs[2].t0, gs[2].t1) == (0.1, 0.9)
    assert s.end == 0.9 + 0.1


def test_prune_and_end():
    from chimera_vocal.control.score import GestureScore

    s = GestureScore()
    # binary fractions, so every end() below is exact
    s.add_all([G(0.0, 0.125, 0.0, 0.0625, 1.0, {"jaw": 1.0}), G(0.0, 0.125, 0.0, 0.0625, 1.0, {"jaw": 1.0}, hold=0.25),
               G(0.25, 0.375, 0.0625, 0.0625, 1.0, {"jaw": 1.0})])
    assert s.end == 0.4375 and s.size() == 3
    s.prune(0.1875)                       # end() <= t is dropped (score.rs:362 keeps end() > t)
    assert [g.release_hold for g in s.gestures()] == [0.25, None]
    s.prune(0.4375)
    assert s.size() == 0 and s.end == 0.4375      # prune leaves `end` alone
    s.clear()
    assert s.end == 0.0


def test_capacity():
    from chimera_vocal.control.score import CAPACITY, GestureScore

    s = GestureScore()
    g = G(0.0, 1.0, 0.1, 0.1, 1.0, {"jaw": 1.0})
    for _ in range(CAPACITY + 3):
        s.add(g)
    assert CAPACITY == 16384 and s.size() == CAPACITY and s.dropped == 3


def test_model_prunes_only_at_tenth_second_boundaries(c2):
    """K-2.8 (control side): a short gesture leaves the model's score exactly at the first control step whose start
    time t crosses a 0.1-s boundary (js_trunc_i32(t*10) != js_trunc_i32((t-dt)*10)) with t - 0.2 >= end."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel
    from chimera_vocal.core.body import BODY_MALE
    from chimera_vocal.core.nbmath import js_trunc_i32

    g1 = G(0.050, 0.060, 0.01, 0.01, 1.0, {"jaw": 1.0})              # end 0.07
    g2 = G(0.100, 0.130, 0.01, 0.02, 1.0, {"jaw": 0.8}, hold=0.034)  # end 0.184
    m = ControlModel(BODY_MALE, c2.tract_index("male"), geometry=GEOMETRY_VEC)
    m.schedule([g1, g2])
    dt = m.dt
    assert dt == c2.DT_48K
    sizes, expected = [], []
    alive, t = 2, 0.0
    ends = sorted([0.07, 0.184])
    for _ in range(400):
        if alive and js_trunc_i32(t * 10.0) != js_trunc_i32((t - dt) * 10.0):
            alive = sum(1 for e in ends if e > t - 0.2)
        expected.append(alive)
        m.advance(1)
        sizes.append(m.score.size())
        t += dt
    assert sizes == expected
    assert expected.index(1) > 0 and expected.index(0) > expected.index(1)
