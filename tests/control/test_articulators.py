"""K-2.9: articulator dynamics (articulators.rs): critically damped tracking with semi-implicit Euler at the control
step dt = 64/48000; 90 % rise 48-104 ms at speed 1 (natural frequencies 13-6 Hz), no overshoot, the
template's own trajectory, target clamping, the bounded velum blend and the speed multiplier.
"""

import json
import math

import numpy as np
import pytest

DT = 64.0 / 48000.0

# The template's articulator positions for `sustain-aa-male` (HumanVoice::new(48000, male, 2), set_articulation of
# vowel_artic("ɑ", male), no gestures, no contacts): artic_x after control block b, keys jaw, tbX, tbY, ttX, ttY,
# root, lipAp, larynx. Source: $CHIMERA_LAB_REFS/tracks/track-sustain-aa-male.full.json (export of template
# 092b16ce12b9, MIT, Copyright (c) 2026 Jameson Hodge).
SUSTAIN_KEYS = ("jaw", "tbX", "tbY", "ttX", "ttY", "root", "lipAp", "larynx")
SUSTAIN_X = {
    0: (0.6076149481067961, 2.8911265150319037, 6.838296702058023, 6.241412821929657, 6.810460479056529,
        -0.002012451815799724, 0.7110399201585162, 0.0009747133576698951),
    1: (0.621515502878552, 2.875253344760877, 6.817361472930632, 6.241347349860832, 6.81033166988982,
        -0.005722444291568858, 0.7309912747401466, 0.0027891100555979325),
    2: (0.6405509413601015, 2.8539482812191626, 6.789262083885865, 6.241112924071012, 6.80987034996111,
        -0.010852903419467458, 0.7580421942424744, 0.005322488012026014),
    5: (0.7191973566879406, 2.769443896392526, 6.677808692508592, 6.238171324603756, 6.80408486233833,
        -0.03248013065188819, 0.867559023123303, 0.01621633669120001),
    20: (1.193039853873647, 2.3350836589245287, 6.104928105961963, 6.118967407895373, 6.572986308808558,
         -0.1738134683893276, 1.4766212154710467, 0.09351540374443791),
    47: (1.5859571911926185, 2.066263716460267, 5.7503797080421295, 5.743343987220938, 5.86958801772317,
         -0.3104153269797861, 1.9119172404985476, 0.18098797922170676),
    80: (1.672936956060512, 2.0256392847032902, 5.696799876636004, 5.5576993173248805, 5.534295563288697,
         -0.3474689862662364, 1.9912143857161302, 0.21030616357948823),
    200: (1.6849949278179421, 2.0220003923810705, 5.692000517513986, 5.518946707263987, 5.4656896312017755,
          -0.35399445970183824, 1.9999981411516108, 0.21698832765999249),
    749: (1.6849999999999987, 2.022000000000002, 5.692000000000005, 5.518926366769177, 5.465653666863971,
          -0.3539999999999996, 1.9999999999999987, 0.2169999999999998),
}
AA_MALE = {"jaw": 1.685, "larynx": 0.217, "lipAp": 2.0, "lipProt": 0.0, "lipWidth": 1.0, "root": -0.354,
           "tbX": 2.022, "tbY": 5.692, "ttCurl": 0.0}


def _oracle_step(x, v, tgt, omega, dt):
    """articulators.rs:141-146, the template's two lines."""
    acc = omega * omega * (tgt - x) - 2.0 * omega * v
    v = v + acc * dt
    return x + v * dt, v


def test_specs_and_neutral_are_the_template_tables(c2):
    from chimera_vocal.control.articulators import ARTIC_SPECS, NEUTRAL

    assert [tuple(s) for s in ARTIC_SPECS] == [tuple(s) for s in c2.TEMPLATE_ARTIC_SPECS]
    assert tuple(NEUTRAL) == c2.TEMPLATE_NEUTRAL


def test_new_bank_matches_v2_initial(c2):
    """ArticulatorBank::new = NEUTRAL with the tip at rest_tip (articulators.rs:79-87); V2 `initial` holds the
    template's artic_x / artic_target (T2: rest_tip uses cos/sin)."""
    from chimera_vocal.control.articulators import ArticulatorBank, rest_tip

    b = ArticulatorBank()
    ini = c2.v2_entry("male")["initial"]
    assert c2.worst_rel(b.x, ini["artic_x"], c2.T2_FLOOR) < c2.T2_TOL
    assert c2.worst_rel(b.target, ini["artic_target"], c2.T2_FLOOR) < c2.T2_TOL
    assert b.tip_free is True and list(b.v) == [0.0] * 17 and list(b.speed) == [1.0] * 17
    tx, ty = rest_tip(0.6, 2.9, 6.85)
    assert c2.worst_rel([tx, ty], ini["artic_x"][3:5], c2.T2_FLOOR) < c2.T2_TOL


@pytest.mark.parametrize("hz,blocks", [(6.0, 78), (8.0, 59), (9.0, 52), (10.0, 47), (11.0, 43), (12.0, 39),
                                       (13.0, 36)])
def test_articulator_rise(hz, blocks):
    """K-2.9: the step response of every key with natural frequency `hz` follows the semi-implicit recurrence
    exactly (T6), reaches 90 % after `blocks` steps (48-104 ms band), and never overshoots."""
    from chimera_vocal.control.articulators import ARTIC_SPECS, ArticulatorBank
    from chimera_vocal.core.keys import ARTIC_KEYS

    keys = [i for i, s in enumerate(ARTIC_SPECS) if s.hz == hz]
    assert keys, hz
    for i in keys:
        lo = ARTIC_SPECS[i].min
        b = ArticulatorBank()
        b.tip_free = False
        b.snap({ARTIC_KEYS[i]: lo})
        b.set_target({ARTIC_KEYS[i]: lo + 1.0})
        x, v, om = lo, 0.0, 2.0 * math.pi * hz
        t90, peak = None, -math.inf
        for n in range(1, 2001):
            b.step(DT, None, 1.0)
            x, v = _oracle_step(x, v, lo + 1.0, om, DT)
            assert b.x[i] == x and b.v[i] == v, (ARTIC_KEYS[i], n, b.x[i], x)
            if t90 is None and b.x[i] - lo >= 0.9:
                t90 = n
            peak = max(peak, b.x[i])
        assert t90 == blocks, (ARTIC_KEYS[i], t90)
        assert 0.048 - 1e-12 <= t90 * DT <= 0.104 + 1e-12
        assert peak <= lo + 1.0 + 1e-12, (ARTIC_KEYS[i], peak)


def test_template_trajectory_sustain_aa(c2):
    """The bank alone reproduces the template's articulator positions of `sustain-aa-male` (T2; the template's
    rest_tip runs on Rust libm, so the tip may differ in the last ulp)."""
    from chimera_vocal.control.articulators import ArticulatorBank
    from chimera_vocal.core.keys import ARTIC_INDEX

    b = ArticulatorBank()
    b.set_target(AA_MALE)
    cols = [ARTIC_INDEX[k] for k in SUSTAIN_KEYS]
    for n in range(750):
        b.step(DT, None, 1.0)
        if n in SUSTAIN_X:
            w = c2.worst_rel(b.x[cols], SUSTAIN_X[n], c2.T2_FLOOR)
            assert w < c2.T2_TOL, (n, w, b.x[cols].tolist())


@pytest.mark.lab
def test_template_trajectory_sustain_aa_full(c2, lab_refs):
    """All 17 positions and velocities of all 750 blocks (positions T2; velocities relative with floor 1e-3 cm/s,
    because they decay to 0 and carry the template's libm ulps)."""
    from chimera_vocal.control.articulators import ArticulatorBank

    full = json.loads((lab_refs / "tracks" / "track-sustain-aa-male.full.json").read_text())
    b = ArticulatorBank()
    b.set_target(AA_MALE)
    for n in range(750):
        b.step(DT, None, 1.0)
        assert c2.worst_rel(b.x, full["blocks"]["artic_x"][n], c2.T2_FLOOR) < c2.T2_TOL, n
        assert c2.worst_rel(b.v, full["blocks"]["artic_v"][n], 1e-3) < c2.T2_TOL, n


def test_set_target_clamps_and_ignores_non_finite():
    from chimera_vocal.control.articulators import ARTIC_SPECS, ArticulatorBank
    from chimera_vocal.core.keys import ARTIC_INDEX

    b = ArticulatorBank()
    before = b.target.copy()
    b.set_target({"jaw": 99.0, "velum": -5.0, "tbX": float("nan"), "root": float("inf")})
    assert b.target[ARTIC_INDEX["jaw"]] == ARTIC_SPECS[ARTIC_INDEX["jaw"]].max
    assert b.target[ARTIC_INDEX["velum"]] == ARTIC_SPECS[ARTIC_INDEX["velum"]].min
    assert b.target[ARTIC_INDEX["tbX"]] == before[ARTIC_INDEX["tbX"]]
    assert b.target[ARTIC_INDEX["root"]] == before[ARTIC_INDEX["root"]]
    b.snap({"lipAp": 1.5})
    assert b.x[ARTIC_INDEX["lipAp"]] == 1.5 and not b.v.any()


def test_blend_rules():
    """articulators.rs:119-139: velum bounded (tgt += min(1, w/2)(value - tgt)); others weighted mean with the base
    weight; speed multiplied only where weight > 0; effective target clamped."""
    from chimera_vocal.control.articulators import ArticulatorBank
    from chimera_vocal.control.score import Blend
    from chimera_vocal.core.keys import ARTIC_INDEX, N_CONTROL

    value, weight, speed = np.zeros(N_CONTROL), np.zeros(N_CONTROL), np.ones(N_CONTROL)
    iv, ij, ir = ARTIC_INDEX["velum"], ARTIC_INDEX["jaw"], ARTIC_INDEX["root"]
    value[iv], weight[iv] = 0.45, 0.5
    value[ij], weight[ij], speed[ij] = 1.6, 3.0, 2.0
    speed[ir] = 5.0                       # weight 0: speed must not apply
    b = ArticulatorBank()
    b.step(DT, Blend(value, weight, speed), 0.05)
    assert b.effective[iv] == -0.15 + min(1.0, 0.5 / 2.0) * (0.45 - -0.15)
    assert b.effective[ij] == (0.6 * 0.05 + 1.6 * 3.0) / (0.05 + 3.0)
    om = 2.0 * math.pi * 10.0 * 2.0
    x, v = _oracle_step(0.6, 0.0, b.effective[ij], om, DT)
    assert b.x[ij] == x and b.v[ij] == v
    om_r = 2.0 * math.pi * 9.0
    x, v = _oracle_step(0.0, 0.0, 0.0, om_r, DT)
    assert b.x[ir] == x
    value[ij], weight[ij] = 50.0, 1e9
    b2 = ArticulatorBank()
    b2.step(DT, Blend(value, weight, speed), 0.05)
    assert b2.effective[ij] == 2.6


def test_tip_follows_rest_tip_only_when_free():
    from chimera_vocal.control.articulators import ArticulatorBank, rest_tip
    from chimera_vocal.core.keys import ARTIC_INDEX

    b = ArticulatorBank()
    b.set_target({"jaw": 2.0})
    for _ in range(30):
        b.step(DT, None, 1.0)
    jx = b.x.copy()
    b.step(DT, None, 1.0)
    tx, ty = rest_tip(jx[ARTIC_INDEX["jaw"]], jx[ARTIC_INDEX["tbX"]], jx[ARTIC_INDEX["tbY"]])
    assert (b.target[ARTIC_INDEX["ttX"]], b.target[ARTIC_INDEX["ttY"]]) == (tx, ty)
    b.tip_free = False
    b.set_target({"ttX": 7.0})
    b.step(DT, None, 1.0)
    assert b.target[ARTIC_INDEX["ttX"]] == 7.0
