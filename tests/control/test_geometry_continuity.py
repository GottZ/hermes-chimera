"""K-2.10: the area function changes smoothly as the articulators move (port of the template's
tests/geometry_continuity.rs): sweeps of each articulator through its speech range from vowel postures, transitions
between calibrated vowels, the labiodental tuck -- no step that survives refinement to 2^-20 of its interval by more
than 0.05 cm^2 (jump_size, geometry_continuity.rs:42-90). The template's known open case (protruded lower lip tucking,
`#[ignore]` there, 0.41 cm^2 male) is an expected failure, strict: the port must reproduce the template, not fix it.

Control side: in the control track, every block-to-block change of a section area / minimum respects the slew limit
of HumanAirway::commit (human_airway.rs:187-253), and the limiter really engages on the known geometric jump.

Geometry runs through GEOMETRY_VEC (the production numpy path); K-2.1/2.2/2.4 tie it to the reference.
Vowel postures: the template's calibrated table (src/phonetics/vowelTargets.generated.json, MIT, Copyright (c) 2026
Jameson Hodge), subset below.
"""

import json
import math

import numpy as np
import pytest

VOWELS = json.loads('{"male":{"i":{"jaw":0.413,"tbX":3.683,"tbY":7.739,"ttX":6.545,"ttY":7.288,"ttCurl":0,"root":0.514,"lipAp":1.046,"lipProt":0,"lipWidth":1,"larynx":0.798},"ɑ":{"jaw":1.685,"tbX":2.022,"tbY":5.692,"ttX":5.519,"ttY":5.466,"ttCurl":0,"root":-0.354,"lipAp":2,"lipProt":0,"lipWidth":1,"larynx":0.217},"u":{"jaw":0.203,"tbX":2.419,"tbY":7.428,"ttX":6.239,"ttY":7.369,"ttCurl":0,"root":0.402,"lipAp":0.291,"lipProt":1.4,"lipWidth":0.065,"larynx":0.252},"ə":{"jaw":0.505,"tbX":2.944,"tbY":6.685,"ttX":6.29,"ttY":6.837,"ttCurl":0,"root":0.421,"lipAp":0.323,"lipProt":0.352,"lipWidth":0.533,"larynx":0.149},"o":{"jaw":0.668,"tbX":1.519,"tbY":6.592,"ttX":5.801,"ttY":6.66,"ttCurl":0,"root":0.227,"lipAp":0.405,"lipProt":0.597,"lipWidth":0.027,"larynx":0.061},"ɝ":{"jaw":0.976,"tbX":2.727,"tbY":7.285,"ttX":5.642,"ttY":8.632,"ttCurl":1,"root":-0.9,"lipAp":0.654,"lipProt":0.592,"lipWidth":0.347,"larynx":-0.921},"æ":{"jaw":1.516,"tbX":3.373,"tbY":6.016,"ttX":6.002,"ttY":5.726,"ttCurl":0.002,"root":-0.16,"lipAp":1.875,"lipProt":0,"lipWidth":1,"larynx":0.328},"ʊ":{"jaw":0.516,"tbX":2.317,"tbY":7.139,"ttX":6.098,"ttY":6.987,"ttCurl":0,"root":0.478,"lipAp":0.44,"lipProt":1.228,"lipWidth":0.199,"larynx":0.425},"e":{"jaw":0.745,"tbX":3.6,"tbY":7.236,"ttX":6.396,"ttY":6.817,"ttCurl":0,"root":0.269,"lipAp":1.315,"lipProt":0,"lipWidth":1,"larynx":0.506}},"female":{"i":{"jaw":0.459,"tbX":3.754,"tbY":7.647,"ttX":6.55,"ttY":7.215,"ttCurl":0,"root":0.295,"lipAp":1.092,"lipProt":0,"lipWidth":1,"larynx":1.016},"ɑ":{"jaw":1.682,"tbX":1.847,"tbY":5.832,"ttX":5.468,"ttY":5.517,"ttCurl":0,"root":-0.665,"lipAp":2,"lipProt":0,"lipWidth":1,"larynx":0.523},"u":{"jaw":0.266,"tbX":2.243,"tbY":7.381,"ttX":6.164,"ttY":7.296,"ttCurl":0,"root":0.205,"lipAp":0.187,"lipProt":1.4,"lipWidth":0.301,"larynx":0.24},"ə":{"jaw":0.818,"tbX":2.996,"tbY":6.841,"ttX":6.186,"ttY":6.614,"ttCurl":0,"root":0.279,"lipAp":0.599,"lipProt":0.351,"lipWidth":0.544,"larynx":0.14},"o":{"jaw":0.966,"tbX":1.827,"tbY":7.11,"ttX":5.776,"ttY":6.579,"ttCurl":0,"root":0.221,"lipAp":0.288,"lipProt":1.4,"lipWidth":0.304,"larynx":0.46},"ɝ":{"jaw":0.694,"tbX":2.581,"tbY":7.338,"ttX":5.71,"ttY":8.898,"ttCurl":1,"root":-0.9,"lipAp":0.657,"lipProt":0.59,"lipWidth":0.349,"larynx":-0.832},"æ":{"jaw":1.483,"tbX":3.712,"tbY":6.328,"ttX":6.119,"ttY":5.86,"ttCurl":0,"root":-0.094,"lipAp":1.851,"lipProt":0,"lipWidth":1,"larynx":0.854},"ʊ":{"jaw":0.573,"tbX":2.314,"tbY":7.268,"ttX":6.076,"ttY":6.981,"ttCurl":0,"root":0.255,"lipAp":0.371,"lipProt":1.4,"lipWidth":0.301,"larynx":0.382},"e":{"jaw":0.8,"tbX":3.593,"tbY":7.368,"ttX":6.372,"ttY":6.815,"ttCurl":0,"root":0.08,"lipAp":1.369,"lipProt":0,"lipWidth":1,"larynx":0.853}},"child":{"i":{"jaw":0.096,"tbX":3.719,"tbY":7.754,"ttX":6.664,"ttY":7.581,"ttCurl":0.001,"root":0.095,"lipAp":0.386,"lipProt":0.004,"lipWidth":0.655,"larynx":-0.001},"ɑ":{"jaw":1.905,"tbX":1.651,"tbY":6.308,"ttX":5.302,"ttY":5.497,"ttCurl":0,"root":-0.9,"lipAp":2,"lipProt":0.361,"lipWidth":0.952,"larynx":-0.353},"u":{"jaw":0.162,"tbX":2.324,"tbY":7.563,"ttX":6.224,"ttY":7.453,"ttCurl":0,"root":-0.646,"lipAp":0.301,"lipProt":0.637,"lipWidth":0.054,"larynx":-0.327},"ə":{"jaw":0.804,"tbX":2.698,"tbY":7.382,"ttX":6.102,"ttY":6.817,"ttCurl":0,"root":-0.503,"lipAp":0.569,"lipProt":0.359,"lipWidth":0.535,"larynx":-0.379},"o":{"jaw":0.78,"tbX":1.814,"tbY":7.259,"ttX":5.847,"ttY":6.794,"ttCurl":0,"root":-0.566,"lipAp":0.131,"lipProt":1.241,"lipWidth":0.301,"larynx":-0.209},"ɝ":{"jaw":0.05,"tbX":1.859,"tbY":7,"ttX":5.722,"ttY":9.357,"ttCurl":1,"root":-0.9,"lipAp":0.12,"lipProt":1.4,"lipWidth":0.214,"larynx":-1.288},"æ":{"jaw":1.367,"tbX":3.496,"tbY":6.678,"ttX":6.106,"ttY":6.08,"ttCurl":0,"root":-0.534,"lipAp":1.601,"lipProt":0.356,"lipWidth":0.548,"larynx":-0.23},"ʊ":{"jaw":0.498,"tbX":2.338,"tbY":7.482,"ttX":6.111,"ttY":7.122,"ttCurl":0,"root":-0.498,"lipAp":0.335,"lipProt":0.6,"lipWidth":0.279,"larynx":-0.249},"e":{"jaw":0.676,"tbX":3.477,"tbY":7.552,"ttX":6.386,"ttY":6.989,"ttCurl":0,"root":-0.219,"lipAp":0.717,"lipProt":0.351,"lipWidth":0.55,"larynx":-0.14}}}')
VOWEL_KEYS = ("tbX", "tbY", "jaw", "lipAp", "lipProt", "lipWidth", "root", "larynx", "ttCurl")   # planner.rs:24
SWEEPS = (("tbY", 5.5, 8.5), ("tbX", 1.5, 4.2), ("jaw", 0.0, 2.0), ("ttY", 6.0, 9.5), ("ttX", 5.0, 7.0),
          ("root", -0.9, 0.9), ("lipAp", -0.2, 2.0), ("velum", -0.25, 1.2))
PAIRS = (("o", "ɝ"), ("i", "ɝ"), ("ɑ", "ɝ"), ("ə", "ɝ"), ("ɑ", "i"), ("u", "i"), ("æ", "u"), ("ʊ", "e"))


class Areas:
    """The tract's 20 + 24 section areas for an articulation (geometry_continuity.rs:14-31)."""

    def __init__(self, c2, body_label: str):
        from chimera_vocal.control.geometry import GEOMETRY_VEC

        self.c2, self.body, self.impl = c2, c2.body_of(body_label), GEOMETRY_VEC
        self.calls = 0

    def at(self, a: np.ndarray) -> np.ndarray:
        from chimera_vocal.control import frame_layout as fl

        self.calls += 1
        tract = self.c2.tract_for(self.body)
        frame = fl.alloc_frame()
        self.impl.compute(a, tract, frame)
        ph, or_, mph, mor, lens = np.zeros(20), np.zeros(24), np.zeros(20), np.zeros(24), np.zeros(2)
        self.impl.resample(a, tract, frame, ph, or_, mph, mor, lens)
        return np.concatenate([ph, or_])


def max_diff(x, y) -> float:
    m = 0.0
    for a, b in zip(x, y):
        m = max(m, abs(float(a) - float(b)))
    return m


def jump_size(areas, u0: float, u1: float, tol: float) -> float:
    """geometry_continuity.rs:42-90: the largest change that survives refinement of [u0, u1] to 2^-20."""
    memo: dict = {}

    def at(u):
        key = u + 0.0
        if key not in memo:
            memo[key] = areas(u)
        return memo[key]

    def diff(a, b):
        return max_diff(at(a), at(b))

    worst = [0.0]

    def visit(lo, hi, depth, d):
        if depth == 20:
            worst[0] = max(worst[0], d)
            return
        mid = (lo + hi) / 2.0
        d1, d2 = diff(lo, mid), diff(mid, hi)
        if d1 > tol:
            visit(lo, mid, depth + 1, d1)
        if d2 > tol:
            visit(mid, hi, depth + 1, d2)

    d = diff(u0, u1)
    if d > tol:
        visit(u0, u1, 0, d)
    return worst[0]


def suspicious(steps, i: int, min_step: float) -> bool:
    """geometry_continuity.rs:93-102: far above the +-8 neighbours' median."""
    s = steps[i]
    lo, hi = max(0, i - 8), min(i + 9, len(steps))
    skip = min(8, i)
    nb = sorted(x for k, x in enumerate(steps[lo:hi]) if k != skip)
    med = nb[len(nb) // 2] if nb else 0.0
    return s > min_step and s > 6.0 * max(med, 0.01)


def _with(base, key, x):
    from chimera_vocal.core.keys import ARTIC_INDEX

    a = base.copy()
    a[ARTIC_INDEX[key]] = x
    return a


def _vowel_base(c2, body, vw):
    """vowel_artic(vw) applied to NEUTRAL with the tip at rest (geometry_continuity.rs:131-135)."""
    from chimera_vocal.control.articulators import rest_tip
    from chimera_vocal.core.keys import ARTIC_INDEX

    v = VOWELS[body][vw]
    base = np.array(c2.TEMPLATE_NEUTRAL)
    for k in VOWEL_KEYS:
        base[ARTIC_INDEX[k]] = v[k]
    tx, ty = rest_tip(v["jaw"], v["tbX"], v["tbY"])
    base[ARTIC_INDEX["ttX"]], base[ARTIC_INDEX["ttY"]] = tx, ty
    return base


def _posture(c2, body, vw):
    """The full calibrated posture, tip included (geometry_continuity.rs:174-175)."""
    from chimera_vocal.core.keys import ARTIC_INDEX

    base = np.array(c2.TEMPLATE_NEUTRAL)
    for k, x in VOWELS[body][vw].items():
        base[ARTIC_INDEX[k]] = x
    return base


def sweep_jumps(c2, bodies, vowels, n):
    jumps = []
    for body in bodies:
        areas = Areas(c2, body)
        for vw in vowels:
            base = _vowel_base(c2, body, vw)
            for key, lo, hi in SWEEPS:
                prev = areas.at(_with(base, key, lo))
                steps = []
                for i in range(1, n + 1):
                    cur = areas.at(_with(base, key, lo + ((hi - lo) * i) / n))
                    steps.append(max_diff(cur, prev))
                    prev = cur
                for i in range(len(steps)):
                    if not suspicious(steps, i, 0.25):
                        continue
                    j = jump_size(lambda u: areas.at(_with(base, key, lo + (hi - lo) * u)), i / n, (i + 1) / n, 0.05)
                    if j > 0.05:
                        jumps.append(f"{body} {vw} {key}={lo + ((hi - lo) * (i + 1)) / n:.3f}: {j:.2f} cm2")
    return jumps


def between_jumps(c2, bodies, n):
    jumps = []
    for body in bodies:
        areas = Areas(c2, body)
        for x, y in PAIRS:
            a, b = _posture(c2, body, x), _posture(c2, body, y)

            def at(u, a=a, b=b):
                return a + (b - a) * u

            steps, prev = [], areas.at(at(0.0))
            for i in range(1, n + 1):
                cur = areas.at(at(i / n))
                steps.append(max_diff(cur, prev))
                prev = cur
            for i in range(len(steps)):
                if not suspicious(steps, i, 0.2):
                    continue
                j = jump_size(lambda u: areas.at(at(u)), i / n, (i + 1) / n, 0.05)
                if j > 0.05:
                    jumps.append(f"{body} {x}->{y} at {(i + 1) / n:.3f}: {j:.2f} cm2")
    return jumps


def tuck_jumps(c2, bodies, jaws, lip_aps, n):
    jumps = []
    for body in bodies:
        areas = Areas(c2, body)
        for jaw in jaws:
            for lip_ap in lip_aps:
                base = np.array(c2.TEMPLATE_NEUTRAL)
                base = _with(_with(_with(base, "jaw", jaw), "lipAp", lip_ap), "lipProt", 0.0)
                for i in range(n):
                    j = jump_size(lambda u: areas.at(_with(base, "labiodental", u)), i / n, (i + 1) / n, 0.05)
                    if j > 0.05:
                        jumps.append(f"{body} jaw={jaw} lipAp={lip_ap} labiodental={(i + 1) / n:.4f}: {j:.2f} cm2")
    return jumps


def protruded_tuck_jumps(c2, bodies, prots, widths, n):
    jumps = []
    for body in bodies:
        areas = Areas(c2, body)
        for lip_prot in prots:
            for lip_w in widths:
                base = np.array(c2.TEMPLATE_NEUTRAL)
                for k, x in (("jaw", 0.3), ("lipAp", 0.05), ("lipProt", lip_prot), ("lipWidth", lip_w)):
                    base = _with(base, k, x)
                prev = areas.at(_with(base, "labiodental", 0.0))
                for i in range(1, n + 1):
                    cur = areas.at(_with(base, "labiodental", i / n))
                    step = max_diff(cur, prev)
                    prev = cur
                    if step <= 0.05:
                        continue
                    j = jump_size(lambda u: areas.at(_with(base, "labiodental", u)), (i - 1) / n, i / n, 0.05)
                    if j > 0.05:
                        jumps.append(f"{body} lipProt={lip_prot} lipWidth={lip_w} labiodental={i / n:.4f}: {j:.2f}")
    return jumps


def test_refinement_finds_a_small_jump_beside_a_steep_ramp():
    """geometry_continuity.rs:216-224 (the harness itself)."""
    j = jump_size(lambda u: [min(1.6 * u, 0.8) + (0.1 if u >= 0.75 else 0.0)], 0.0, 1.0, 0.05)
    assert abs(j - 0.1) < 0.0005, j
    assert jump_size(lambda u: [min(1.6 * u, 0.8)], 0.0, 1.0, 0.05) == 0.0


def test_geometry_continuity_sweeps(c2):
    """K-2.10 (quick): male, vowels i ɑ u ə, 8 sweeps, 100 steps."""
    assert sweep_jumps(c2, ["male"], ["i", "ɑ", "u", "ə"], 100) == []


@pytest.mark.slow
def test_geometry_continuity_sweeps_full(c2):
    """K-2.10 (template size): 3 bodies x 4 vowels x 8 sweeps x 400 steps."""
    assert sweep_jumps(c2, ["male", "female", "child"], ["i", "ɑ", "u", "ə"], 400) == []


def test_geometry_continuity_between_vowels(c2):
    """K-2.10 (quick): male, 8 vowel pairs, all articulators at once, 100 steps."""
    assert between_jumps(c2, ["male"], 100) == []


@pytest.mark.slow
def test_geometry_continuity_between_vowels_full(c2):
    assert between_jumps(c2, ["male", "female", "child"], 300) == []


def test_geometry_continuity_labiodental_tuck(c2):
    """K-2.10 (quick): /f v/ handover, every step refined; male, two jaws, two apertures, 100 steps."""
    assert tuck_jumps(c2, ["male"], [0.3, 1.2], [0.05, 0.8], 100) == []


@pytest.mark.slow
def test_geometry_continuity_labiodental_tuck_full(c2):
    assert tuck_jumps(c2, ["male", "female", "child"], [0.3, 0.5, 0.8, 1.2], [0.05, 0.3, 0.8], 400) == []


def test_grazing_tip_vertex(c2):
    """geometry_continuity.rs:243-256: a fan ray grazing the tip of a calibrated /ɝ/ exactly at a vertex."""
    areas = Areas(c2, "male")
    base = _with(_posture(c2, "male", "ɝ"), "ttX", 5.625)
    y = 8.588839057347066
    a, b, c = (areas.at(_with(base, "ttY", t)) for t in (y - 1e-9, y, y + 1e-9))
    assert max_diff(a, b) < 1e-3 and max_diff(b, c) < 1e-3


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="template's known open case (geometry_continuity.rs:262-264, #[ignore]): "
                                       "protruded lower lip tucking jumps 0.41 cm2 (male, lab ~0.8612)")
def test_geometry_continuity_known_jump_protruded_tuck(c2):
    assert protruded_tuck_jumps(c2, ["male"], [0.2], [0.5], 400) == []


def _slew_limits(old, new_area, k2, dt):
    """Allowed change of one section in one block (human_airway.rs:194-251)."""
    open_r, gain, seal_r = 45.0 * k2, 40.0, 5.0 * k2
    lo, hi, ease = 0.03 * k2, 0.4 * k2, 0.03 * k2
    if old < ease:
        close = seal_r * math.sqrt(max(0.0, old) / ease)
    else:
        u = 0.0 if old <= lo else 1.0 if old >= hi else (old - lo) / (hi - lo)
        close = seal_r + (open_r - seal_r) * u * u * (3.0 - 2.0 * u)
    return (open_r + gain * new_area) * dt, (close + gain * old) * dt


def _check_slew(track, k2, dt):
    """Every block-to-block change within the limits (1e-12 slack); returns how many changes sat at the limit."""
    at_limit, bad = 0, []
    for key in ("tube_area", "tube_min"):
        a = getattr(track, key)
        for b in range(1, a.shape[0]):
            for i in range(a.shape[1]):
                old, new = float(a[b - 1, i]), float(a[b, i])
                up, down = _slew_limits(old, new, k2, dt)
                d = new - old
                if d > up * (1 + 1e-12) + 1e-15 or d < -down * (1 + 1e-12) - 1e-15:
                    bad.append(f"{key} block {b} section {i}: {old:.6f} -> {new:.6f}")
                elif abs(d - up) <= 1e-12 * max(1.0, up) or abs(d + down) <= 1e-12 * max(1.0, down):
                    at_limit += 1
        if key == "tube_min":
            assert (track.tube_min <= track.tube_area).all() and (track.tube_min >= 0.0).all()
    assert not bad, bad[:10]
    return at_limit


def test_track_slew_limit_holds_through_the_known_jump(c2):
    """K-2.10 (control): the protruded lower lip tucks through the template's known geometric jump; the control
    track spreads it under the slew limit (red when the limit is off)."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel
    from chimera_vocal.core.body import BODY_MALE

    m = ControlModel(BODY_MALE, c2.tract_index("male"), geometry=GEOMETRY_VEC)
    m.set_articulation({"jaw": 0.3, "lipAp": 0.05, "lipProt": 0.2, "lipWidth": 0.5})
    m.advance(150)
    m.set_articulation({"labiodental": 1.0})
    track = m.advance(200)
    assert _check_slew(track, 1.0, m.dt) > 0


def test_track_slew_limit_holds_for_fast_vowel_changes(c2):
    """K-2.10 (control): /i/ -> /ɑ/ -> closed lips -> /ɑ/ as fast as the articulators move, male and child."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel

    for label in ("male", "child"):
        body = c2.body_of(label)
        m = ControlModel(body, c2.tract_index(label), geometry=GEOMETRY_VEC)
        parts = []
        for patch in ({k: VOWELS[label]["i"][k] for k in VOWEL_KEYS}, {k: VOWELS[label]["ɑ"][k] for k in VOWEL_KEYS},
                      {"lipAp": -0.4}, {"lipAp": 2.0}):
            m.set_articulation(patch)
            parts.append(m.advance(90))
        track = c2.concat_chunks(parts)
        assert _check_slew(track, body.scale * body.scale, m.dt) > 0, label
