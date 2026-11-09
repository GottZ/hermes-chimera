"""K-2.11: oral vowels stay oral -- in the control track, the velum port target (velum_area) is < 0.01 cm^2 at every
block inside a vowel segment that lies at least 50 ms from the nearest nasal segment (plan segments of the V3
sources). Measured on the template (full V3 exports): 0.0 there for hello-male, stop-male and song-female; it is
the velum's bounded blend (articulators.rs:122-127) that keeps nasality out of the neighbouring vowels. A velum
gesture of strength 40 (instead of 4), or a weighted-mean blend for the velum, puts 0.03-0.04 cm^2 into those
vowels (measured in S1, reports/S1-C2.md) -- the guard test keeps that detection alive.
"""

import json

import pytest

from chimera_vocal.core.types import Gesture

LIMIT = 0.01        # cm^2
MARGIN = 0.05       # s from the nearest nasal segment

# Vowel and nasal segments ([ipa, t0, t1]) of the plans behind the V3 scenarios -- the repository heads carry no
# plan segments. Source: scenario.info.plan_segments of $CHIMERA_LAB_REFS/tracks/track-<scenario>.full.json (S0-V
# export of template 092b16ce12b9, MIT, Copyright (c) 2026 Jameson Hodge).
SEGMENTS = json.loads('{"hello-male":{"vowel":[["ə",0.19,0.245704],["o",0.319704,0.400664],["u",0.400664,0.46690400000000004],["ɝ",0.5469040000000001,0.766904]],"nasal":[]},"stop-male":{"vowel":[["ɑ",0.26637500000000003,0.46281500000000003],["ə",0.5425650000000001,0.606963],["ə",0.7158380000000001,0.7836380000000001],["ɑ",0.858638,1.115838],["æ",1.520588,1.767588],["ə",2.078838,2.144937],["e",2.329012,2.414548],["i",2.414548,2.4845319999999997],["ə",2.6004319999999996,2.6682319999999997],["i",2.743232,2.8782319999999997],["u",2.953232,3.0617319999999997],["ə",3.1117319999999995,3.1795319999999996],["æ",3.2595319999999997,3.5065319999999995]],"nasal":[["n",2.144937,2.2008870000000003]]},"song-female":{"vowel":[["ɪ",0.3,0.7719999999999998],["ə",0.8999999999999999,1.3399999999999999],["ɪ",1.5,1.972],["ə",2.1,2.596],["ɪ",2.7,3.24],["ə",3.3000000000000003,3.692],["ɑ",3.9000000000000004,4.928000000000001],["a",5.1000000000000005,5.58],["ʊ",5.58,5.7],["a",5.7,6.1415999999999995],["ɪ",6.1415999999999995,6.252],["ʌ",6.3,6.795999999999999],["ɝ",6.8999999999999995,7.451999999999999],["ʌ",7.499999999999999,7.9959999999999996],["u",8.1,8.7],["ɑ",8.7,9.797999999999998]],"nasal":[["ŋ",0.7719999999999998,0.8359999999999999],["ŋ",1.972,2.036],["n",6.795999999999999,6.851999999999999]]}}')


def _segments(trk):
    segs = SEGMENTS[trk.name]
    vowels = [(t0, t1, ipa) for ipa, t0, t1 in segs["vowel"]]
    nasals = [(t0, t1) for _, t0, t1 in segs["nasal"]]
    return vowels, nasals


def _leaks(trk, velum_area, dt):
    """Blocks (time after the step) inside an oral stretch of a vowel with velum_area >= LIMIT."""
    vowels, nasals = _segments(trk)
    out = []
    for b, v in enumerate(velum_area):
        t = (b + 1) * dt
        inside = [ipa for a, e, ipa in vowels if a <= t <= e]
        if not inside:
            continue
        dist = min([max(a - t, t - e, 0.0) for a, e in nasals] or [9.0])
        if dist >= MARGIN and v >= LIMIT:
            out.append((b, round(t, 4), inside[0], float(v)))
    return out


def _run(c2, name, n_blocks=None, velum_strength=None):
    from chimera_vocal.control.geometry import GEOMETRY_VEC

    trk = c2.decode_track(name)
    m = c2.new_model(trk, GEOMETRY_VEC)
    for o in trk.scenario["ops"]:
        if o["op"] == "schedule" and velum_strength is not None:
            gs = [Gesture.from_json(g) for g in o["gestures"]]
            m.schedule([Gesture(g.t0, g.t1, g.ramp_in, g.ramp_out,
                                velum_strength if "velum" in g.targets else g.strength, g.targets, g.speed,
                                g.release_hold, g.label, g.tag) for g in gs])
        else:
            c2.apply_ops(m, [o])
    track = m.advance(n_blocks or trk.n_blocks)
    return trk, track, m.dt


@pytest.mark.parametrize("name", ["hello-male", "stop-male",
                                  pytest.param("song-female", marks=pytest.mark.slow)])
def test_nasal_leak(c2, name):
    trk, track, dt = _run(c2, name)
    vowels, nasals = _segments(trk)
    assert vowels and (name == "hello-male" or nasals)
    leaks = _leaks(trk, track.velum_area, dt)
    assert not leaks, f"{len(leaks)} oral-vowel blocks with velum >= {LIMIT}: {leaks[:10]}"


def test_nasal_leak_criterion_detects_spread(c2):
    """Guard (the K-2.11 red proof kept in the suite): with the velum gesture at strength 40 the same criterion
    finds the leak in stop-male ("... and take ...", the vowel before /n/)."""
    trk, track, dt = _run(c2, "stop-male", n_blocks=1800, velum_strength=40.0)
    leaks = _leaks(trk, track.velum_area, dt)
    assert leaks, "strength 40 must spread nasality into the vowel before the nasal"
