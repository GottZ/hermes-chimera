"""K-2.11: oral vowels stay oral -- in the control track, the velum port target (velum_area) is < 0.01 cm^2 at every
block inside a vowel segment that lies at least 50 ms from the nearest nasal segment (plan segments of the V3
sources). Measured on the template (full V3 exports): 0.0 there for hello-male, stop-male and song-female; it is
the velum's bounded blend (articulators.rs:122-127) that keeps nasality out of the neighbouring vowels. A velum
gesture of strength 40 (instead of 4), or a weighted-mean blend for the velum, puts 0.03-0.04 cm^2 into those
vowels (measured in S1, reports/S1-C2.md) -- the guard test keeps that detection alive.
"""

import pytest

from chimera_vocal.core.types import Gesture

LIMIT = 0.01        # cm^2
MARGIN = 0.05       # s from the nearest nasal segment


def _segments(trk):
    segs = trk.scenario["info"]["plan_segments"]
    vowels = [(s["t0"], s["t1"], s["ipa"]) for s in segs if s["kind"] == "vowel"]
    nasals = [(s["t0"], s["t1"]) for s in segs if s["kind"] == "nasal"]
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
