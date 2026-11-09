"""K-4.9 test_lead_in_raise (score validation): "strength" on beat 0 with lead_in 0.15 needs
0.258 s before the beat (measured on the template's plan_song) => warning, effective lead_in 0.27 s
(0.258 + 0.005 rounded up to 10 ms), no gesture starts before 0 s. Red proof: a fixed lower bound 0.15 without the
check."""

from __future__ import annotations

import pytest


def strength(lead_in: float | None, lyric: str = "strength") -> dict:
    s = {"bpm": 90, "voices": [{"body": "male", "notes": [{"pitch": "D3", "beats": 2, "lyric": lyric},
                                                           {"pitch": "F3", "beats": 2, "lyric": "now."}]}]}
    if lead_in is not None:
        s["lead_in"] = lead_in
    return s


def earliest(jobs) -> float:
    return min(g.t0 - g.ramp_in for j in jobs for g in j.gestures)


def test_lead_in_raised(pipeline):
    from chimera_vocal.music.schema import validate_score
    report = validate_score(strength(0.15), g2p=pipeline.g2p)
    assert report.ok
    assert report.lead_in == pytest.approx(0.27, abs=1e-12)
    warns = [i for i in report.issues if i.severity == "warning" and i.path == "lead_in"]
    assert len(warns) == 1 and "0.27" in warns[0].message
    assert report.normalized["lead_in"] == pytest.approx(0.27, abs=1e-12)
    _, jobs, _, _ = pipeline.compile(strength(0.15))
    assert earliest(jobs) >= 0.005 - 1e-12, "no gesture before 0 s (and the 5-ms margin holds)"


def test_ah_raise_and_no_raise(pipeline):
    from chimera_vocal.music.schema import validate_score
    r0 = validate_score(strength(0.0, "ah"), g2p=pipeline.g2p)
    assert r0.lead_in == pytest.approx(0.09, abs=1e-12), "ah: 0.080 + 0.005 -> 0.09"
    r1 = validate_score(strength(None), g2p=pipeline.g2p)
    assert r1.lead_in == pytest.approx(0.3) and not [i for i in r1.issues if i.path == "lead_in"]
