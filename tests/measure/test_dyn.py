"""K-4.7 test_dyn (TESTPLAN L4, P1-23): after the mixer the RMS rises strictly pp < p < mp < mf < f < ff by at least
1.5 dB per step, and the transitions do not click. ``dyn`` = pressure ramp (8 source commands over 80 ms ending on
the beat) + mix gain envelope (pp -12 … ff +4 dB). Red proof: dyn only as pressure (the voice's hold limiter at 0.75
flattens f/ff)."""

from __future__ import annotations

import numpy as np
import pytest

from chimera_vocal.music.mixer import mix
from support.pitch import envelope_db, rms_db

LEVELS = ["pp", "p", "mp", "mf", "f", "ff"]
BPM = 60


def dyn_score(dyn: bool = True) -> dict:
    notes = []
    for lv in LEVELS:
        n = {"pitch": "A3", "beats": 2, "lyric": "ah"}
        if dyn:
            n["dyn"] = lv
        notes.append(n)
    return {"bpm": BPM, "voices": [{"body": "male", "notes": notes}]}


def test_dyn_levels(pipeline):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(dyn_score())
    assert any(c.op == "source" for c in jobs[0].commands), "dyn ramps the pressure with source commands"
    out, _ = mix(tracks, mixes, n_mix, stereo=False, normalize=True)
    t0 = report.lead_in
    levels = []
    for k in range(len(LEVELS)):
        a = t0 + 2 * k + 0.5
        levels.append(rms_db(out[int(a * 48000):int((a + 1.0) * 48000)]))
    steps = np.diff(levels)
    assert np.all(steps >= 1.5), f"RMS per level {np.round(levels, 2)} dB, steps {np.round(steps, 2)}"
    assert mixes[0].gain_env is not None and mixes[0].gain_env[0] == (0, -12.0), "the first level holds from 0"


def test_dyn_no_click(pipeline):
    """Around each level change the 10-ms envelope moves at most 1.5 dB more per frame than the same note change
    without ``dyn`` (the 80-ms ramp spreads <= 6.1 dB over 8 frames), and no sample step exceeds 1.5× the largest
    step inside the neighbouring notes."""
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(dyn_score())
    out, _ = mix(tracks, mixes, n_mix, stereo=False, normalize=True)
    r0, j0, m0, n0, t0_tracks = pipeline.tracks(dyn_score(dyn=False))
    plain, _ = mix(t0_tracks, m0, n0, stereo=False, normalize=True)
    env, env0 = envelope_db(out), envelope_db(plain)
    t0 = report.lead_in
    for k in range(1, len(LEVELS)):
        b = int(round((t0 + 2 * k) * 100))          # 10-ms frame of the beat
        jumps = np.abs(np.diff(env[b - 12:b + 8]))
        ref = np.abs(np.diff(env0[b - 12:b + 8]))
        assert jumps.max() <= ref.max() + 1.5, \
            f"transition {LEVELS[k - 1]}->{LEVELS[k]}: 10-ms jump {jumps.max():.2f} dB (without dyn {ref.max():.2f})"
    d, d0 = np.abs(np.diff(out.astype(np.float64))), np.abs(np.diff(plain.astype(np.float64)))

    def ratio(dd, x, b):
        after = x[b + 960:b + 24000].astype(np.float64)
        return dd[b - 4800:b + 960].max() / np.sqrt(np.mean(after * after))

    for k in range(1, len(LEVELS)):
        b = int((t0 + 2 * k) * 48000)
        r_dyn, r_plain = ratio(d, out, b), ratio(d0, plain, b)
        assert r_dyn <= 1.5 * r_plain, f"click at {LEVELS[k - 1]}->{LEVELS[k]}: step/RMS {r_dyn:.3f} vs {r_plain:.3f}"
    # breakpoints: per level change two points 80 ms apart (old level at the ramp start, new level on the beat)
    pts = list(mixes[0].gain_env)
    gains = {"pp": -12.0, "p": -8.0, "mp": -4.0, "mf": 0.0, "f": 2.0, "ff": 4.0}
    assert len(pts) == 1 + 2 * (len(LEVELS) - 1)
    assert [s for s, _ in pts] == sorted(s for s, _ in pts), "samples never go back"
    for k in range(1, len(LEVELS)):
        (sa, da), (sb, db) = pts[2 * k - 1], pts[2 * k]
        assert (da, db) == (gains[LEVELS[k - 1]], gains[LEVELS[k]])
        assert sb - sa == round(0.08 * 48000), "an 80-ms ramp, never a step"
        assert sb + mixes[0].offset == round((t0 + 2 * k) * 48000), "the ramp ends on the beat"
