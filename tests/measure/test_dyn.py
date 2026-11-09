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


def dyn_score() -> dict:
    notes = []
    for lv in LEVELS:
        notes.append({"pitch": "A3", "beats": 2, "lyric": "ah", "dyn": lv})
    return {"bpm": BPM, "voices": [{"body": "male", "notes": notes}]}


def test_dyn_levels(pipeline):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(dyn_score())
    assert any(c.op == "source" for c in jobs[0].commands), "dyn ramps the pressure with source commands"
    assert mixes[0].gain_env is not None and len(mixes[0].gain_env) == jobs[0].n_out
    out, _ = mix(tracks, mixes, n_mix, stereo=False, normalize=True)
    t0 = report.lead_in
    levels = []
    for k in range(len(LEVELS)):
        a = t0 + 2 * k + 0.5
        levels.append(rms_db(out[int(a * 48000):int((a + 1.0) * 48000)]))
    steps = np.diff(levels)
    assert np.all(steps >= 1.5), f"RMS per level {np.round(levels, 2)} dB, steps {np.round(steps, 2)}"


def test_dyn_no_click(pipeline):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(dyn_score())
    out, _ = mix(tracks, mixes, n_mix, stereo=False, normalize=True)
    env = envelope_db(out)
    t0 = report.lead_in
    for k in range(1, len(LEVELS)):
        b = int(round((t0 + 2 * k) * 100))          # 10-ms frame of the beat
        around = env[b - 12:b + 8]
        jumps = np.abs(np.diff(around))
        assert jumps.max() <= 3.0, f"transition {LEVELS[k - 1]}->{LEVELS[k]}: 10-ms jump {jumps.max():.2f} dB"
    d = np.abs(np.diff(out.astype(np.float64)))
    seg = [d[int((t0 + 2 * k + 0.3) * 48000):int((t0 + 2 * k + 1.7) * 48000)].max() for k in range(len(LEVELS))]
    for k in range(1, len(LEVELS)):
        b = int((t0 + 2 * k) * 48000)
        assert d[b - 4800:b + 960].max() <= 1.5 * max(seg[k - 1], seg[k]), "no click at the transition"
