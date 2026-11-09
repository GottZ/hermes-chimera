"""K-4.5 test_mix_levels (mixer): the normalised mix peaks at -1 dBFS (±0.01 dB) with no
sample above; with normalisation off the mix is only ever lowered; equal-power pan keeps the power constant.
Red proof: linear pan law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chimera_vocal.core.wavio import read_wav
from chimera_vocal.music.compile import VoiceMix
from chimera_vocal.music.mixer import mix
from support.pitch import peak_dbfs

TARGET = 10 ** (-1 / 20)


def choir(n_voices: int = 3) -> dict:
    return {"bpm": 100, "voices": [{"body": b, "notes": [{"pitch": p + 7 * i, "beats": 2, "lyric": "ah"}
                                                         for p in (50, 53, 55)]}
                                   for i, b in enumerate(("male", "female", "child")[:n_voices])]}


def test_normalised_peak(pipeline):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(choir())
    out, info = mix(tracks, mixes, n_mix, stereo=False, normalize=True)
    assert peak_dbfs(out) == pytest.approx(-1.0, abs=0.01)
    assert float(np.max(np.abs(out.astype(np.float64)))) <= TARGET, "no sample above -1 dBFS"
    assert info.clipped_samples == 0
    assert info.gain_db == pytest.approx(20 * math.log10(TARGET / info.peak_before), abs=1e-6)


def test_no_normalise_only_lowers(pipeline):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(choir())
    quiet = [VoiceMix(m.offset, -30.0, 0.0, m.gain_env) for m in mixes]
    out, info = mix(tracks, quiet, n_mix, stereo=False, normalize=False)
    expected = np.zeros(n_mix)
    for t, m in zip(tracks, quiet):
        expected[m.offset:m.offset + len(t)] += t.astype(np.float64) * 10 ** (-30 / 20)
    assert info.gain_db == 0.0, "quiet mix: no boost"
    assert np.allclose(out.astype(np.float64), expected, rtol=0, atol=1e-6)
    loud = [VoiceMix(m.offset, 12.0, 0.0, m.gain_env) for m in mixes]
    out2, info2 = mix(tracks, loud, n_mix, stereo=False, normalize=False)
    assert info2.peak_before > TARGET and info2.gain_db < 0
    assert float(np.max(np.abs(out2.astype(np.float64)))) <= TARGET


@pytest.mark.parametrize("pan", [-1.0, -0.5, 0.0, 0.3, 1.0])
def test_equal_power_pan(pipeline, pan):
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(choir(1))
    centre, _ = mix(tracks, [VoiceMix(mixes[0].offset, -12.0, 0.0, None)], n_mix, stereo=True, normalize=False)
    out, _ = mix(tracks, [VoiceMix(mixes[0].offset, -12.0, pan, None)], n_mix, stereo=True, normalize=False)
    p = lambda a: float(np.sum(a.astype(np.float64) ** 2))
    assert p(out[:, 0]) + p(out[:, 1]) == pytest.approx(p(centre[:, 0]) + p(centre[:, 1]), rel=1e-5)
    th = (pan + 1) * math.pi / 4
    mono = p(centre[:, 0]) / math.cos(math.pi / 4) ** 2
    assert p(out[:, 0]) == pytest.approx(mono * math.cos(th) ** 2, rel=1e-5, abs=1e-12)
    assert p(out[:, 1]) == pytest.approx(mono * math.sin(th) ** 2, rel=1e-5, abs=1e-12)


def test_engine_wav_peak(stub_planner, fake_workers, make_engine, tmp_path):
    res = make_engine(workers=3).render_score(choir(), str(tmp_path), "levels", want_mp3=False)
    sr, x = read_wav(res.path_wav)
    assert peak_dbfs(x) == pytest.approx(-1.0, abs=0.01), "16-bit WAV peak at -1 dBFS"
