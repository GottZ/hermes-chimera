"""Self-tests of the C4 measurement tools (tests/support/pitch.py) on synthetic signals with known truth.
Red proof of the f0 tool: naked YIN reports the octave above on an H2-dominant tone (E1 pitch-range report)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from support.fakes import FakeVoice
from support.pitch import (cents, compare_split, envelope_db, harmonic_score, hz, lpc_formants, naked_yin_f0, onsets,
                           peak_dbfs, rms_db, sung_f0, third_octave_db, yin_track)
from chimera_vocal.core.types import Gesture

FS = 48000


def held(f0: float, seconds: float = 1.0) -> np.ndarray:
    v = FakeVoice()
    v.set_source({"f0": f0})
    v.settle_pitch()
    v.schedule([Gesture(0.0, seconds, 0.01, 0.01, 1.0, {"breath": 1.0}, label="breath")])
    return v.render(int(seconds * FS)).astype(np.float64)


@pytest.mark.parametrize("midi", [36, 48, 57, 64, 72, 84])
def test_naked_yin_fails_on_h2_dominant(midi):
    x = held(hz(midi))[12000:36000]
    assert abs(cents(naked_yin_f0(x, FS, fmax=3000.0), hz(midi)) - 1200) < 10, "the tone is a real octave trap"


@pytest.mark.parametrize("midi", [28, 36, 48, 57, 64, 72, 84, 96])
def test_sung_f0_octave_checked(midi):
    x = held(hz(midi))[12000:36000]
    assert abs(cents(sung_f0(x, FS, hz(midi)), hz(midi))) <= (1.0 if midi <= 84 else 3.0)


def test_sung_f0_does_not_snap_to_target():
    x = held(hz(60) * 432 / 440)[12000:36000]
    assert cents(sung_f0(x, FS, hz(60)), hz(60)) == pytest.approx(1200 * math.log2(432 / 440), abs=1.0)
    sine = np.sin(2 * np.pi * 440 * np.arange(FS) / FS)
    assert abs(cents(sung_f0(sine, FS, 220.0), 440.0)) < 1.0, "no subharmonic invented for a pure tone"


def test_harmonic_score_prefers_true_f0():
    x = held(220.0)[12000:60000]
    assert harmonic_score(x, FS, 220.0) > harmonic_score(x, FS, 440.0)


def test_yin_track_gate():
    x = np.concatenate([np.zeros(FS // 2), held(200.0, 0.5)])
    t, f = yin_track(x, FS, fmin=60, fmax=500)
    assert np.all(np.isnan(f[t < 0.45])) and np.isfinite(f[t > 0.6]).mean() > 0.9


def test_onsets_exact():
    x = np.zeros(FS)
    for i in (4800, 20000, 41000):
        x[i:i + 48] = np.exp(-np.arange(48) / 8.0) * 0.5
    assert onsets(x) == [4800, 20000, 41000]


def resonator_vowel(formants, bws, f0=110.0, seconds=0.5):
    """Impulse train through cascaded two-pole resonators (known formants)."""
    n = int(seconds * FS)
    x = np.zeros(n)
    x[::int(FS / f0)] = 1.0
    for f, b in zip(formants, bws):
        r = math.exp(-math.pi * b / FS)
        c1, c2 = 2 * r * math.cos(2 * math.pi * f / FS), -r * r
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[i] + c1 * (y[i - 1] if i else 0.0) + c2 * (y[i - 2] if i > 1 else 0.0)
        x = y / np.max(np.abs(y))
    return x


@pytest.mark.parametrize("formants", [(700, 1220, 2600), (300, 2300, 3000), (350, 900, 2300)])
def test_lpc_formants(formants):
    x = resonator_vowel(formants, (80, 90, 120))
    est = lpc_formants(x[12000:12000 + 1536 * 2], FS, order=12)
    for f in formants[:2]:
        # LPC bias near a harmonic: F1 300 Hz between 220/330 Hz reads ~4 % high; K-4.2 compares port and native
        # with the same tool, so the bias cancels there
        assert min(abs(e - f) / f for e in est) <= (0.05 if f < 400 else 0.03), (formants, est)


def test_levels_and_spectra():
    x = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(FS) / FS)
    assert peak_dbfs(x) == pytest.approx(20 * math.log10(0.5), abs=1e-6)
    assert rms_db(x) == pytest.approx(20 * math.log10(0.5 / math.sqrt(2)), abs=1e-3)
    env = envelope_db(x)
    assert len(env) == 100 and np.ptp(env) < 0.01
    bands = third_octave_db(x)
    assert int(np.argmax(bands)) == 10, "1 kHz is band 10 of 100·2^(k/3)"


def test_compare_split_identity_and_shift():
    x = held(205.0, 2.0)
    m = compare_split(x, x, onsets_s=[0.5])
    assert m.env_p95_db == 0 and m.f0_p90_ct == 0 and m.f0_onset_p90_ct == 0 and m.third_rms_db == 0
    assert m.octave_fixed == 0 and m.bands_excluded == 0
    y = held(205.0 * 2 ** (30 / 1200), 2.0)
    # p90 over frames carries the frame-level YIN spread (window 1024 as E1-A): 30 ct shift reads 30–36 ct
    assert compare_split(x, y, fmin=100, fmax=500).f0_p90_ct == pytest.approx(32, abs=5)


def test_compare_split_keeps_real_octave():
    """No octave folding: a render a true octave higher reads ~1200 ct (the harmonic sum confirms both octaves)."""
    x = held(150.0, 1.5)
    y = held(300.0, 1.5)
    m = compare_split(x, y, fmin=60, fmax=700)
    assert m.f0_p90_ct == pytest.approx(1200, abs=15), m


def test_compare_split_resolves_yin_octave_slips():
    """FakeVoice is H2-dominant: YIN slips an octave in some frames of one track but not the other; the harmonic sum
    resolves those frames instead of folding, and the count is reported."""
    x = held(205.0, 2.0)
    m = compare_split(x, held(205.0, 2.0), fmin=60, fmax=500)
    assert m.f0_p90_ct == 0
    glide = FakeVoice()
    glide.set_source({"f0": 205.0})
    glide.schedule([Gesture(0.0, 2.0, 0.01, 0.01, 1.0, {"breath": 1.0}, label="breath")])
    y = glide.render(96000).astype(np.float64)        # unsettled: glides up from 110 Hz
    m2 = compare_split(x, y, fmin=60, fmax=500, onsets_s=[0.0])
    assert m2.f0_onset_p90_ct > 300, m2              # the real 110 -> 205 Hz glide stays visible


def test_compare_split_band_selection():
    x = held(205.0, 2.0)
    assert compare_split(x, x, bands="active40").bands_excluded > 0
    with pytest.raises(ValueError):
        compare_split(x, x, bands="some")
