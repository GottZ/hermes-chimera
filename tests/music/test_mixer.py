"""Mixer contract (INTERFACES §8.4): MixWriter (disk) writes exactly what ``mix`` (memory) computes, voices land at
their offsets, gain/gain_env/pan apply per sample, added voice files are consumed, ``estimate_memory`` is bounded.
(K-4.5 levels: tests/measure/test_mix_levels.py; K-5.10 memory: test_mix_memory.py.)"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chimera_vocal.core.wavio import read_wav, wav_bytes
from chimera_vocal.music.compile import VoiceMix
from chimera_vocal.music.mixer import MixInfo, MixWriter, estimate_memory, mix, pan_gains


def tracks_and_mixes(rng: np.random.Generator, stereo: bool, env: bool):
    lengths = [5000, 7000, 3000]
    offsets = [0, 640, 64 * 120]
    tracks = [(rng.standard_normal(n) * 0.3).astype(np.float32) for n in lengths]
    mixes = []
    for i, (n, o) in enumerate(zip(lengths, offsets)):
        ge = None
        if env and i == 1:
            ge = np.linspace(0.25, 1.6, n).astype(np.float32)
        mixes.append(VoiceMix(offset=o, gain_db=[-3.0, 0.0, 4.5][i], pan=[-0.7, 0.0, 1.0][i] if stereo else 0.0,
                              gain_env=ge))
    n_mix = max(o + n for o, n in zip(offsets, lengths)) + 100
    return tracks, mixes, n_mix


def write_tracks(tmp_path, tracks):
    paths = []
    for i, t in enumerate(tracks):
        p = tmp_path / f"v{i}.f32"
        p.write_bytes(t.astype("<f4").tobytes())
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("stereo", [False, True], ids=["mono", "stereo"])
@pytest.mark.parametrize("normalize", [True, False], ids=["norm", "no-norm"])
@pytest.mark.parametrize("env", [False, True], ids=["flat", "env"])
@pytest.mark.parametrize("chunk", [1 << 16, 777], ids=["chunk64k", "chunk777"])
def test_writer_equals_reference(tmp_path, stereo, normalize, env, chunk):
    rng = np.random.default_rng(7)
    tracks, mixes, n_mix = tracks_and_mixes(rng, stereo, env)
    ref, info = mix(tracks, mixes, n_mix, stereo=stereo, normalize=normalize)
    assert ref.dtype == np.float32 and ref.shape == ((n_mix, 2) if stereo else (n_mix,))
    tmp = tmp_path / "mix"
    w = MixWriter(str(tmp), n_mix, stereo=stereo, chunk=chunk)
    for p, t, m in zip(write_tracks(tmp_path, tracks), tracks, mixes):
        w.add_voice(p, len(t), m)
    wav = tmp_path / "out.wav"
    info2 = w.finish(str(wav), normalize=normalize)
    assert isinstance(info2, MixInfo) and info2 == info
    assert wav.read_bytes() == wav_bytes(ref, 48000), "MixWriter output differs from mix()"
    assert not any(p.is_file() for p in tmp.rglob("*")) if tmp.exists() else True


def test_offsets_and_gains_mono(tmp_path):
    a = np.full(100, 0.1, np.float32)
    b = np.full(50, 0.2, np.float32)
    mixes = [VoiceMix(0, 0.0, 0.0, None), VoiceMix(64, -6.0, 0.0, None)]
    out, info = mix([a, b], mixes, 200, stereo=False, normalize=False)
    g = 10 ** (-6 / 20)
    assert out[:64] == pytest.approx(np.full(64, 0.1), abs=1e-7)
    assert out[64:100] == pytest.approx(np.full(36, 0.1 + 0.2 * g), abs=1e-6)
    assert out[100:114] == pytest.approx(np.full(14, 0.2 * g), abs=1e-6)
    assert np.all(out[114:] == 0)
    assert info.gain_db == pytest.approx(0.0) and info.peak_before == pytest.approx(0.1 + 0.2 * g, abs=1e-6)


def test_gain_env_per_sample():
    a = np.full(8, 0.1, np.float32)
    env = np.array([0, 0.5, 1, 1, 2, 2, 1, 0], np.float32)
    out, _ = mix([a], [VoiceMix(0, 0.0, 0.0, env)], 8, stereo=False, normalize=False)
    assert out == pytest.approx(0.1 * env, abs=1e-7)


@pytest.mark.parametrize("pan", [-1.0, -0.5, 0.0, 0.25, 1.0])
def test_pan_gains_equal_power(pan):
    l, r = pan_gains(pan)
    th = (pan + 1) * math.pi / 4
    assert (l, r) == pytest.approx((math.cos(th), math.sin(th)), abs=1e-15)
    assert l * l + r * r == pytest.approx(1.0, abs=1e-15)


def test_added_voice_files_consumed(tmp_path):
    t = np.zeros(1000, np.float32)
    p = tmp_path / "v.f32"
    p.write_bytes(t.tobytes())
    w = MixWriter(str(tmp_path / "mix"), 1000, stereo=False)
    w.add_voice(str(p), 1000, VoiceMix(0, 0.0, 0.0, None))
    assert not p.exists(), "a voice file is deleted once it is in the sum (D§8.4)"
    info = w.finish(str(tmp_path / "o.wav"))
    assert info.gain_db == 0.0 and info.peak_before == 0.0, "silence: g = 1"
    sr, x = read_wav(tmp_path / "o.wav")
    assert sr == 48000 and np.all(x == 0)


def test_estimate_memory_bounded():
    host_s, disk_s = estimate_memory(48000 * 10, True, [48000 * 10] * 2, parallel=3)
    host_l, disk_l = estimate_memory(48000 * 300, True, [48000 * 300] * 8, parallel=3)
    assert host_l <= 64.0 and host_l == pytest.approx(host_s, rel=0.05), "host memory independent of length"
    # temp disk: 8 voice files (float32) + float32 stereo sum + 16-bit stereo WAV
    expect = (8 * 48000 * 300 * 4 + 48000 * 300 * 2 * 4 + 48000 * 300 * 2 * 2) / 1e6
    assert disk_l == pytest.approx(expect, rel=0.1)
    assert disk_l > disk_s
