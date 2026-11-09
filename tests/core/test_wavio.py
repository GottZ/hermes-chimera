"""K-0.7: WAV writing exactly like chimera-cli/src/wav.rs:24-33."""

import math
import struct
import wave

import numpy as np
import pytest

from chimera_vocal.core import wavio


def ref_pcm16(v: float) -> int:
    """Independent statement of the rule: float32 first, then non-finite -> 0, clamp, js_round(v * 32767)."""
    v = float(np.float32(v))
    if not math.isfinite(v):
        v = 0.0
    v = min(max(v, -1.0), 1.0)
    y = v * 32767.0
    r = math.floor(y)
    return int(r + 1 if y - r >= 0.5 else r)


def _f32_sensitive_values(n: int = 64) -> list[float]:
    """float64 values whose int16 changes when the float32 step is skipped."""
    rng = np.random.default_rng(7)
    out = []
    while len(out) < n:
        k = int(rng.integers(-32767, 32767))
        v = (k + 0.5) / 32767.0 + float(rng.normal()) * 1e-12
        if ref_pcm16(v) != (lambda y: int(math.floor(y) + (1 if y - math.floor(y) >= 0.5 else 0)))(v * 32767.0):
            out.append(v)
    return out


def _data(b: bytes) -> bytes:
    assert b[:4] == b"RIFF" and b[8:16] == b"WAVEfmt " and b[36:40] == b"data"
    return b[44:]


def test_pcm16_rule():
    vals = [0.0, -0.0, 0.25, -0.25, 1.0, -1.0, 1.5, -2.0, 0.1, -0.1, 0.999999999,
            float("nan"), float("inf"), float("-inf"), 1e-45, 3e38]
    vals += _f32_sensitive_values()
    got = np.frombuffer(_data(wavio.wav_bytes(np.array(vals), 48000)), dtype="<i2").tolist()
    assert got == [ref_pcm16(v) for v in vals]
    assert got[2] == 8192 and got[3] == -8192             # 8191.75 / -8191.75 round to nearest
    assert got[11:14] == [0, 0, 0] and got[6:8] == [32767, -32767]


def test_rot_beleg_detects_missing_f32_step():
    """The f32-sensitive values really separate the rule from int(v*32767)/round-without-f32."""
    vals = _f32_sensitive_values(8)
    naive = [int(v * 32767) for v in vals]
    assert naive != [ref_pcm16(v) for v in vals]


def test_header_and_stdlib_wave(tmp_path):
    x = np.linspace(-1.0, 1.0, 1001)
    p = tmp_path / "m.wav"
    wavio.write_wav(p, x, 48000)
    b = p.read_bytes()
    assert struct.unpack("<4sI4s4sIHHIIHH4sI", b[:44]) == (
        b"RIFF", 36 + 2002, b"WAVE", b"fmt ", 16, 1, 1, 48000, 96000, 2, 16, b"data", 2002)
    with wave.open(str(p), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 48000, 1001)
        assert w.readframes(1001) == b[44:]


def test_stereo_interleave(tmp_path):
    x = np.stack([np.full(5, 0.25), np.full(5, -0.5)], axis=1)
    b = wavio.wav_bytes(x, 48000)
    assert struct.unpack("<H", b[22:24])[0] == 2 and struct.unpack("<I", b[28:32])[0] == 48000 * 4
    assert np.frombuffer(_data(b), dtype="<i2").tolist() == [ref_pcm16(0.25), ref_pcm16(-0.5)] * 5
    with pytest.raises(ValueError):
        wavio.wav_bytes(np.zeros((4, 3)), 48000)


def test_chunked_equals_whole(tmp_path):
    rng = np.random.default_rng(1)
    for shape in ((10007,), (10007, 2)):
        x = rng.uniform(-1.2, 1.2, size=shape).astype(np.float32)
        ch = 1 if len(shape) == 1 else 2
        whole, part = tmp_path / f"w{ch}.wav", tmp_path / f"c{ch}.wav"
        wavio.write_wav(whole, x, 48000)
        cuts = [0, 1, 78, 4096, 4097, 10007]
        wavio.write_wav_chunked(part, (x[a:b] for a, b in zip(cuts, cuts[1:])), 48000, ch, len(x))
        assert part.read_bytes() == whole.read_bytes()
    bad = tmp_path / "bad.wav"
    with pytest.raises(ValueError):
        wavio.write_wav_chunked(bad, [np.zeros(10)], 48000, 1, 11)
    assert not bad.exists()


def test_float32_and_read_back(tmp_path):
    x = np.array([0.1, -0.7, 1.5, float("nan")], dtype=np.float64)
    p = tmp_path / "f.wav"
    wavio.write_wav(p, x, 96000, bits=32)
    b = p.read_bytes()
    assert struct.unpack("<H", b[20:22])[0] == 3 and struct.unpack("<H", b[34:36])[0] == 32
    assert b[44:] == x.astype("<f4").tobytes()
    rate, y = wavio.read_wav(p)
    assert rate == 96000 and np.array_equal(y[:3], x[:3].astype(np.float32).astype(np.float64))
    q = tmp_path / "s.wav"
    wavio.write_wav(q, np.stack([x[:3], -x[:3]], axis=1), 48000)
    rate, y = wavio.read_wav(q)
    assert rate == 48000 and y.shape == (3, 2)
    assert np.allclose(y[:, 0], np.clip(x[:3], -1, 1), atol=0.5 / 32767)


def test_read_f32(tmp_path):
    x = np.array([0.5, -0.25, 3.0], dtype="<f4")
    p = tmp_path / "raw.f32"
    p.write_bytes(x.tobytes())
    y = wavio.read_f32(p)
    assert y.dtype == np.float32 and np.array_equal(y, x)
