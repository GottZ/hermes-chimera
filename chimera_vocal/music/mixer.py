"""Disk-backed mixer: peak host memory O(chunk · channels), independent of score length.

Arithmetic (shared by MixWriter and the in-memory ``mix``): per voice and sample the contribution is
float32(track · 10^(gain_db/20) [· gain_env gain] · pan_gain) in float64, added in voice order to a float32 sum. Mono
ignores ``pan``; stereo uses equal power, θ = (pan + 1)·π/4 -> (cos θ, sin θ). Normalisation is pure scaling:
g = T/peak (``normalize``) or min(1, T/peak), T = 10^(target/20), then clamped to the largest float32 <= T.
The sum lives in a raw float32 file accessed by seek/read/write per chunk (no memory map: mapped pages would count
towards the host RSS that this design keeps bounded)."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .compile import VoiceMix


@dataclass(frozen=True)
class MixInfo:
    peak_before: float
    gain_db: float
    channels: int
    clipped_samples: int             # always 0 (pure scaling, no limiter)


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan: θ = (pan + 1)·π/4 -> (cos θ, sin θ)."""
    th = (float(pan) + 1.0) * math.pi / 4.0
    return math.cos(th), math.sin(th)


def env_gains(points: "Sequence[tuple[int, float]] | None", start: int, k: int) -> "np.ndarray | None":
    """float32 linear gains of samples start … start+k-1 from dB breakpoints (piecewise linear in dB, constant
    before the first and after the last point). Pointwise, so every chunking gives the same values."""
    import numpy as np
    if points is None:
        return None
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    db = np.interp(np.arange(start, start + k, dtype=np.float64), xs, ys)
    return (10.0 ** (db / 20.0)).astype(np.float32)


def _contrib(track: "np.ndarray", g: float, env: "np.ndarray | None", channels: int, pan: float) -> "np.ndarray":
    import numpy as np
    x = np.asarray(track, dtype=np.float32).astype(np.float64) * g
    if env is not None:
        x = x * np.asarray(env, dtype=np.float32).astype(np.float64)
    if channels == 1:
        return x.astype(np.float32).reshape(-1, 1)
    gl, gr = pan_gains(pan)
    return np.stack([(x * gl).astype(np.float32), (x * gr).astype(np.float32)], axis=1)


def _limit(target_peak_dbfs: float) -> tuple[float, "np.float32"]:
    import numpy as np
    t = 10.0 ** (target_peak_dbfs / 20.0)
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(0))
    return t, f


def _gain(peak: float, normalize: bool, target_peak_dbfs: float) -> float:
    t, _ = _limit(target_peak_dbfs)
    if peak <= 0.0:
        return 1.0
    return t / peak if normalize else min(1.0, t / peak)


def _scale(block: "np.ndarray", g: float, lim) -> "np.ndarray":
    import numpy as np
    y = (block.astype(np.float64) * g).astype(np.float32)
    np.clip(y, -lim, lim, out=y)
    return y


def _info(peak: float, g: float, channels: int) -> MixInfo:
    return MixInfo(peak, 0.0 if g == 1.0 else 20.0 * math.log10(g), channels, 0)


class MixWriter:
    """Each finished voice (float32 file) is added chunk-wise into a float32 sum on disk; after the last voice:
    pass 1 peak, pass 2 scale + write WAV chunk-wise. Temp files removed in all outcomes; an added voice file is
    deleted once it is in the sum."""

    def __init__(self, tmp_dir: str, n_mix: int, *, stereo: bool, chunk: int = 1 << 16) -> None:
        self.tmp_dir = str(tmp_dir)
        self._made_dir = not os.path.isdir(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.n_mix = int(n_mix)
        self.channels = 2 if stereo else 1
        self.chunk = max(1, int(chunk))
        self.sum_path = os.path.join(self.tmp_dir, f"mix-sum-{uuid.uuid4().hex}.f32")
        with open(self.sum_path, "wb") as f:
            f.truncate(self.n_mix * self.channels * 4)
        self._closed = False

    def add_voice(self, pcm_path: str, n: int, mix: "VoiceMix") -> None:
        import numpy as np
        if self._closed:
            raise RuntimeError("MixWriter already finished")
        g = 10.0 ** (float(mix.gain_db) / 20.0)
        n = int(n)
        off = int(mix.offset)
        n_use = max(0, min(n, self.n_mix - off))
        ch = self.channels
        try:
            with open(pcm_path, "rb") as src, open(self.sum_path, "r+b") as dst:
                done = 0
                while done < n_use:
                    k = min(self.chunk, n_use - done)
                    src.seek(4 * done)
                    tr = np.fromfile(src, dtype="<f4", count=k)
                    if tr.size != k:
                        raise ValueError(f"{pcm_path}: {tr.size} samples, expected {k} at {done}")
                    env = env_gains(mix.gain_env, done, k)
                    c = _contrib(tr, g, env, ch, mix.pan)
                    pos = (off + done) * ch * 4
                    dst.seek(pos)
                    s = np.fromfile(dst, dtype="<f4", count=k * ch).reshape(k, ch)
                    s = s + c
                    dst.seek(pos)
                    dst.write(s.astype("<f4").tobytes())
                    done += k
        finally:
            pass
        os.remove(pcm_path)

    def _chunks(self) -> Iterator["np.ndarray"]:
        import numpy as np
        with open(self.sum_path, "rb") as f:
            done = 0
            while done < self.n_mix:
                k = min(self.chunk, self.n_mix - done)
                yield np.fromfile(f, dtype="<f4", count=k * self.channels).reshape(k, self.channels)
                done += k

    def finish(self, wav_path: str, *, normalize: bool = True, target_peak_dbfs: float = -1.0) -> MixInfo:
        import numpy as np
        from ..core.wavio import write_wav_chunked
        try:
            peak = 0.0
            for b in self._chunks():
                if b.size:
                    peak = max(peak, float(np.max(np.abs(b.astype(np.float64)))))
            g = _gain(peak, normalize, target_peak_dbfs)
            _, lim = _limit(target_peak_dbfs)
            ch = self.channels
            gen = (_scale(b, g, lim) if ch == 2 else _scale(b, g, lim).reshape(-1) for b in self._chunks())
            tmp = f"{wav_path}.part-{uuid.uuid4().hex[:8]}"
            try:
                write_wav_chunked(tmp, gen, 48000, ch, self.n_mix)
                os.replace(tmp, wav_path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            return _info(peak, g, ch)
        finally:
            self.abort()

    def abort(self) -> None:
        self._closed = True
        try:
            os.remove(self.sum_path)
        except FileNotFoundError:
            pass
        if self._made_dir:
            try:
                os.rmdir(self.tmp_dir)
            except OSError:
                pass


def mix(tracks: "Sequence[np.ndarray]", mixes: "Sequence[VoiceMix]", n_mix: int, *, stereo: bool,
        normalize: bool = True, target_peak_dbfs: float = -1.0) -> "tuple[np.ndarray, MixInfo]":
    """In-memory reference implementation (same arithmetic as MixWriter)."""
    import numpy as np
    ch = 2 if stereo else 1
    s = np.zeros((int(n_mix), ch), np.float32)
    for tr, m in zip(tracks, mixes):
        off = int(m.offset)
        k = max(0, min(len(tr), n_mix - off))
        if k == 0:
            continue
        env = env_gains(m.gain_env, 0, k)
        s[off:off + k] = s[off:off + k] + _contrib(np.asarray(tr)[:k], 10.0 ** (float(m.gain_db) / 20.0), env, ch,
                                                   m.pan)
    peak = float(np.max(np.abs(s.astype(np.float64)))) if s.size else 0.0
    g = _gain(peak, normalize, target_peak_dbfs)
    _, lim = _limit(target_peak_dbfs)
    out = _scale(s, g, lim)
    return (out if stereo else out.reshape(-1)), _info(peak, g, ch)


def estimate_memory(n_mix: int, stereo: bool, voices: Sequence[int], parallel: int) -> tuple[float, float]:
    """(host MB, temp disk MB): host = mixer chunk buffers + one 1-s float32 piece per running job (independent of
    the score length); disk = voice files (float32) + float32 sum + 16-bit WAV."""
    ch = 2 if stereo else 1
    host = (3 * (1 << 16) * ch * 8 + max(1, int(parallel)) * 2 * 48000 * 4) / 1e6
    disk = (sum(int(v) for v in voices) * 4 + int(n_mix) * ch * 4 + int(n_mix) * ch * 2) / 1e6
    return host, disk
