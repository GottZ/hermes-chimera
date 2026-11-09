"""WAV I/O. Writing follows chimera-cli/src/wav.rs:24-33 exactly: each sample is first rounded to float32
(the engine's output type), then as float64 non-finite -> 0.0, clamp to [-1, 1], int16 = js_round(v*32767).
``bits=32`` writes IEEE float32 (format 3) like the CLI's ``--float``. Mono ``(n,)`` or interleaved stereo
``(n, 2)``. Imports numpy (load on demand only).
"""

from __future__ import annotations

import io
import os
import struct
from typing import BinaryIO, Iterable

import numpy as np

_FMT_PCM = 1
_FMT_FLOAT = 3
_FMT_EXTENSIBLE = 0xFFFE


def _frames(samples: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(samples)
    if x.ndim == 1:
        return x, 1
    if x.ndim == 2 and x.shape[1] in (1, 2):
        return x, int(x.shape[1])
    raise ValueError(f"samples must be (n,) or (n, 2), got shape {x.shape}")


def _encode(x: np.ndarray, bits: int) -> bytes:
    """Interleaved little-endian sample bytes for a (n,) or (n, c) block."""
    f32 = np.asarray(x, dtype=np.float32)
    if bits == 32:
        return np.ascontiguousarray(f32, dtype="<f4").tobytes()
    if bits != 16:
        raise ValueError("bits must be 16 or 32")
    v = f32.astype(np.float64)
    v[~np.isfinite(v)] = 0.0
    np.clip(v, -1.0, 1.0, out=v)
    y = v * 32767.0
    r = np.floor(y)
    r = r + ((y - r) >= 0.5)          # js_round: halves toward +inf
    return np.ascontiguousarray(r.astype("<i2")).tobytes()


def _header(n_frames: int, rate: int, channels: int, bits: int) -> bytes:
    bps = bits // 8
    data_bytes = n_frames * channels * bps
    if data_bytes + 36 > 0xFFFFFFFF:
        raise ValueError("WAV data larger than 4 GiB")
    fmt = _FMT_FLOAT if bits == 32 else _FMT_PCM
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + data_bytes), b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, fmt, channels, rate, rate * channels * bps, channels * bps, bits),
        b"data", struct.pack("<I", data_bytes),
    ))


def wav_bytes(samples: np.ndarray, rate: int, *, bits: int = 16) -> bytes:
    x, ch = _frames(samples)
    return _header(len(x), int(rate), ch, bits) + _encode(x, bits)


def write_wav(path: str | os.PathLike, samples: np.ndarray, rate: int, *, bits: int = 16) -> None:
    data = wav_bytes(samples, rate, bits=bits)
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def write_wav_chunked(path: str | os.PathLike, chunks: Iterable[np.ndarray], rate: int, channels: int,
                      n_frames: int) -> None:
    """16-bit WAV written header first (``n_frames`` known), then chunk by chunk (memory stays bounded for long mixes).

    Byte-identical to ``write_wav(path, concatenate(chunks), rate)``. A frame-count mismatch removes the file
    and raises ValueError.
    """
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    written = 0
    try:
        with open(path, "wb") as f:
            f.write(_header(int(n_frames), int(rate), channels, 16))
            for c in chunks:
                x, ch = _frames(c)
                if ch != channels:
                    raise ValueError(f"chunk has {ch} channels, expected {channels}")
                f.write(_encode(x, 16))
                written += len(x)
        if written != n_frames:
            raise ValueError(f"wrote {written} frames, header says {n_frames}")
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def _read_chunks(f: BinaryIO) -> tuple[tuple[int, int, int, int], bytes]:
    if f.read(4) != b"RIFF":
        raise ValueError("not a RIFF file")
    f.read(4)
    if f.read(4) != b"WAVE":
        raise ValueError("not a WAVE file")
    fmt = None
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            raise ValueError("no data chunk")
        cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
        if cid == b"fmt ":
            raw = f.read(size)
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", raw[:16])
            if tag == _FMT_EXTENSIBLE and len(raw) >= 26:
                tag = struct.unpack("<H", raw[24:26])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            return fmt, f.read(size)
        else:
            f.seek(size + (size & 1), io.SEEK_CUR)
        if cid == b"fmt " and size & 1:
            f.seek(1, io.SEEK_CUR)


def read_wav(path: str | os.PathLike) -> tuple[int, np.ndarray]:
    """(rate, float64 samples): PCM16 scaled by 1/32767 (inverse of the writer), float32 as is.

    Mono -> (n,), stereo -> (n, 2).
    """
    with open(path, "rb") as f:
        (tag, ch, rate, bits), data = _read_chunks(f)
    if tag == _FMT_PCM and bits == 16:
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32767.0
    elif tag == _FMT_FLOAT and bits == 32:
        x = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV format tag {tag} with {bits} bits")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch)
    return int(rate), x


def read_f32(path: str | os.PathLike) -> np.ndarray:
    """Raw little-endian float32 (the template's fixtures), as float32."""
    return np.fromfile(path, dtype="<f4")
