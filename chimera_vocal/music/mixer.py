"""Disk-backed mixer (INTERFACES §8.4, P1-05): peak host memory O(chunk · channels), independent of score length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .compile import VoiceMix


@dataclass(frozen=True)
class MixInfo:
    peak_before: float
    gain_db: float
    channels: int
    clipped_samples: int             # always 0 (pure scaling, no limiter)


class MixWriter:
    """Each finished voice (float32 file) is added chunk-wise into a float32 sum on disk; after the last voice:
    pass 1 peak, pass 2 scale + write WAV chunk-wise. Temp files removed in all outcomes."""

    def __init__(self, tmp_dir: str, n_mix: int, *, stereo: bool, chunk: int = 1 << 16) -> None:
        raise NotImplementedError

    def add_voice(self, pcm_path: str, n: int, mix: "VoiceMix") -> None:
        raise NotImplementedError

    def finish(self, wav_path: str, *, normalize: bool = True, target_peak_dbfs: float = -1.0) -> MixInfo:
        raise NotImplementedError

    def abort(self) -> None:
        raise NotImplementedError


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan: θ = (pan + 1)·π/4 -> (cos θ, sin θ)."""
    raise NotImplementedError


def mix(tracks: "Sequence[np.ndarray]", mixes: "Sequence[VoiceMix]", n_mix: int, *, stereo: bool,
        normalize: bool = True, target_peak_dbfs: float = -1.0) -> "tuple[np.ndarray, MixInfo]":
    """In-memory reference implementation (same arithmetic as MixWriter)."""
    raise NotImplementedError


def estimate_memory(n_mix: int, stereo: bool, voices: Sequence[int], parallel: int) -> tuple[float, float]:
    """(host MB, temp disk MB) for a mix of ``n_mix`` samples and voices of the given rendered lengths."""
    raise NotImplementedError
