"""Vocal-tract geometry, numpy form (INTERFACES §6.3 ``GEOMETRY_VEC``; DESIGN §2.5).

The same function as ``geometry_ref`` (HumanVocalTract::compute / ::resample), vectorised per posture over
rays x wall segments for the numpy kernel path (CPython). Sequential places stay small loops: the spike clamp
(human.rs:1196-1201, each sample against its already clamped left neighbour), the stable insertion sort of the
ray crossings (human.rs:946-956) and the binary search of the resampling (human.rs:1303-1316). Same frame and
tract layout as ``geometry_ref`` (``frame_layout``). Agreement with the reference: <= 1e-12 relative on random
postures (TESTPLAN K-2.4), T2 on the golden postures (K-2.2).
"""

from __future__ import annotations

import numpy as np


def compute(artic: np.ndarray, tract: np.ndarray, frame: np.ndarray) -> None:
    """HumanVocalTract::compute (human.rs:725-1263) into ``frame``; signature of GeometryImpl.compute."""
    raise NotImplementedError


def resample(artic: np.ndarray, tract: np.ndarray, frame: np.ndarray, out_ph: np.ndarray, out_or: np.ndarray,
             min_ph: np.ndarray, min_or: np.ndarray, lens: np.ndarray) -> None:
    """HumanVocalTract::resample (human.rs:1288-1366); signature of GeometryImpl.resample."""
    raise NotImplementedError
