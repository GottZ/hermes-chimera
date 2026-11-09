"""Layout of the geometry frame and tract constants (INTERFACES §6.3), plus readers for tests and the control model.

The offsets are defined in the JIT file ``geometry_ref.py`` (a JIT file cannot import the package) and
re-exported here. Callers outside the geometry read a frame only through the functions below, so the layout can
change without touching them.
"""

from __future__ import annotations

import numpy as np

from ..core.body import HumanBody
from .geometry_ref import (  # noqa: F401  (re-exported layout)
    F_CONTACT_BODY,
    F_CONTACT_BODY_NX,
    F_CONTACT_BODY_NY,
    F_CONTACT_TIP,
    F_CONTACT_TIP_NX,
    F_CONTACT_TIP_NY,
    F_GLOTTIS_X,
    F_GLOTTIS_Y,
    F_GRID,
    F_LENGTH,
    F_LIP_AREA,
    F_LIP_S,
    F_N_GRID,
    F_N_OUTLINE,
    F_OUTLINES,
    F_PHARYNX_LENGTH,
    F_SCRATCH,
    F_SPLIT_INDEX,
    F_TEETH_S,
    F_TIP_X,
    F_TIP_Y,
    F_TONGUE_CX,
    F_TONGUE_CY,
    F_TONGUE_RADIUS,
    F_VELUM_PORT,
    G_AREA,
    G_D,
    G_IN_X,
    G_IN_Y,
    G_MID_X,
    G_MID_Y,
    G_OUT_X,
    G_OUT_Y,
    G_RAY,
    G_REGION,
    G_S,
    MAX_GRID,
    MAX_PTS,
    N_OUTLINES,
    N_SCALAR_SLOTS,
    N_SCRATCH,
    NFRAME,
    NG,
    NTRACT,
    O_INNER,
    O_LOWER_LIP,
    O_OUTER,
    O_TONGUE,
    O_UPPER_LIP,
    O_VELUM,
)

# the golden test's 16 scalars, in its order (golden_geometry.rs:37); frame slots F_LENGTH + i
GOLDEN_SCALARS: tuple[str, ...] = (
    "length", "pharynx_length", "teeth_s", "lip_s", "velum_port", "lip_area", "tongue_radius", "tip_x", "tip_y",
    "split_index", "contact_tip", "contact_body", "contact_tip_nx", "contact_tip_ny", "contact_body_nx",
    "contact_body_ny",
)
_EXTRA_SCALARS = (("glottis_x", F_GLOTTIS_X), ("glottis_y", F_GLOTTIS_Y), ("tongue_cx", F_TONGUE_CX),
                  ("tongue_cy", F_TONGUE_CY), ("n_grid", F_N_GRID))

# outline names (golden JSON keys in parentheses: outer, inner, tongue, velum, upperLip, lowerLip), O_* order
OUTLINES: tuple[str, ...] = ("outer", "inner", "tongue", "velum", "upper_lip", "lower_lip")
GRID_COLUMNS: tuple[str, ...] = ("region", "d", "area", "s", "mid_x", "mid_y", "inner_x", "inner_y", "outer_x",
                                 "outer_y", "ray")


def alloc_frame() -> np.ndarray:
    """A zeroed frame, float64[NFRAME] (C-contiguous)."""
    return np.zeros(NFRAME, dtype=np.float64)


def tract_constants(body: HumanBody) -> np.ndarray:
    """float64[NTRACT]: everything HumanVocalTract::new precomputes for ``body`` (human.rs:644-679): scale, body
    numbers, area law, landmarks (PNS, condyle, polar centres) and the fixed polylines (pharyngeal wall, palate,
    nasal cavity)."""
    raise NotImplementedError


def frame_scalars(frame: np.ndarray) -> dict[str, float]:
    """The scalars of a computed frame: GOLDEN_SCALARS (split_index as float, like the golden file) plus glottis,
    tongue centre and grid size."""
    out = {name: float(frame[F_LENGTH + i]) for i, name in enumerate(GOLDEN_SCALARS)}
    for name, off in _EXTRA_SCALARS:
        out[name] = float(frame[off])
    return out


def frame_grid(frame: np.ndarray) -> dict[str, np.ndarray]:
    """The grid of a computed frame as columns (GRID_COLUMNS) of length n_grid; ``region`` as int64."""
    n = int(frame[F_N_GRID])
    if not 0 <= n <= MAX_GRID:
        raise ValueError(f"grid size {n} outside 0..{MAX_GRID}")
    rows = frame[F_GRID:F_GRID + n * NG].reshape(n, NG)
    cols = {name: rows[:, i].copy() for i, name in enumerate(GRID_COLUMNS)}
    cols["region"] = cols["region"].astype(np.int64)
    return cols


def frame_outline(frame: np.ndarray, name: str) -> np.ndarray:
    """Outline ``name`` (one of OUTLINES) as flat interleaved coordinates [x0, y0, x1, y1, ...] (the golden
    file's form)."""
    o = OUTLINES.index(name)
    n = int(frame[F_N_OUTLINE + o])
    if not 0 <= n <= MAX_PTS:
        raise ValueError(f"outline {name}: {n} points outside 0..{MAX_PTS}")
    base = F_OUTLINES + o * 2 * MAX_PTS
    return frame[base:base + 2 * n].copy()
