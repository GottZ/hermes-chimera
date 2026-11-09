"""Vocal-tract geometry, scalar reference form: HumanVocalTract::compute / ::resample (anatomy/human.rs:725-1370,
anatomy/geom.rs) in the numba subset.

JIT file (INTERFACES §6.4a): loaded by the worker's numba loader under the fixed module name
``_chimera_geometry_ref`` (entry points ``geometry_compute``, ``geometry_resample``; manifest in
``control/jit_manifest.py``). Imported as part of the package, ``jit`` is the identity and this module is the
CPython reference of the tests (``GEOMETRY_REF_CPYTHON`` in ``control/geometry.py``).

Rules: imports only ``math``, ``numpy`` and ``_chimera_nbmath``; every function is ``@jit``; functions take only
numpy arrays (float64, C-contiguous) and Python scalars; no classes. Because a JIT file cannot import the
package, the frame and tract layouts are defined HERE and re-exported by ``control/frame_layout.py``.

Frame (float64[NFRAME], written by geometry_compute, read by geometry_resample and the control model):
scalars at F_* (the first 16 in the order of the golden test's ``scalars``, golden_geometry.rs:37), the grid as
rows of NG columns starting at F_GRID (row count in F_N_GRID), six outlines as interleaved (x, y) points starting
at F_OUTLINES + o * 2 * MAX_PTS (point counts in F_N_OUTLINE + o), then private scratch space of the geometry.
Tract (float64[NTRACT], written by frame_layout.tract_constants(body)): everything HumanVocalTract::new
precomputes (human.rs:644-679). Its member offsets are private to this module (S2 fills them in).
"""

import math

import numpy as np

try:
    import _chimera_nbmath as nbm
except ImportError:
    from ..core import nbmath as nbm

jit = globals().get("CHIMERA_JIT", lambda f: f)

# ---- frame layout -------------------------------------------------------------------------------------------
MAX_GRID = 64        # grid rows (template: 12 pharyngeal + 31 oral fan + 6 front rays = 49 at most)
MAX_PTS = 256        # points per outline (human.rs:568 POINTS)

# scalars; F_LENGTH .. F_CONTACT_BODY_NY are the golden test's 16 `scalars` in that order (golden_geometry.rs:37)
F_LENGTH = 0
F_PHARYNX_LENGTH = 1
F_TEETH_S = 2
F_LIP_S = 3
F_VELUM_PORT = 4
F_LIP_AREA = 5
F_TONGUE_RADIUS = 6
F_TIP_X = 7
F_TIP_Y = 8
F_SPLIT_INDEX = 9
F_CONTACT_TIP = 10
F_CONTACT_BODY = 11
F_CONTACT_TIP_NX = 12
F_CONTACT_TIP_NY = 13
F_CONTACT_BODY_NX = 14
F_CONTACT_BODY_NY = 15
F_GLOTTIS_X = 16
F_GLOTTIS_Y = 17
F_TONGUE_CX = 18
F_TONGUE_CY = 19
F_N_GRID = 20
F_N_OUTLINE = 21     # 21..26: point count of outline o (O_* order)
N_SCALAR_SLOTS = 32

# grid row r, column c at F_GRID + r * NG + c (GridSample, human.rs:453-468)
G_REGION = 0         # 0 epilarynx, 1 pharynx, 2 oral, 3 front
G_D = 1
G_AREA = 2
G_S = 3
G_MID_X = 4
G_MID_Y = 5
G_IN_X = 6
G_IN_Y = 7
G_OUT_X = 8
G_OUT_Y = 9
G_RAY = 10
NG = 11
F_GRID = N_SCALAR_SLOTS

# outlines (HumanFrame fields checked by the golden test, golden_geometry.rs:38-42)
O_OUTER = 0
O_INNER = 1
O_TONGUE = 2
O_VELUM = 3
O_UPPER_LIP = 4
O_LOWER_LIP = 5
N_OUTLINES = 6
F_OUTLINES = F_GRID + MAX_GRID * NG

F_SCRATCH = F_OUTLINES + N_OUTLINES * 2 * MAX_PTS
N_SCRATCH = 16384    # private working space of geometry_compute (sized in S2)
NFRAME = F_SCRATCH + N_SCRATCH

# ---- tract layout (HumanVocalTract::new, human.rs:644-679) ---------------------------------------------------
NTRACT = 1024        # provisional size; member offsets are defined with the implementation (S2)


@jit
def geometry_compute(artic, tract, frame):
    """HumanVocalTract::compute (human.rs:725-1263) for ``artic`` (float64[17], ARTIC_KEYS order) on the body
    whose constants are ``tract`` (float64[NTRACT]); writes grid, scalars, contacts and outlines into ``frame``
    (float64[NFRAME]). Stateless apart from ``frame``: the contacts the template keeps on the tract are frame
    scalars here."""
    raise NotImplementedError


@jit
def geometry_resample(artic, tract, frame, out_ph, out_or, min_ph, min_or, lens):
    """HumanVocalTract::resample (human.rs:1288-1366) of the frame last computed for ``artic``: harmonic-mean
    section areas into out_ph[n_ph] / out_or[n_or], sub-section minima into min_ph / min_or, and
    lens[0] = pharynx length, lens[1] = oral length (cm)."""
    raise NotImplementedError
