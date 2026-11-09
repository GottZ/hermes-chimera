"""The geometry seam between control and acoustics: ``GeometryImpl`` and its CPython-side
instances. ``ref-numba`` exists only inside a worker process, built by ``dsp/backend.py`` from the loaded
``_chimera_geometry_ref`` module (entry points in ``control/jit_manifest.py``).

This module is plain Python (not a JIT file); ``GeometryImpl`` is never passed into a compiled function (numba
caches argument types by module path, and the plugin is imported under a different package name per install).
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import numpy as np

from . import geometry_ref, geometry_vec


class GeometryImpl(NamedTuple):
    name: Literal["ref-cpython", "ref-numba", "vec"]
    compute: Callable[[np.ndarray, np.ndarray, np.ndarray], None]
    # compute(artic: float64[17], tract: float64[NTRACT], frame: float64[NFRAME]) -> None
    resample: Callable[..., None]
    # resample(artic, tract, frame, out_ph: float64[n_ph], out_or: float64[n_or], min_ph: float64[n_ph],
    #          min_or: float64[n_or], lens: float64[2]) -> None; lens = (ph_len, or_len)


GEOMETRY_REF_CPYTHON = GeometryImpl("ref-cpython", geometry_ref.geometry_compute, geometry_ref.geometry_resample)
GEOMETRY_VEC = GeometryImpl("vec", geometry_vec.compute, geometry_vec.resample)
