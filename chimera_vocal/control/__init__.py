"""Control and geometry: articulators, gesture score, vocal-tract geometry, control track (C2, INTERFACES §6.3).

The seam types are re-exported lazily (PEP 562), so ``import chimera_vocal.control`` stays cheap and loading a
JIT file of this package by path never pulls in the rest:
``from chimera_vocal.control import ControlModel, TrackChunk, GeometryImpl, GEOMETRY_VEC, GEOMETRY_REF_CPYTHON``.
"""

from __future__ import annotations

import importlib

_LAZY = {
    "GeometryImpl": "geometry",
    "GEOMETRY_REF_CPYTHON": "geometry",
    "GEOMETRY_VEC": "geometry",
    "TrackChunk": "track",
    "ControlModel": "track",
    "JIT_FILES": "jit_manifest",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value
