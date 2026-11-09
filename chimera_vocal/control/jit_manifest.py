"""JIT files of the control package. The worker's loader (dsp/backend.py) loads them after
``core.jit.CORE_JIT_FILES`` and before the acoustics' files."""

from __future__ import annotations

from ..core.jit import JitFile

JIT_FILES: tuple[JitFile, ...] = (
    JitFile("control/geometry_ref.py", "_chimera_geometry_ref", ("geometry_compute", "geometry_resample")),
)
