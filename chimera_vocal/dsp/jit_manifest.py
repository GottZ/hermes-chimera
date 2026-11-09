"""JIT files of the acoustics (INTERFACES §6.4a). No numba import here."""

from __future__ import annotations

from ..core.jit import JitFile

JIT_FILES: tuple[JitFile, ...] = (
    JitFile("dsp/kernel_numba.py", "_chimera_kernel_numba", ("run_samples", "output_chain", "flush_subnormals")),
)
