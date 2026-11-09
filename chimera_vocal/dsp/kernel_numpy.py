"""Acoustic kernel, numpy form: the fallback that runs everywhere numpy runs (INTERFACES §6.4, DESIGN §2.4).

Same three entry points and the same semantics as ``kernel_numba`` on the same state arrays; per control block the
geometry-dependent network terms are precomputed as arrays, per sample only the state-dependent rest runs.

Ported from chimera-vocal-synth crates/chimera/src/airway/network.rs, sources/glottis.rs, airway/decimator.rs and
voice/human_voice.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import numpy as np

from . import layout as L


class KernelImpl(NamedTuple):
    """Plain container OUTSIDE the JIT (INTERFACES §6.4); never passed into a JIT function."""

    name: Literal["numba", "numpy"]
    run_samples: Callable
    output_chain: Callable
    flush_subnormals: Callable


def glottis_block(glot, glot_i, dp, area, f0_now, voicing):
    """Same contract as kernel_numba.glottis_block (the glottis step this kernel uses)."""
    raise NotImplementedError


def flush_subnormals(nf, bf, rf, noise_f, glot, sc):
    """Same contract as kernel_numba.flush_subnormals."""
    raise NotImplementedError


def run_samples(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, glot, glot_i, sc, si,
                ph_nodes, or_nodes,
                t_area, t_min, t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung,
                n_internal, raw):
    """Same contract as kernel_numba.run_samples."""
    raise NotImplementedError


def output_chain(sc, si, dec_c, dec_buf, raw, n_internal, out):
    """Same contract as kernel_numba.output_chain."""
    raise NotImplementedError


KERNEL_NUMPY = KernelImpl("numpy", run_samples, output_chain, flush_subnormals)

__all__ = ["KernelImpl", "KERNEL_NUMPY", "run_samples", "output_chain", "flush_subnormals", "glottis_block", "L"]
