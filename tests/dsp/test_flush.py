"""K-3.8 test_subnormal_flush: flushing |x| < 1e-30 per control block changes the output by
≤ 1e-12 × peak, and after a long pause the state holds no subnormals (without the flush it does)."""

from __future__ import annotations

import numpy as np
import pytest

import support
from chimera_vocal.dsp import layout as L

SLOW_NUMPY = [pytest.param("numpy", id="numpy", marks=pytest.mark.slow),
              pytest.param("numba", id="numba", marks=pytest.mark.numba)]
TINY = np.finfo(np.float64).tiny

FLUSHED = {"nf": ("rows", L.FLUSH_NODE_ROWS), "bf": ("rows", L.FLUSH_BRANCH_ROWS), "rf": ("rows", L.FLUSH_RAD_ROWS),
           "noise_f": ("cols", L.FLUSH_NOISE_COLS), "glot": ("items", L.FLUSH_GLOT), "sc": ("items", L.FLUSH_SC)}


def test_flush_tables_are_state_not_constants():
    """Every flushed slot is decaying state; constants, geometry and per-sample scratch are never touched."""
    assert L.N_AREA not in L.FLUSH_NODE_ROWS and L.N_WALL_M not in L.FLUSH_NODE_ROWS
    assert L.B_PORT_AREA not in L.FLUSH_BRANCH_ROWS and L.R_L_PREV not in L.FLUSH_RAD_ROWS
    assert L.G_F0 not in L.FLUSH_GLOT and L.G_PHASE not in L.FLUSH_GLOT and L.G_DRIFT_Y not in L.FLUSH_GLOT
    assert L.S_DC_COEF not in L.FLUSH_SC and L.S_OUTPUT_GAIN not in L.FLUSH_SC and L.S_T not in L.FLUSH_SC
    assert L.FLUSH_BELOW == 1e-30


def test_flush_unit(kernel_name, run):
    """flush_subnormals zeroes exactly the flushed slots below 1e-30 and leaves everything else bit-identical."""
    res = run(kernel_name, "flush_unit")
    for name, (how, sel) in FLUSHED.items():
        before, after = res[f"before_{name}"], res[f"after_{name}"]
        mask = np.zeros(before.shape, dtype=bool)
        if how == "rows":
            mask[list(sel), :] = True
        elif how == "cols":
            mask[:, list(sel)] = True
        else:
            mask[list(sel)] = True
        small = np.abs(before) < L.FLUSH_BELOW
        want = np.where(mask & small, 0.0, before)
        assert want.tobytes() == after.tobytes(), (
            f"{name}: differs at {np.argwhere(want != after)[:5].tolist()}")
        assert (mask & small).any(), f"{name}: the seeded state has no flushable value (test data)"


@pytest.mark.parametrize("scenario", ["hello-male", "sustain-aa-male"])
def test_flush_output_head(kernel_name, scenario, run):
    """The flush changes the audio of the recorded heads by ≤ 1e-12 × peak."""
    a = run(kernel_name, "flush", flush=True, voiced_blocks=80, pause_blocks=0, scenario=scenario)
    b = run(kernel_name, "flush", flush=False, voiced_blocks=80, pause_blocks=0, scenario=scenario)
    d = support.compare_audio(b["audio"], a["audio"], 1e-12)
    assert d.first_diverge is None, f"flush changed the output from sample {d.first_diverge} ({d.max_err:.2e} of peak)"


@pytest.mark.parametrize("kernel_name", SLOW_NUMPY)
def test_flush_after_pause(kernel_name, run):
    """Speech onset, then 3.2 s of pause: with the flush no subnormal is left in the state at any block end;
    without it subnormals appear (the precondition that makes this test meaningful); audio within 1e-12 × peak."""
    kw = dict(voiced_blocks=80, pause_blocks=2400, scenario="hello-male")
    a = run(kernel_name, "flush", flush=True, **kw)
    b = run(kernel_name, "flush", flush=False, **kw)
    assert int(b["subnormals"].max()) > 0, "no subnormals without the flush: lengthen the pause"
    assert int(a["subnormals"].max()) == 0, f"subnormals left with the flush (max {int(a['subnormals'].max())})"
    d = support.compare_audio(b["audio"], a["audio"], 1e-12)
    assert d.first_diverge is None, f"flush changed the output from sample {d.first_diverge} ({d.max_err:.2e})"


@pytest.mark.lab
def test_flush_output_full(kernel_name, run):
    """Whole "Hello world." (lab track): flush on vs off within 1e-12 × peak."""
    import recorded_control as rc
    n = rc.track("hello-male", full=True).n_samples
    on = run(kernel_name, "track_audio", scenario="hello-male", n_out=n, full=True, flush=True)
    off = run(kernel_name, "track_audio", scenario="hello-male", n_out=n, full=True, flush=False)
    d = support.compare_audio(off["audio"], on["audio"], 1e-12)
    assert d.first_diverge is None, f"flush changed the output from sample {d.first_diverge} ({d.max_err:.2e})"
