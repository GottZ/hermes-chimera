"""Comparison helpers for kernel state and audio (tolerance classes T3, T5, T6)."""

from __future__ import annotations

import numpy as np

import dsp_cases
import support

ROWS_2D = ("nf", "bf", "rf", "noise_f")
ROWS_1D_AS_ROW = ("dec_buf",)
ELEMENTWISE = ("glot", "sc")


def state_diff_t5(a: dict, b: dict, *, rtol: float = 1e-9) -> list[str]:
    """T5 (numba vs numpy state): integer arrays exact; float rows element-wise relative ≤ rtol with a floor of
    1e-6 × the row's largest magnitude (2-D rows, dec_buf); glot and sc element-wise relative ≤ rtol."""
    errs: list[str] = []
    for k in dsp_cases.STATE_INT:
        if not np.array_equal(a[k], b[k]):
            i = np.flatnonzero(np.asarray(a[k]).ravel() != np.asarray(b[k]).ravel())[:5]
            errs.append(f"{k}: integers differ at flat {i.tolist()}")
    for k in ROWS_2D + ROWS_1D_AS_ROW + ELEMENTWISE:
        x = np.atleast_2d(np.asarray(a[k], dtype=np.float64))
        y = np.atleast_2d(np.asarray(b[k], dtype=np.float64))
        if x.shape != y.shape:
            errs.append(f"{k}: shape {x.shape} vs {y.shape}")
            continue
        if k in ELEMENTWISE:
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        for r in range(x.shape[0]):
            xr, yr = x[r], y[r]
            same = (xr == yr) | (np.isnan(xr) & np.isnan(yr))
            if same.all():
                continue
            floor = max(1e-6 * float(np.nanmax(np.abs(np.concatenate([xr, yr])))), 1e-300)
            d = np.abs(xr - yr) / np.maximum(np.maximum(np.abs(xr), np.abs(yr)), floor)
            d[same] = 0.0
            d[np.isnan(d)] = np.inf
            w = float(d.max())
            if w > rtol:
                errs.append(f"{k}[{r}]: worst relative {w:.3e} at column {int(d.argmax())}")
    return errs


def assert_state_t5(a: dict, b: dict, label: str = "", *, rtol: float = 1e-9) -> None:
    errs = state_diff_t5(a, b, rtol=rtol)
    assert not errs, f"{label}: " + "; ".join(errs[:12])


def assert_state_equal(a: dict, b: dict, label: str = "") -> None:
    """T6: bit-identical state."""
    for k in dsp_cases.STATE_FLOAT + dsp_cases.STATE_INT:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape and x.tobytes() == y.tobytes(), f"{label}: {k} differs"


def assert_bits(a, b, label: str = "") -> None:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    assert x.shape == y.shape, f"{label}: length {x.shape} vs {y.shape}"
    if x.tobytes() != y.tobytes():
        i = int(np.flatnonzero(x.view(np.uint32) != y.view(np.uint32))[0])
        raise AssertionError(f"{label}: first bit difference at sample {i}: {x[i]!r} vs {y[i]!r}")


def t3(reference, got, label: str = "") -> support.AudioDiff:
    return support.assert_t3(reference, got, label)
