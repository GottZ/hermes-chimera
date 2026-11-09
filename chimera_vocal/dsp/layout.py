"""Row and field indices of the kernel state: one state, two kernels.

The constants are defined in ``kernel_numba.py`` (a JIT file may import nothing from the package, and numba reads
module globals as compile-time constants); this module re-exports them for the builder, the numpy kernel, the
voice and the tests, and adds name tables. New rows are added in ``kernel_numba.py`` only.
"""

from __future__ import annotations

from . import kernel_numba as _k

_NAMES = [n for n in dir(_k) if n.isupper() and not n.startswith("_")]
globals().update({n: getattr(_k, n) for n in _NAMES})
__all__ = list(_NAMES) + ["NODE_ROWS", "NODE_INT_ROWS", "BRANCH_ROWS", "BRANCH_INT_ROWS", "RAD_ROWS",
                          "NOISE_COLS", "GLOT_FIELDS", "GLOT_INT_FIELDS", "SC_FIELDS", "SI_FIELDS", "row_names"]


def _table(prefix: str, size_name: str) -> tuple[str, ...]:
    """Names of the rows prefix* in index order, e.g. ("N_P", "N_AREA", ...); checks they cover 0..size-1 once."""
    size = getattr(_k, size_name)
    rows = sorted((getattr(_k, n), n) for n in _NAMES
                  if n.startswith(prefix) and n != size_name and isinstance(getattr(_k, n), int))
    names = [n for _, n in rows]
    idx = [i for i, _ in rows]
    assert idx == list(range(size)), (prefix, idx, size)
    return tuple(names)


NODE_ROWS = _table("N_", "NF")
NODE_INT_ROWS = _table("NI_", "NI")
BRANCH_ROWS = _table("B_", "BF")
BRANCH_INT_ROWS = _table("BI_", "BI")
RAD_ROWS = _table("R_", "RF")
NOISE_COLS = _table("Z_", "NZ")
SC_FIELDS = _table("S_", "S")
SI_FIELDS = _table("SI_", "SI")
GLOT_INT_FIELDS = _table("GI_", "GI")


def _glot_fields() -> tuple[str, ...]:
    out = [""] * _k.G
    for n in _NAMES:
        if n.startswith("G_") and n not in ("G_DRIFT_Y", "G_DRIFT_A"):
            out[getattr(_k, n)] = n
    for i in range(3):
        out[_k.G_DRIFT_Y + i] = f"G_DRIFT_Y{i}"
        out[_k.G_DRIFT_A + i] = f"G_DRIFT_A{i}"
    assert all(out), out
    return tuple(out)


GLOT_FIELDS = _glot_fields()


def row_names(array: str) -> tuple[str, ...]:
    """Row/field names of a VoiceState array ("nf", "ni", "bf", "bi", "rf", "noise_f", "glot", "glot_i", "sc", "si")."""
    return {"nf": NODE_ROWS, "ni": NODE_INT_ROWS, "bf": BRANCH_ROWS, "bi": BRANCH_INT_ROWS, "rf": RAD_ROWS,
            "noise_f": NOISE_COLS, "glot": GLOT_FIELDS, "glot_i": GLOT_INT_FIELDS, "sc": SC_FIELDS,
            "si": SI_FIELDS}[array]
