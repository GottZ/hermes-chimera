"""State layout and JIT manifest of the acoustics (INTERFACES §6.2, §6.4, §6.4a) — consistency checks."""

from __future__ import annotations

import ast

from chimera_vocal.core import glottis_params as gp
from chimera_vocal.core.jit import CORE_JIT_FILES, JitFile
from chimera_vocal.dsp import kernel_numba, kernel_numpy
from chimera_vocal.dsp import layout as L
from chimera_vocal.dsp.jit_manifest import JIT_FILES
from support import PACKAGE_DIR

ENTRY = ("run_samples", "output_chain", "flush_subnormals")


def test_manifest_is_the_interface():
    assert JIT_FILES == (JitFile("dsp/kernel_numba.py", "_chimera_kernel_numba", ENTRY),)
    for f in CORE_JIT_FILES + JIT_FILES:
        assert (PACKAGE_DIR / f.relpath).is_file()


def test_both_kernels_export_the_entry_points():
    assert kernel_numpy.KernelImpl._fields == ("name",) + ENTRY
    assert kernel_numpy.KERNEL_NUMPY.name == "numpy"
    for name in ENTRY + ("glottis_block",):
        a = getattr(kernel_numba, name)
        b = getattr(kernel_numpy, name)
        assert callable(a) and callable(b)
        assert a.__code__.co_varnames[: a.__code__.co_argcount] == b.__code__.co_varnames[: b.__code__.co_argcount], name
    assert kernel_numpy.KERNEL_NUMPY.run_samples is kernel_numpy.run_samples


def test_layout_tables_cover_every_slot():
    assert len(L.NODE_ROWS) == L.NF and len(L.BRANCH_ROWS) == L.BF and len(L.RAD_ROWS) == L.RF
    assert len(L.NOISE_COLS) == L.NZ and len(L.SC_FIELDS) == L.S and len(L.SI_FIELDS) == L.SI
    assert len(L.GLOT_FIELDS) == L.G and len(L.GLOT_INT_FIELDS) == L.GI
    assert L.row_names("nf")[L.N_WALL_Y] == "N_WALL_Y"


def test_glottis_param_slots_match_core():
    assert L.GLOTTIS_NPARAMS == gp.N_GLOTTIS
    for i, name in enumerate(gp.GLOTTIS_FIELDS):
        assert getattr(kernel_numba, "G_" + name.upper()) == i == getattr(gp, "G_" + name.upper())


def test_layout_constants_are_literals_in_the_jit_file():
    """numba reads module globals of the JIT file as constants: every layout name is a plain literal assignment
    (no computed values that could differ between the package import and the worker's fixed-name import)."""
    tree = ast.parse((PACKAGE_DIR / "dsp" / "kernel_numba.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            ok = isinstance(node.value, (ast.Constant, ast.Tuple, ast.BinOp, ast.Name)) or (
                isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "math")
            assert ok, ast.dump(node)[:200]


def test_physics_constants_match_core():
    """The JIT file carries its own copies of physics.rs constants (it may not import core.physics)."""
    from chimera_vocal.core import physics
    import math
    for name in ("RHO", "C_SOUND", "MU", "NU", "RHO_C2", "RE_CRIT"):
        assert getattr(kernel_numba, name) == getattr(physics, name), name
    assert kernel_numba.PI == math.pi and kernel_numba.TAU == math.pi * 2.0
