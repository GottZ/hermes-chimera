"""K-2.1 / K-2.2: the 114 golden postures (tests/golden/geometry.json = template crates/chimera/tests/golden,
checked by golden_geometry.rs:14-64) through both CPython-side geometry forms.

Per case: grid size and regions exact; grid d/area/s, the 16 scalars (incl. contacts and their normals), the
outlines present in the case and resample(18, 18) (lengths, areas, minima) <= 1e-9 relative with floor 1e-6 (T2).
"""

import pytest

import support


def _run(c2, impl):
    cases = support.golden("geometry")
    assert len(cases) == 114
    failures, worst = [], 0.0
    for ci, case in enumerate(cases):
        w, problems = c2.check_golden_case(impl, case)
        worst = max(worst, w)
        if problems or not w < c2.T2_TOL:
            failures.append(f"case {ci} ({case['body']}): worst {w:.3e} {problems}")
    assert not failures, f"{len(failures)} of 114 cases fail T2:\n" + "\n".join(failures[:20])
    return worst


def test_golden_geometry_ref(c2):
    """K-2.1: GEOMETRY_REF_CPYTHON (control/geometry_ref.py with jit = identity)."""
    from chimera_vocal.control.geometry import GEOMETRY_REF_CPYTHON

    assert GEOMETRY_REF_CPYTHON.name == "ref-cpython"
    print("ref worst", _run(c2, GEOMETRY_REF_CPYTHON))


def test_golden_geometry_vec(c2):
    """K-2.2: GEOMETRY_VEC (control/geometry_vec.py, numpy)."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC

    assert GEOMETRY_VEC.name == "vec"
    print("vec worst", _run(c2, GEOMETRY_VEC))


@pytest.mark.parametrize("key", ["outer", "inner", "tongue", "velum", "upperLip", "lowerLip", "grid", "scalars"])
def test_golden_geometry_coverage(key):
    """The golden file really exercises what the checks above read (guards the harness, not the geometry)."""
    cases = support.golden("geometry")
    present = sum(1 for c in cases if key in c)
    assert present >= 1, key
    regions = {int(r[0]) for c in cases for r in c["grid"]}
    assert regions >= {1, 2, 3}
