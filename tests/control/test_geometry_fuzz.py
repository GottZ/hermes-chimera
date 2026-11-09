"""K-2.4: differential fuzz of the two CPython-side geometry forms, GEOMETRY_REF_CPYTHON vs GEOMETRY_VEC, on random
postures uniformly inside the articulator ranges (articulators.rs:24-46) over five bodies, plus edge postures (every
articulator at its range ends). Everything the frame exposes is compared -- grid (all columns), scalars incl.
contacts and normals, velum_port, teeth_s, outlines -- and resample with the V2 section counts of the body:
<= 1e-12 relative (floor 1e-6), integers exact.
"""

import numpy as np
import pytest

FUZZ_TOL = 1e-12
LABELS = ("male", "female", "child", "mix-f050", "mix-f050-c050")


def _edge_postures(c2):
    """NEUTRAL with one articulator at its min / max, and all at min / all at max."""
    base = np.array(c2.TEMPLATE_NEUTRAL)
    out = [np.array([lo for lo, _, _ in c2.TEMPLATE_ARTIC_SPECS]), np.array([hi for _, hi, _ in c2.TEMPLATE_ARTIC_SPECS])]
    for i, (lo, hi, _) in enumerate(c2.TEMPLATE_ARTIC_SPECS):
        for x in (lo, hi):
            a = base.copy()
            a[i] = x
            out.append(a)
    return out


def _fuzz(c2, n_random, seed):
    from chimera_vocal.control.geometry import GEOMETRY_REF_CPYTHON, GEOMETRY_VEC

    rng = np.random.default_rng(seed)
    posts = [(LABELS[i % len(LABELS)], a) for i, a in enumerate(_edge_postures(c2))]
    posts += [(LABELS[i % len(LABELS)], c2.random_artic(rng)) for i in range(n_random)]
    failures = []
    for i, (label, a) in enumerate(posts):
        body = c2.body_of(label)
        idx = c2.v2_entry(label)
        ref = c2.run_geometry(GEOMETRY_REF_CPYTHON, body, a, idx["n_ph"], idx["n_or"])
        vec = c2.run_geometry(GEOMETRY_VEC, body, a, idx["n_ph"], idx["n_or"])
        diff = c2.compare_geometry(ref, vec, FUZZ_TOL)
        if diff:
            failures.append(f"posture {i} ({label}, artic={a.tolist()}): {diff[:4]}")
    assert not failures, f"{len(failures)} of {len(posts)} postures differ:\n" + "\n".join(failures[:10])
    return len(posts)


def test_geometry_fuzz(c2):
    """K-2.4 (quick): 36 edge + 200 random postures."""
    assert _fuzz(c2, 200, 20261109) == 236


@pytest.mark.slow
def test_geometry_fuzz_full(c2):
    """K-2.4 (TESTPLAN size): 2 000 random postures, another seed."""
    assert _fuzz(c2, 2000, 4242) == 2036


def test_fuzz_postures_reach_the_branches(c2):
    """Guard of the harness: the random postures cover the conditional branches of compute that depend only on
    the articulation (lateral > 0, dorsum > 0, labiodental > 0, velum > 0, compressed lips), each often."""
    rng = np.random.default_rng(20261109)
    a = np.array([c2.random_artic(rng) for _ in range(200)])
    from chimera_vocal.core.keys import ARTIC_INDEX as I

    for key in ("lateral", "dorsum", "labiodental", "velum"):
        assert (a[:, I[key]] > 0).sum() >= 50, key
    assert (a[:, I["lipAp"]] < 0).sum() >= 10
