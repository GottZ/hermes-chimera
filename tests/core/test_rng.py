"""K-0.3 (mulberry32 vectors, CPython + numba), K-0.4 (Drift vectors) -- V1 tests/vectors/rng.json."""

import json
import math
import subprocess
import sys

import pytest

from chimera_vocal.core import jsmath as jm
from chimera_vocal.core import nbmath as nb
from support import PACKAGE_DIR, REPO_ROOT, rel, vector


def _cases():
    return vector("rng")["rng"]


def test_rng_vectors_class():
    cases = _cases()
    assert {c["seed"] for c in cases} >= {12345, 777, 4242, 1, 0}
    for c in cases:
        r = jm.Rng(c["seed"])
        for i, want in enumerate(c["next"]):
            got = r.next()
            assert got == want, (c["seed"], i, got, want)
            if i < len(c["states"]):
                assert r.state == c["states"][i], (c["seed"], i, r.state, c["states"][i])
        r = jm.Rng(c["seed"])
        assert [r.gauss() for _ in c["gauss"]] == c["gauss"], c["seed"]
        r = jm.Rng(c["seed"])
        assert [r.bipolar() for _ in c["bipolar"]] == c["bipolar"], c["seed"]


def test_rng_vectors_functional():
    """nbmath.rng_next / rng_gauss on the same vectors (the form the kernels use)."""
    for c in _cases():
        s = (c["seed"] & 0xFFFFFFFF) or 1
        for i, want in enumerate(c["next"]):
            s, v = nb.rng_next(s)
            assert v == want and (i >= len(c["states"]) or s == c["states"][i])
        s = (c["seed"] & 0xFFFFFFFF) or 1
        for want in c["gauss"]:
            s, v = nb.rng_gauss(s)
            assert v == want


def test_drift_vectors():
    """K-0.4: <= 1e-15 relative (libm exp vs. the platform's; sqrt and the arithmetic are exact)."""
    for d in vector("rng")["drift"]:
        dr = jm.Drift(d["control_rate"], d["base_hz"], d["seed"])
        worst = 0.0
        for i, want in enumerate(d["steps"]):
            worst = max(worst, rel(dr.step(), want, 1e-300))
        assert worst <= 1e-15, (d, worst)


_NUMBA_SCRIPT = r"""
import importlib.util, json, math, sys
import numba
path, payload = sys.argv[1], json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location("_chimera_nbmath", path)
mod = importlib.util.module_from_spec(spec)
mod.CHIMERA_JIT = numba.njit(cache=False, nogil=True, fastmath=False, error_model="numpy")
sys.modules["_chimera_nbmath"] = mod
spec.loader.exec_module(mod)
assert isinstance(mod.rng_next, numba.core.registry.CPUDispatcher)
out = {"rng": [], "scalar": {}}
for seed, n, ng in payload["seeds"]:
    s = (seed & 0xFFFFFFFF) or 1
    nx, st = [], []
    for _ in range(n):
        s, v = mod.rng_next(s); nx.append(v); st.append(s)
    s = (seed & 0xFFFFFFFF) or 1
    g = []
    for _ in range(ng):
        s, v = mod.rng_gauss(s); g.append(v)
    out["rng"].append({"seed": seed, "next": nx, "states": st, "gauss": g})
def enc(x):
    return repr(float(x)) if isinstance(x, float) else x
for name, args_list in payload["scalar"].items():
    f = getattr(mod, name)
    out["scalar"][name] = [enc(f(*[float(a) for a in args])) for args in args_list]
print(json.dumps(out))
"""

_SCALAR_INPUTS = {
    "js_round": [[x] for x in (2.5, -2.5, 0.49999999999999994, -0.6, -0.4, -0.0, 1e300, "inf", "nan")],
    "js_min": [[0.0, -0.0], [-0.0, 0.0], ["nan", 1.0], [1.0, 2.0]],
    "js_max": [[0.0, -0.0], [-0.0, 0.0], [1.0, "nan"], [1.0, 2.0]],
    "js_sign": [[x] for x in (3.0, -2.0, -0.0, "nan")],
    "js_trunc_i32": [[x] for x in (3.9, -3.9, 2147483648.0, 4294967301.0, "nan", "inf", 1e30)],
    "clamp": [["nan", 0.0, 1.0], [-1.0, 0.0, 1.0], [0.25, 0.0, 1.0]],
    "smoothstep": [[0.0, 1.0, 0.3], [1.0, 1.0, 2.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
    "fdiv": [[1.0, 0.0], [-1.0, 0.0], [1.0, -0.0], [0.0, 0.0], [3.0, 4.0]],
}


@pytest.mark.numba
def test_rng_vectors_numba():
    """K-0.3n: nbmath compiled by numba (fixed module name, worker-style loading) is bit-identical to the
    vectors and to CPython for every scalar helper. Runs in a subprocess: this test process never imports
    numba (INTERFACES §6.4a rule 5)."""
    cases = _cases()
    payload = {"seeds": [[c["seed"], len(c["next"]), len(c["gauss"])] for c in cases], "scalar": _SCALAR_INPUTS}
    r = subprocess.run([sys.executable, "-I", "-c", _NUMBA_SCRIPT, str(PACKAGE_DIR / "core" / "nbmath.py")],
                       input=json.dumps(payload), capture_output=True, text=True, timeout=600, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout)
    for c, got in zip(cases, out["rng"]):
        assert got["next"] == c["next"] and got["states"][: len(c["states"])] == c["states"], c["seed"]
        assert got["gauss"] == c["gauss"], c["seed"]
    for name, args_list in _SCALAR_INPUTS.items():
        f = getattr(nb, name)
        for args, g in zip(args_list, out["scalar"][name]):
            want = f(*[float(a) for a in args])
            got = float(g) if isinstance(g, str) else g
            if isinstance(want, float):
                same = (want != want and got != got) or (
                    want == got and math.copysign(1.0, want) == math.copysign(1.0, got))
            else:
                same = want == got
            assert same, (name, args, got, want)
