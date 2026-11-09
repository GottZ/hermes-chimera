"""K-3.2 test_glottis_vectors: the kernels' glottis step against V5 (tests/vectors/glottis.json `next`), ≤ 1e-12
relative; with CHIMERA_LAB_REFS also all 2 000 samples and the private state after 1 000 and 2 000 samples."""

from __future__ import annotations

import numpy as np
import pytest

from chimera_vocal.core import glottis_params as gp
from chimera_vocal.core.types import Seeds
from chimera_vocal.dsp import glottis
from chimera_vocal.dsp import layout as L
from support import load_json, rel, vector

V5 = vector("glottis")
SETS = [s["name"] for s in V5["next"]]
TOL = 1e-12


def _worst(got, want, floor=1e-300):
    w = 0.0
    for a, b in zip(got, want):
        w = max(w, rel(float(a), float(b), floor))
    return w


def test_glottis_layout_matches_params():
    """The 17 parameter slots of the kernel layout are core.glottis_params' field order."""
    assert L.GLOTTIS_NPARAMS == gp.N_GLOTTIS == 17
    for i, name in enumerate(gp.GLOTTIS_FIELDS):
        assert getattr(L, "G_" + name.upper()) == i, name


def test_init_glottis_matches_template():
    """ParametricGlottis::new(96000) = V5 full `state_initial` (all sets start from it)."""
    glot, glot_i = glottis.init_glottis(96000.0, Seeds())
    assert glot.shape == (L.G,) and glot.dtype == np.float64 and glot_i.shape == (L.GI,) and glot_i.dtype == np.int64
    assert glot[: gp.N_GLOTTIS].tolist() == list(gp.DEFAULT_GLOTTIS)
    want = vector("topology")["bodies"][0]["initial"]["glottis_state"]
    got = glottis.glottis_state(glot, glot_i)
    assert set(got) == set(glottis.STATE_FIELDS)
    for k, w in want.items():
        g = got[k]
        if isinstance(w, list):
            assert _worst(g, w) <= 1e-15, k
        else:
            assert rel(float(g), float(w), 1e-300) <= 1e-15, (k, g, w)


def test_init_glottis_seeds():
    glot, glot_i = glottis.init_glottis(96000.0, Seeds(1, 2, 3))
    assert glot_i[L.GI_RNG] == 2 and glot_i[L.GI_DRIFT_RNG] == 3
    glot, glot_i = glottis.init_glottis(96000.0, Seeds(1, 0, 0))
    assert glot_i[L.GI_RNG] == 1 and glot_i[L.GI_DRIFT_RNG] == 1, "Rng::new maps seed 0 to 1 (math.rs:188)"


@pytest.mark.parametrize("set_name", SETS)
def test_glottis_vectors(kernel_name, set_name, run):
    i = SETS.index(set_name)
    s = V5["next"][i]
    res = run(kernel_name, "glottis", set_index=i)
    idx = V5["next_index"]
    for key in ("area", "f0_now", "voicing"):
        got = res[key][idx]
        w = _worst(got, s[key])
        assert w <= TOL, f"{set_name} {key}: worst relative difference {w:.3e}"


@pytest.mark.lab
@pytest.mark.parametrize("set_name", SETS)
def test_glottis_vectors_full(kernel_name, set_name, run, lab_refs):
    full = load_json(lab_refs / "tracks" / "glottis-next.full.json")
    s = full["next"][SETS.index(set_name)]
    for n, key in ((1000, "state_after_1000"), (2000, "state_after_2000")):
        res = run(kernel_name, "glottis", set_index=SETS.index(set_name), n=n)
        for k in ("area", "f0_now", "voicing"):
            w = _worst(res[k], s[k][:n])
            assert w <= TOL, f"{set_name} {k} (n={n}): {w:.3e}"
        st = glottis.glottis_state(res["glot"], res["glot_i"])
        for k, want in s[key].items():
            got = st[k]
            if isinstance(want, list):
                assert _worst(got, want) <= TOL, (k, got, want)
            elif isinstance(want, int) and not isinstance(want, bool):
                assert got == want, (k, got, want)
            else:
                assert rel(float(got), float(want), 1e-300) <= TOL, (k, got, want)
