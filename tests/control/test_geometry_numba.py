"""K-2.3: geometry_ref compiled by numba (worker-style loading under the fixed module name of
control/jit_manifest.py) equals the CPython reference to <= 1e-13 relative (floor 1e-6); integers (grid size,
regions, split index, outline sizes) exact. numba runs in a subprocess only -- this test process never imports numba.

Also the JIT-file rules of geometry_ref.py and the manifest, which need no numba.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

import support
from support import PACKAGE_DIR, REPO_ROOT

N_RANDOM = 200
NUMBA_TOL = 1e-13

# The loader below mirrors load_numba (chimera_vocal/dsp/backend.py) for the two files involved; it is the test's
# own copy so the control tests need no dsp code. FASTMATH is replaced by the red proof only.
_NUMBA_SCRIPT = r"""
import importlib.util, json, sys
import numpy as np
import numba
pkg, work = sys.argv[1], sys.argv[2]
FASTMATH = False
jit = numba.njit(cache=False, nogil=True, fastmath=FASTMATH, error_model="numpy", boundscheck=False)
mods = {}
for rel, name in (("core/nbmath.py", "_chimera_nbmath"), ("control/geometry_ref.py", "_chimera_geometry_ref")):
    spec = importlib.util.spec_from_file_location(name, pkg + "/" + rel)
    mod = importlib.util.module_from_spec(spec)
    mod.CHIMERA_JIT = jit
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mods[name] = mod
g = mods["_chimera_geometry_ref"]
assert isinstance(g.geometry_compute, numba.core.registry.CPUDispatcher)
assert isinstance(g.geometry_resample, numba.core.registry.CPUDispatcher)
job = json.load(open(work + "/job.json"))
artics = np.load(work + "/artics.npy")
tracts = np.load(work + "/tracts.npy")
nframe = int(job["nframe"])
frames = np.zeros((len(artics), nframe))
res = np.zeros((len(artics), 4 * 18 + 2))
for i, (a, ti) in enumerate(zip(artics, job["tract_of"])):
    f = np.zeros(nframe)
    g.geometry_compute(np.ascontiguousarray(a), np.ascontiguousarray(tracts[ti]), f)
    ph, or_, mph, mor, lens = np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(18), np.zeros(2)
    g.geometry_resample(np.ascontiguousarray(a), np.ascontiguousarray(tracts[ti]), f, ph, or_, mph, mor, lens)
    frames[i] = f
    res[i] = np.concatenate([ph, or_, mph, mor, lens])
np.save(work + "/frames.npy", frames)
np.save(work + "/res.npy", res)
print("ok")
"""


def test_geometry_manifest_and_jit_rules():
    from chimera_vocal.control.jit_manifest import JIT_FILES
    from chimera_vocal.core.jit import JitFile

    assert JIT_FILES == (JitFile("control/geometry_ref.py", "_chimera_geometry_ref",
                                 ("geometry_compute", "geometry_resample")),)
    for f in JIT_FILES:
        assert (PACKAGE_DIR / f.relpath).is_file()
        assert support.check_jit_file(PACKAGE_DIR / f.relpath) == [], f.relpath


def test_frame_layout_is_the_jit_files_layout():
    """frame_layout re-exports the offsets that geometry_ref (a JIT file, which cannot import it) defines."""
    from chimera_vocal.control import frame_layout as fl
    from chimera_vocal.control import geometry_ref as gr

    for name in dir(gr):
        if name.startswith(("F_", "G_", "O_", "N_")) or name in ("NFRAME", "NTRACT", "NG", "MAX_GRID", "MAX_PTS"):
            if hasattr(fl, name):
                assert getattr(fl, name) == getattr(gr, name), name
    assert fl.alloc_frame().shape == (fl.NFRAME,) and fl.alloc_frame().dtype == np.float64
    assert len(fl.GOLDEN_SCALARS) == 16 and fl.F_CONTACT_BODY_NY - fl.F_LENGTH == 15


def _postures(c2):
    """(body, artic) pairs: the 114 golden postures plus N_RANDOM random ones over five bodies."""
    from chimera_vocal.core.body import BODIES

    out = [(BODIES[c["body"]], c2.artic_of(c["artic"])) for c in support.golden("geometry")]
    rng = np.random.default_rng(20261109)
    labels = ("male", "female", "child", "mix-f050", "mix-f050-c050")
    for i in range(N_RANDOM):
        out.append((c2.body_of(labels[i % len(labels)]), c2.random_artic(rng)))
    return out


@pytest.mark.numba
def test_geometry_numba(c2, tmp_path):
    """K-2.3: ref-numba == ref-cpython <= 1e-13 relative on golden + random postures."""
    from chimera_vocal.control import frame_layout as fl
    from chimera_vocal.control.geometry import GEOMETRY_REF_CPYTHON

    posts = _postures(c2)
    bodies, tract_of, tracts = [], [], []
    for body, _ in posts:
        if body not in bodies:
            bodies.append(body)
            tracts.append(c2.tract_for(body))
        tract_of.append(bodies.index(body))
    np.save(tmp_path / "artics.npy", np.array([a for _, a in posts]))
    np.save(tmp_path / "tracts.npy", np.array(tracts))
    (tmp_path / "job.json").write_text(json.dumps({"nframe": fl.NFRAME, "tract_of": tract_of}))
    r = subprocess.run([sys.executable, "-I", "-c", _NUMBA_SCRIPT, str(PACKAGE_DIR), str(tmp_path)],
                       capture_output=True, text=True, timeout=1800, cwd=REPO_ROOT)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-4000:]
    frames, res = np.load(tmp_path / "frames.npy"), np.load(tmp_path / "res.npy")
    failures, worst = [], 0.0
    for i, (body, a) in enumerate(posts):
        want = c2.run_geometry(GEOMETRY_REF_CPYTHON, body, a)
        got = {"scalars": fl.frame_scalars(frames[i]), "grid": fl.frame_grid(frames[i]),
               "outlines": {n: fl.frame_outline(frames[i], n) for n in fl.OUTLINES},
               "resample": {"ph": res[i, 0:18], "or": res[i, 18:36], "phMin": res[i, 36:54], "orMin": res[i, 54:72],
                            "phLen": float(res[i, 72]), "orLen": float(res[i, 73])}}
        diff = c2.compare_geometry(want, got, NUMBA_TOL)
        if diff:
            failures.append(f"posture {i}: {diff[:4]}")
    assert not failures, f"{len(failures)} of {len(posts)} postures differ:\n" + "\n".join(failures[:15])
