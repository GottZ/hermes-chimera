"""K-3.13 test_numba_cache_scoping (INTERFACES §6.4a): (a) all entry points with their real argument types in two
worker processes whose parent package names differ: cache hits, no ModuleNotFoundError; (b) a changed function in
nbmath.py gives a new cache folder and a new result; (c) numba.config.CACHE_DIR and os.environ unchanged after
loading; (d) an unwritable cache gives cache=False and a correct render; (e) AST rules of every JIT file."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

import dsp_numba
import support
from chimera_vocal.dsp import backend
from support import PACKAGE_DIR, REPO_ROOT, TESTS_DIR, check_jit_file

ENTRY = ("run_samples", "output_chain", "flush_subnormals")

# Worker without the shared prelude: imports the plugin ONLY as <parent>.chimera_vocal (argv: tmp, parent, base, tests)
PARENT_WORKER = r"""
import collections, importlib, importlib.util, json, sys
tmp, parent, base, tests = sys.argv[1:5]
sys.path[:0] = [tmp, tests]
import support
pkg = parent + ".chimera_vocal"
backend = importlib.import_module(pkg + ".dsp.backend")
network = importlib.import_module(pkg + ".dsp.network")
voice = importlib.import_module(pkg + ".dsp.voice")
L = importlib.import_module(pkg + ".dsp.layout")
core_jit = importlib.import_module(pkg + ".core.jit")
dsp_jit = importlib.import_module(pkg + ".dsp.jit_manifest")
types = importlib.import_module(pkg + ".core.types")
body = importlib.import_module(pkg + ".core.body")
import numpy as np
if importlib.util.find_spec(pkg + ".control.jit_manifest") is None:
    backend.all_jit_files = lambda: core_jit.CORE_JIT_FILES + dsp_jit.JIT_FILES
files = backend.all_jit_files()
mods = backend.load_jit_modules(files, backend.numba_cache_dir(base))
k = backend.kernel_from_module(mods["_chimera_kernel_numba"])
d = support.vector("track-sustain-aa-male")
TC = collections.namedtuple("TC", "n_blocks tube_area tube_min tube_len velum_area noise_n noise_idx noise_par "
                                  "noise_inherit glot_par p_lung")
def rows(src, idx_list, inh_list, n):
    nn = np.array([len(x) for x in idx_list], dtype=np.int64)
    ni = np.full((n, 3, 3), -1, dtype=np.int64); nh = np.full((n, 3), -1, dtype=np.int64)
    npar = np.zeros((n, 3, 3))
    for b in range(n):
        for j, t in enumerate(idx_list[b]): ni[b, j] = t
        for j, t in enumerate(inh_list[b]): nh[b, j] = t
        for j, t in enumerate(src["noise_par"][b]): npar[b, j] = t
    a = lambda key: np.ascontiguousarray(np.array(src[key], dtype=np.float64)[:n])
    return TC(n, a("tube_area"), a("tube_min"), a("tube_len"), a("velum_area"), nn, ni, npar, nh, a("glot_par"),
              a("p_lung"))
ini = d["initial"]
init = rows({k2: [ini[k2]] for k2 in ("tube_area", "tube_min", "tube_len", "velum_area", "noise_par", "glot_par",
                                       "p_lung")}, [ini["noise_idx"]], [ini["noise_inherit"]], 1)
idx0, inh0 = d["int_runs"][0][1], d["int_runs"][0][2]
blocks = rows(d["float"], [idx0] * 20, [inh0] * 20, 20)
st, ix = network.human_net(body.BODY_MALE, 96000.0, types.Seeds())
network.configure_output(st, 48000.0, 2)
network.apply_initial(st, ix, init)
raw = np.zeros(128 * 20); out = np.zeros(64 * 20, dtype=np.float32)
voice.run(k, st, ix, blocks, 128 * 20, raw)
n = k.output_chain(st.sc, st.si, st.dec_c, st.dec_buf, raw, 128 * 20, out)
k.flush_subnormals(st.nf, st.bf, st.rf, st.noise_f, st.glot, st.sc)
m = sys.modules["_chimera_kernel_numba"]
stats = {e: {"hits": sum(getattr(m, e).stats.cache_hits.values()),
             "misses": sum(getattr(m, e).stats.cache_misses.values())} for e in %(ENTRY)r}
print(json.dumps({"n": int(n), "checksum": float(np.abs(out.astype(np.float64)).sum()), "stats": stats,
                  "kernel_module": m.__name__, "parent_loaded": pkg in sys.modules}))
""" % {"ENTRY": ENTRY}


def _parent_run(tmp, parent, base):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-I", "-c", PARENT_WORKER, str(tmp), parent, str(base), str(TESTS_DIR)],
                       capture_output=True, text=True, timeout=1800, cwd=str(tmp))
    assert r.returncode == 0, f"{parent}: rc {r.returncode}\n{r.stderr[-5000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.numba
def test_cache_scoping_parent_names(tmp_path):
    """(a) worker 1 as parent_a.chimera_vocal compiles into the cache; worker 2 as parent_b.chimera_vocal loads every
    entry point from it (hits, no misses, no ModuleNotFoundError) and renders the same samples."""
    for parent in ("parent_a", "parent_b"):            # each process sees only its own parent (as two Hermes
        (tmp_path / parent).mkdir()                      # scopes of different processes would)
        os.symlink(REPO_ROOT, tmp_path / parent / parent, target_is_directory=True)
    base = tmp_path / "cache"
    a = _parent_run(tmp_path / "parent_a", "parent_a", base)
    b = _parent_run(tmp_path / "parent_b", "parent_b", base)
    assert a["kernel_module"] == b["kernel_module"] == "_chimera_kernel_numba"
    assert a["n"] == b["n"] == 64 * 20 and a["checksum"] == b["checksum"]
    for e in ENTRY:
        assert a["stats"][e]["misses"] >= 1, (e, a["stats"][e])
        assert b["stats"][e]["misses"] == 0 and b["stats"][e]["hits"] >= 1, (e, b["stats"][e])


CHANGED_WORKER = r"""
import json, sys
root, tests, base = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [root, tests, tests + "/dsp"]
import dsp_numba, dsp_cases
dsp_numba.install_standalone_overrides()
from chimera_vocal.dsp import backend
dsp_cases._LOADED.update(dsp_numba.load_for_tests(base))
res = dsp_cases.case_track_audio("numba", "sustain-aa-male", 4800)
print(json.dumps({"cache_dir": backend.numba_cache_dir(base), "audio": res["audio"].astype(float).tolist()[:4800],
                  "stats": dsp_numba.dispatch_stats(), "file": dsp_cases.__file__,
                  "nbmath": sys.modules["_chimera_nbmath"].__file__}))
"""


@pytest.mark.numba
def test_cache_changed_callee(tmp_path):
    """(b) Changing a function in nbmath.py (a callee in ANOTHER file than the kernel) changes the cache folder, so
    the kernel is recompiled and the result changes (E1-B §2.6: same folder would return stale code)."""
    import subprocess
    import sys

    roots = {}
    for name in ("orig", "changed"):
        root = tmp_path / name
        shutil.copytree(PACKAGE_DIR, root / "chimera_vocal", ignore=shutil.ignore_patterns("__pycache__"))
        roots[name] = root
    nb = roots["changed"] / "chimera_vocal" / "core" / "nbmath.py"
    src = nb.read_text(encoding="utf-8")
    assert src.count("* 1.7320508") == 1
    nb.write_text(src.replace("* 1.7320508", "* 1.7320509"), encoding="utf-8")
    base = tmp_path / "cache"
    out = {}
    for name, root in roots.items():
        r = subprocess.run([sys.executable, "-I", "-c", CHANGED_WORKER, str(root), str(TESTS_DIR), str(base)],
                           capture_output=True, text=True, timeout=1800, cwd=str(root))
        assert r.returncode == 0, f"{name}: {r.stderr[-5000:]}"
        out[name] = json.loads(r.stdout.strip().splitlines()[-1])
        assert out[name]["nbmath"].startswith(str(root)), out[name]["nbmath"]
    assert out["orig"]["cache_dir"] != out["changed"]["cache_dir"]
    assert out["orig"]["cache_dir"].startswith(str(base)) and out["changed"]["cache_dir"].startswith(str(base))
    assert out["changed"]["stats"]["_chimera_kernel_numba.run_samples"]["misses"] >= 1
    assert out["orig"]["audio"] != out["changed"]["audio"], "changed jitter constant had no effect (stale cache?)"


SCOPE_WORKER = r"""
import json, os
import numba
before = (numba.config.CACHE_DIR, dict(os.environ))
import dsp_cases
from chimera_vocal.dsp import backend
mods = backend.load_jit_modules(backend.all_jit_files(), backend.numba_cache_dir(sys.argv[3]))
after_load = (numba.config.CACHE_DIR, dict(os.environ))
dsp_cases._LOADED["numba"] = backend.kernel_from_module(mods["_chimera_kernel_numba"])
dsp_cases.case_track_audio("numba", "sustain-aa-male", 64)          # compiles (and caches) the entry points
after = (numba.config.CACHE_DIR, dict(os.environ))
print(json.dumps({"same_cache_dir": before[0] == after_load[0] == after[0],
                  "same_env": before[1] == after_load[1] == after[1],
                  "cache_dir_after": str(after[0]), "info": backend.load_info()}))
"""


@pytest.mark.numba
def test_cache_config_untouched(tmp_path):
    """(c) numba.config.CACHE_DIR and os.environ are the same before and after loading."""
    r = dsp_numba.run_script(SCOPE_WORKER, str(tmp_path / "cache"))
    assert r.returncode == 0, r.stderr[-5000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["same_cache_dir"] and res["same_env"], res
    assert res["info"]["cached"] is True and res["info"]["cache_dir"].startswith(str(tmp_path / "cache"))
    assert any((tmp_path / "cache").rglob("*.nbi")), "nothing written to the cache folder"


@pytest.mark.numba
def test_cache_unwritable(tmp_path, run):
    """(d) A cache base that cannot be created (a regular file in the way): cache=False, the render is correct."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    script = r"""
import dsp_cases
from chimera_vocal.dsp import backend
dsp_cases._LOADED.update(dsp_numba.load_for_tests(sys.argv[3]))
res = dsp_cases.case_track_audio("numba", "sustain-aa-male", 4800)
print(json.dumps({"info": backend.load_info(), "audio": res["audio"].astype(float).tolist()}))
"""
    r = dsp_numba.run_script(script, str(blocker))
    assert r.returncode == 0, r.stderr[-5000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["info"]["cached"] is False, res["info"]
    assert res["info"]["reason"]
    ref = run("numpy", "track_audio", scenario="sustain-aa-male", n_out=4800)
    support.assert_t3(ref["audio"], np.array(res["audio"], dtype=np.float32), "uncached numba vs numpy")
    assert blocker.is_file() and blocker.read_text(encoding="utf-8") == "x"


def test_jit_files_follow_rules():
    """(e) every JIT file: only math/numpy/_chimera_nbmath imports (+ the ..core.nbmath fallback), the jit line
    first, @jit on every function, no classes."""
    files = [PACKAGE_DIR / "dsp" / "kernel_numba.py", PACKAGE_DIR / "core" / "nbmath.py"]
    geo = PACKAGE_DIR / "control" / "geometry_ref.py"
    if geo.is_file():
        files.append(geo)
    for f in files:
        assert check_jit_file(f) == [], (f, check_jit_file(f))


def test_cache_dir_is_content_addressed(tmp_path):
    """numba_cache_dir: <base>/<sha256 of all JIT files in order>[:16]; no numba import, nothing created."""
    import hashlib
    import sys
    files = dsp_numba.standalone_jit_files()
    h = hashlib.sha256()
    for f in files:
        h.update((PACKAGE_DIR / f.relpath).read_bytes())
    want = os.path.join(str(tmp_path), h.hexdigest()[:16])
    got = dsp_numba.cache_dir_for(str(tmp_path))
    assert got == want
    assert not os.path.exists(want)
    assert "numba" not in sys.modules
