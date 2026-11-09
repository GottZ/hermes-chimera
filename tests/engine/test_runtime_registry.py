"""K-5.11 test_runtime_registry (TESTPLAN L5, INTERFACES §9.4, P1-25): two copies of the package under different
names in one process (Hermes loads one plugin copy per profile scope) share Engine, scheduler and jobs when their key
(version + code hash) is equal; another key gets its own runtime. get_engine: first config wins; reconfigure
replaces. Runs in a fresh interpreter (the registry is process-global under ``_chimera_runtime``)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from support import REPO_ROOT

SCRIPT = textwrap.dedent(r"""
    import importlib, json, os, re, shutil, sys
    root, tmp = sys.argv[1], sys.argv[2]
    for name in ("plugA", "plugB", "plugC", "plugD"):
        d = os.path.join(tmp, name)
        os.makedirs(d)
        open(os.path.join(d, "__init__.py"), "w").close()
        shutil.copytree(os.path.join(root, "chimera_vocal"), os.path.join(d, "chimera_vocal"),
                        ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(tmp, "plugC", "chimera_vocal", "presets.py"), "a") as f:
        f.write("\n# a different build of the same version\n")
    init_d = os.path.join(tmp, "plugD", "chimera_vocal", "__init__.py")
    src = open(init_d).read()
    open(init_d, "w").write(re.sub(r'__version__ = "[^"]*"', '__version__ = "99.0.0"', src))
    sys.path.insert(0, tmp)
    mods = {n: importlib.import_module(n + ".chimera_vocal.runtime") for n in ("plugA", "plugB", "plugC", "plugD")}
    pk = {n: importlib.import_module(n + ".chimera_vocal") for n in mods}
    e1 = pk["plugA"].get_engine(pk["plugA"].EngineConfig(workers=3))     # creates the runtime: its config wins
    a, b, c, d = (mods[n].runtime() for n in ("plugA", "plugB", "plugC", "plugD"))
    out = {"numpy_after_runtime": "numpy" in sys.modules}
    with a.lock:
        if a.jobs is None:
            a.jobs = {"registry": "jobs of A"}
    out.update({
        "same": a is b, "engine": a.engine is b.engine, "scheduler": a.scheduler is b.scheduler,
        "jobs": b.jobs is a.jobs and b.jobs is not None,
        "c_own": c is not a and c.engine is not a.engine and c.scheduler is not a.scheduler,
        "d_own": d is not a and d is not c and d.engine is not a.engine,
        "keys": {n: m.package_key() for n, m in mods.items()},
        "fixed_module": "_chimera_runtime" in sys.modules,
        "engine_sched": a.engine.scheduler is a.scheduler,
    })
    e2 = pk["plugB"].get_engine(pk["plugB"].EngineConfig(workers=2))
    out["get_engine_shared"] = e1 is e2 is a.engine
    out["first_config_wins"] = e1.status().workers
    e3 = pk["plugB"].reconfigure(pk["plugB"].EngineConfig(workers=2))
    out["reconfigured"] = e3 is not e1 and pk["plugA"].get_engine() is e3 and e3.status().workers == 2
    out["runtime_follows"] = mods["plugA"].runtime().engine is e3
    for m in (e3,):
        m.shutdown()
    print(json.dumps(out))
""")


def test_runtime_fields():
    import dataclasses
    from chimera_vocal.runtime import Runtime
    assert [f.name for f in dataclasses.fields(Runtime)] == ["engine", "scheduler", "lock", "jobs"]
    assert dataclasses.fields(Runtime)[3].default is None


def test_runtime_registry(tmp_path):
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(REPO_ROOT), str(tmp_path)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.splitlines()[-1])
    assert out["numpy_after_runtime"] is False, "creating the runtime imports no numpy"
    assert out["fixed_module"]
    assert out["same"] and out["engine"] and out["scheduler"] and out["jobs"], out
    assert out["engine_sched"], "the engine renders through the registered scheduler"
    keys = out["keys"]
    assert keys["plugA"] == keys["plugB"] != keys["plugC"] != keys["plugD"]
    assert keys["plugA"].startswith("0.1.0-") and keys["plugD"].startswith("99.0.0-")
    assert len(keys["plugA"].split("-", 1)[1]) == 16
    assert out["c_own"] and out["d_own"], out
    assert out["get_engine_shared"] and out["first_config_wins"] == 3
    assert out["reconfigured"] and out["runtime_follows"]
