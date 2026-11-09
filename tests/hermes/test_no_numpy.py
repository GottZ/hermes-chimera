"""K-6.11: the plugin loads and explains itself without numpy (tools hidden, provider unavailable, restart hint)."""

from __future__ import annotations

import json
import subprocess
import sys

import c5fakes

PROBE = r"""
import json, os, sys, tempfile
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "numba"):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import c5fakes, hermes_constants
hermes_constants.HOME = sys.argv[3]
pkg = c5fakes.load_plugin("chimera")
seam = c5fakes.submodule(pkg, "chimera_hermes._vocal")
vocal = c5fakes.make_fake_vocal(c5fakes.FakeEngine(numpy_ok=False))
seam.api = lambda: vocal
ctx = c5fakes.FakeCtx()
pkg.register(ctx)
prov = ctx.tts_providers[0]
out = {"available": prov.is_available(), "available_again": prov.is_available(),
       "check_fns": {n: t["check_fn"]() for n, t in ctx.tools.items()},
       "status": ctx.commands["chimera"]["handler"]("status")}
try:
    prov.synthesize("Hello.", os.path.join(sys.argv[3], "x.mp3"))
    out["synth"] = "no error"
except BaseException as e:
    out["synth"] = [type(e).__name__, str(e)]
try:
    next(iter(prov.stream("Hello.", format="pcm")))
    out["stream"] = "no error"
except BaseException as e:
    out["stream"] = [type(e).__name__, str(e)]
out["tool"] = json.loads(ctx.tools["chimera_speak"]["handler"]({"text": "Hello."}))
out["numpy_loaded"] = any(m.split(".")[0] in ("numpy", "numba") for m in sys.modules)
print(json.dumps(out))
"""


# K-6.11: loads without numpy; is_available() False; check_fn False; status names restart and reinstall
# (red proof: numpy imported at module level)
def test_no_numpy(tmp_path):
    r = subprocess.run([sys.executable, "-I", "-c", PROBE, str(c5fakes.HERE), str(c5fakes.FAKEHERMES_DIR),
                        str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["available"] is False and out["available_again"] is False
    assert set(out["check_fns"]) == {"chimera_render", "chimera_check", "chimera_speak", "chimera_job"}
    assert not any(out["check_fns"].values()), out["check_fns"]
    status = out["status"].lower()
    assert "numpy" in status and "restart hermes" in status and "reinstall" in status, out["status"]
    assert out["synth"][0] == "FileNotFoundError", out["synth"]
    assert "restart hermes" in out["synth"][1].lower() and "reinstall" in out["synth"][1].lower()
    assert out["stream"][0] == "FileNotFoundError", out["stream"]
    assert out["tool"]["success"] is False and "restart" in out["tool"]["error"].lower()
    assert out["numpy_loaded"] is False
    assert not (tmp_path / "x.mp3").exists() and not (tmp_path / "x.wav").exists()
