"""Integration with a real Hermes (marker ``hermes``; K-6.7, K-6.10).

Needs HERMES_UPSTREAM (a Hermes checkout) and a Hermes interpreter (HERMES_PYTHON, or the venv next to/inside the
checkout). Every Hermes call runs in a subprocess with a fresh HERMES_HOME (fixture ``hermes_lab``).
"""

from __future__ import annotations

import json
import subprocess

import pytest

import c5fakes

pytestmark = pytest.mark.hermes

MEDIA_PROBE = r"""
import json, os, sys, types, importlib, importlib.util
root = sys.argv[1]
ns = types.ModuleType("hermes_plugins"); ns.__path__ = []; sys.modules["hermes_plugins"] = ns
spec = importlib.util.spec_from_file_location("hermes_plugins.chimera", os.path.join(root, "__init__.py"),
                                              submodule_search_locations=[root])
mod = importlib.util.module_from_spec(spec); mod.__path__ = [root]; sys.modules["hermes_plugins.chimera"] = mod
spec.loader.exec_module(mod)
paths = importlib.import_module("hermes_plugins.chimera.chimera_hermes.paths")
media = importlib.import_module("hermes_plugins.chimera.chimera_hermes.media")
from gateway.platforms.base import BasePlatformAdapter, validate_media_delivery_path
from gateway.media_policy import media_delivery_strict
d, stem = paths.new_output_stem("Evening Round!", "score")
wav = os.path.join(str(d), stem + ".wav")
with open(wav, "wb") as f:
    f.write(b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(28))
line = media.media_line(wav)
reply = "Here is your round.\n" + line + "\nEnjoy!"
found, cleaned = BasePlatformAdapter.extract_media(reply)
outside = os.path.join(sys.argv[2], "elsewhere.wav")
with open(outside, "wb") as f:
    f.write(b"RIFF")
print(json.dumps({"wav": wav, "line": line, "found": found, "cleaned": cleaned, "strict": media_delivery_strict(),
                  "valid": validate_media_delivery_path(wav), "outside_valid": validate_media_delivery_path(outside),
                  "audio_root": str(paths.audio_dir())}))
"""


# K-6.7: the MEDIA line is found by extract_media and the path passes the strict delivery check
# (red proof: files stored under /tmp)
def test_media_line(hermes_lab):
    env = dict(hermes_lab.env, HERMES_MEDIA_DELIVERY_STRICT="1", HERMES_MEDIA_TRUST_RECENT_FILES="0")
    r = subprocess.run([str(hermes_lab.python), "-c", MEDIA_PROBE, str(c5fakes.REPO_ROOT), str(hermes_lab.tmp)],
                       env=env, capture_output=True, text=True, timeout=300, cwd=str(hermes_lab.tmp))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["strict"] is True
    assert out["outside_valid"] is None, "strict mode must reject a file outside the Hermes caches (probe sanity)"
    assert out["line"] == "MEDIA:" + out["wav"]
    assert out["found"] == [[out["wav"], False]], out["found"]
    assert out["line"] not in out["cleaned"]
    assert out["valid"] == out["wav"], f"strict media check rejected {out['wav']}"
    assert out["wav"].startswith(str(hermes_lab.home)) and "/cache/audio/chimera/" in out["wav"].replace("\\", "/")


def _validate(hermes_lab, path):
    cmd = ([str(hermes_lab.hermes)] if hermes_lab.hermes else [str(hermes_lab.python), "-m", "hermes_cli.main"])
    r = subprocess.run(cmd + ["plugins", "validate", str(path)], env=hermes_lab.env, capture_output=True, text=True,
                       timeout=600, cwd=str(hermes_lab.tmp))
    return r.returncode, r.stdout + r.stderr


# K-6.10: `hermes plugins validate` passes for the plugin and for accel/ (marker accepted by Hermes' parser),
# no warnings besides the lab's "install out of sync" (red proof: marker with `not (...)`)
def test_validate_cli(hermes_lab):
    for path in (c5fakes.REPO_ROOT, c5fakes.REPO_ROOT / "accel"):
        rc, out = _validate(hermes_lab, path)
        assert rc == 0, out[-4000:]
        assert "Validation passed." in out, out[-4000:]
        assert "python dependencies — 1 requirements from pyproject" in out, out[-4000:]
        assert "register() ran in isolation" in out and "declared tools — matches registrations" in out, out[-4000:]
        lines = [ln for ln in out.splitlines() if ln.startswith(("⚠", "✗"))]
        unexpected = [ln for ln in lines if "install out of sync" not in ln]
        assert unexpected == [], "\n".join(unexpected)


SESSION_PROBE = r"""
import json, os, shutil, sys, threading, time
root, home = sys.argv[1], os.environ["HERMES_HOME"]
dst = os.path.join(home, "plugins", "chimera")
shutil.copytree(root, dst, ignore=shutil.ignore_patterns(".git", "tests", "tools", "accel", "__pycache__", "*.wav"))
with open(os.path.join(home, "config.yaml"), "a", encoding="utf-8") as f:
    f.write("plugins:\n  enabled: [chimera]\n")
from hermes_cli.plugins import get_plugin_manager, get_plugin_command_handler
mgr = get_plugin_manager()
mgr.discover_and_load()
listed = [p for p in mgr.list_plugins() if p.get("name") == "chimera" or p.get("key") == "chimera"]
mod = next(m for n, m in sys.modules.items() if n.startswith("hermes_plugins.chimera") and n.endswith("chimera_hermes.tools"))
pkg = mod.__name__.rsplit(".chimera_hermes.tools", 1)[0]
tools, vocal, sessions = mod, sys.modules[pkg + ".chimera_hermes._vocal"], sys.modules[pkg + ".chimera_hermes.sessions"]
tools._numpy_cache = True       # chimera_job needs no numpy; this Hermes venv has none
from hermes_state import SessionDB
db = SessionDB()
db.create_session("S1", "cli", model="m")
db.publish_compression_child(parent_session_id="S1", child_session_id="S2", source="cli",
                             messages=[{"role": "user", "content": "summary"}], require_compression_lease=False)
for sid in ("U1", "R1", "R2"):
    db.create_session(sid, "cli", model="m")
db.close()
reg = vocal.jobs()

def start(sid):
    def fn(progress, cancel):
        cancel.wait(120)
        if cancel.is_set():
            raise RuntimeError("cancelled")
        return {"success": True, "status": "done"}
    return reg.start("render", tools._owner({"session_id": sid}), fn, eta_s=0).id

j1, j2, jr1, jr2 = start("S1"), start("U1"), start("R1"), start("R2")
from tools.registry import registry
out = {"plugin": listed, "lineage": sessions.compression_lineage("S2")}
call = lambda jid, sid: json.loads(registry.dispatch("chimera_job", {"job_id": jid, "wait_s": 0}, session_id=sid, task_id="t"))
out["after_compression"] = call(j1, "S2").get("status")
out["foreign"] = call(j1, "U1").get("error", "")
from hermes_cli.plugin_command_dispatch import invoke_plugin_command
h = get_plugin_command_handler("chimera")
out["slash_cancel"] = invoke_plugin_command(h, "cancel", source=None, session_id="U1", session_key="U1", platform="cli")
time.sleep(0.2)
out["slash_j2"] = reg.snapshot(j2).status
out["slash_j1"] = reg.snapshot(j1).status
out["slash_legacy"] = h("cancel")
from hermes_cli.lifecycle import finalize_session, invoke_hook
finalize_session(session_id="S2", platform="cli", reason="session_boundary")
invoke_hook("on_session_reset", session_id="R2", platform="tui")
invoke_hook("on_session_reset", session_id="N1", reason="new_session", platform="telegram", old_session_id="R1",
            new_session_id="N1")
time.sleep(0.3)
out["finalize_j1"] = reg.snapshot(j1).status
out["reset_jr1"] = reg.snapshot(jr1).status
out["rebuild_jr2"] = reg.snapshot(jr2).status
reg.cancel_owned(["session_id:R2"])
print(json.dumps(out, default=str))
"""


# Session end and context compression against Hermes itself: the plugin loads through Hermes' plugin manager with
# its two session hooks; a job started in session S1 is reachable through Hermes' tool dispatch from S2 after a
# compression published by Hermes' own SessionDB, not from another session; Hermes' slash-command dispatch passes the
# session id to /chimera cancel (which stops only that session's render); Hermes' finalize_session for S2 stops the
# job started in S1; on_session_reset stops the old_session_id's job and leaves a session that is only rebuilt alone
# (red proof R2A-13/R2A-21: no hooks, binding on session_id only)
def test_session_lifecycle_with_hermes(hermes_lab):
    r = subprocess.run([str(hermes_lab.python), "-c", SESSION_PROBE, str(c5fakes.REPO_ROOT)], env=hermes_lab.env,
                       capture_output=True, text=True, timeout=300, cwd=str(hermes_lab.tmp))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(out["plugin"]) == 1 and out["plugin"][0]["error"] is None, out["plugin"]
    assert out["plugin"][0]["hooks"] == 2 and out["plugin"][0]["commands"] == 1, out["plugin"]
    assert out["lineage"] == ["S1", "S2"]
    assert out["after_compression"] == "running", out
    assert "unknown job" in out["foreign"], out
    assert out["slash_cancel"].startswith("stopping 1 chimera render"), out
    assert (out["slash_j2"], out["slash_j1"]) == ("cancelled", "running"), out
    assert "0.22.6" in out["slash_legacy"] and "chimera_job" in out["slash_legacy"], out
    assert out["finalize_j1"] == "cancelled" and out["reset_jr1"] == "cancelled", out
    assert out["rebuild_jr2"] == "running", out


SCAN_PROBE = r"""
import json, sys
from pathlib import Path
from tools.plugin_guard import scan_plugin
r = scan_plugin(Path(sys.argv[1]))
print(json.dumps({"verdict": r.verdict, "kinds": sorted({f.pattern_id for f in r.findings}),
                  "severities": sorted({f.severity for f in r.findings})}))
"""


# The README's disclosure names every kind of note Hermes' install-time scanner reports for this tree
# (red proof R2A-24: two of ten kinds named)
def test_install_scanner_notes_disclosed(hermes_lab):
    r = subprocess.run([str(hermes_lab.python), "-c", SCAN_PROBE, str(c5fakes.REPO_ROOT)], env=hermes_lab.env,
                       capture_output=True, text=True, timeout=300, cwd=str(hermes_lab.tmp))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["verdict"] == "safe" and out["severities"] == ["medium"], out
    readme = (c5fakes.REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [k for k in out["kinds"] if f"`{k}`" not in readme]
    assert missing == [], f"README does not name these scanner notes: {missing}"
