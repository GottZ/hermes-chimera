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
