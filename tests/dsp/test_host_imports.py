"""K-3.17 test_host_never_imports_numba (REVIEW-1 P1-11): with numba installed, the host-side calls never import
numba (or llvmlite). C3 part: everything a host does with the acoustics package (import, cache folder, marker
check, numpy render, numpy self-test). S3 part: the Engine API (speak, render_score, status, warmup)."""

from __future__ import annotations

import json

import pytest

import dsp_numba

pytestmark = pytest.mark.numba    # meaningful only where numba is installed

HOST_C3 = r"""
import importlib, pathlib
base = sys.argv[3]
mods = ["chimera_vocal", "chimera_vocal.dsp", "chimera_vocal.dsp.layout", "chimera_vocal.dsp.network",
        "chimera_vocal.dsp.glottis", "chimera_vocal.dsp.decimator", "chimera_vocal.dsp.kernel_numba",
        "chimera_vocal.dsp.kernel_numpy", "chimera_vocal.dsp.jit_manifest", "chimera_vocal.dsp.backend",
        "chimera_vocal.dsp.voice"]
for m in mods:
    importlib.import_module(m)
from chimera_vocal.dsp import backend, voice
from chimera_vocal.dsp.kernel_numpy import KERNEL_NUMPY
from chimera_vocal.core.body import BodyMix
d = dsp_numba.cache_dir_for(base)
pathlib.Path(d).mkdir(parents=True, exist_ok=True)
(pathlib.Path(d) / backend.MARKER_NAME).write_text(json.dumps({"content_hash": "x", "numba": "0.68.1",
    "llvmlite": "0.50.0", "numpy": "2.5.4", "python": "3.14.6", "platform": "x", "cpu": "x",
    "selftest": {"passed": True}, "created": 0}), encoding="utf-8")
seen = {"after_import": "numba" in sys.modules}
backend.numba_cache_dir(base) if dsp_numba.control_manifest_available() else dsp_numba.cache_dir_for(base)
seen["after_cache_dir"] = "numba" in sys.modules
dsp_numba.marker_ok_for(base)
seen["after_marker_ok"] = "numba" in sys.modules
import recorded_control as rc
voice.CONTROL_FACTORY = rc.steady_factory()
v = voice.HumanVoice(48000.0, BodyMix(), 2, kernel=KERNEL_NUMPY)
v.set_gate(1.0)
v.render(640)
seen["after_render"] = "numba" in sys.modules
voice.CONTROL_FACTORY = rc.recorded_factory("sustain-aa-male")
r = backend.selftest(KERNEL_NUMPY, None)
seen["after_selftest"] = "numba" in sys.modules
seen["llvmlite"] = "llvmlite" in sys.modules
seen["selftest_passed"] = bool(r.passed)
print(json.dumps(seen))
"""


def test_host_never_imports_numba_c3(tmp_path):
    import importlib.util
    assert importlib.util.find_spec("numba") is not None
    r = dsp_numba.run_script(HOST_C3, str(tmp_path / "cache"))
    assert r.returncode == 0, r.stderr[-5000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen.pop("selftest_passed") is True
    assert not any(seen.values()), seen


HOST_ENGINE = r"""
import time
import chimera_vocal as cv
eng = cv.get_engine(cv.EngineConfig(cache_dir=sys.argv[3], workers=2))
st = eng.status()
eng.warmup()
audio, kernel = eng.speak("Hello.", "male")
res = eng.render_score({"bpm": 120, "voices": [{"notes": [{"pitch": "A3", "beats": 1, "lyric": "ah"}]}]},
                       sys.argv[3] + "-out", "t", want_mp3=False)
for _ in range(600):
    if eng.status().numba_state in ("ready", "failed"):
        break
    time.sleep(0.5)
st2 = eng.status()
eng.shutdown()
print(json.dumps({"numba": "numba" in sys.modules, "llvmlite": "llvmlite" in sys.modules,
                  "state": st2.numba_state, "kernel": kernel}))
"""


def test_host_never_imports_numba_engine(tmp_path):
    """S3: Engine.speak/render_score/status/warmup leave numba out of the host process."""
    import chimera_vocal
    if not hasattr(chimera_vocal, "get_engine"):
        pytest.skip("S3: needs C4's Engine API")
    r = dsp_numba.run_script(HOST_ENGINE, str(tmp_path / "cache"), timeout=3600)
    assert r.returncode == 0, r.stderr[-5000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen["numba"] is False and seen["llvmlite"] is False, seen
