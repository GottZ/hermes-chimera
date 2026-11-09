"""Tool handlers and jobs (K-6.6, K-6.8, K-6.9 handler part, K-6.12)."""

from __future__ import annotations

import json
import random
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

import c5fakes

SCORE = {"title": "Test round", "bpm": 90,
         "voices": [{"name": "lead", "body": "female", "notes": [{"pitch": "C4", "beats": 1, "lyric": "Ah."}]}]}
JOB_ID = re.compile(r"chm-[0-9a-f]{32}")


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


# ------------------------------------------------------------------------------------------------ K-6.6

_GENERIC = [None, [], "x", 42, 3.5, True, [{"score": {}}], "", 0]
_BAD_WAIT = ["abc", [], {}, float("nan"), float("inf"), float("-inf"), -1, -0.5, True, False, "10"]
_BAD_SCORE_OBJ = [None, 5, 1.5, True, False, [], [1, 2], ["bpm"], 0]
_BAD_SCORE_STR = ["", "{", "not json", "[1, 2]", "null", "42", '"x"', "{" * 50, "{'bpm': 80}", "\x00", "[" * 5000,
                  "true", "{\"bpm\": 80,}"]
_NON_STR = [5, [], {}, True, 1.5, ["a"]]


def _score_cases(rng):
    full = json.dumps(SCORE)
    return [
        lambda: rng.choice(_GENERIC),
        lambda: {},
        lambda: {"bpm": 90, "voices": SCORE["voices"]},
        lambda: {"score": rng.choice(_BAD_SCORE_OBJ)},
        lambda: {"score": rng.choice(_BAD_SCORE_STR)},
        lambda: {"score": full[: rng.randint(1, len(full) - 1)]},
    ]


def broken_args(tool: str, n: int = 200) -> list:
    """Deterministic invalid argument sets for one tool (each invalid by construction)."""
    rng = random.Random({"chimera_render": 61, "chimera_check": 62, "chimera_speak": 63, "chimera_job": 64}[tool])
    if tool == "chimera_render":
        cases = _score_cases(rng) + [
            lambda: {"score": SCORE, "wait_s": rng.choice(_BAD_WAIT)},
            lambda: {"score": SCORE, "filename": rng.choice(_NON_STR)},
        ]
    elif tool == "chimera_check":
        cases = _score_cases(rng)
    elif tool == "chimera_speak":
        cases = [
            lambda: rng.choice(_GENERIC),
            lambda: {},
            lambda: {"text": rng.choice([None, 5, [], {}, True, "", "   ", "\n\t", 1.5])},
            lambda: {"text": "Hello.", "voice": rng.choice(_NON_STR + ["alloy", "nova", "female;pitch=9999",
                                                                          "male;bogus=1", "Male Deep"])},
            lambda: {"text": "Hello.", "speed": rng.choice(["fast", [], {}, True, float("nan"), float("inf"), -1, 0])},
            lambda: {"text": "Hello.", "wait_s": rng.choice(_BAD_WAIT)},
        ]
    else:
        cases = [
            lambda: rng.choice(_GENERIC),
            lambda: {},
            lambda: {"job_id": rng.choice([None, 5, [], {}, True, "", "   ", 1.5])},
            lambda: {"job_id": rng.choice(["chm-0", "x" * 500, "../../etc/passwd", "chm-" + "0" * 32])},
            lambda: {"job_id": "chm-%032x" % rng.getrandbits(128)},
            lambda: {"job_id": "chm-" + "1" * 32, "wait_s": rng.choice(_BAD_WAIT)},
        ]
    return [rng.choice(cases)() for _ in range(n)]


# K-6.6: 200 broken argument sets per tool -> JSON success:false, never an exception (red proof: unguarded json.loads)
@pytest.mark.parametrize("name", ["chimera_render", "chimera_check", "chimera_speak", "chimera_job"])
def test_tool_handlers_never_raise(registered, plugin, name):
    handler = registered.tools[name]["handler"]
    for i, args in enumerate(broken_args(name)):
        try:
            out = handler(args, conversation_id="conv-fuzz", session_id="s", task_id="t")
        except BaseException as e:  # noqa: BLE001
            pytest.fail(f"case {i} {args!r:.200}: handler raised {type(e).__name__}: {e}")
        assert isinstance(out, str), f"case {i}: {type(out).__name__}"
        data = json.loads(out)
        assert data.get("success") is False, f"case {i} {args!r:.200}: {out[:300]}"
        assert isinstance(data.get("error"), str) and data["error"], f"case {i}: {out[:300]}"
    assert plugin.engine.calls_of("render_score") == [] and plugin.engine.calls_of("speak") == []


ENGINE_ERRORS = [ValueError("bad value"), RuntimeError("worker died"), OSError(28, "No space left on device"),
                 KeyError("x"), MemoryError(), ZeroDivisionError("div")]


# K-6.6 (engine side): any engine exception becomes success:false
@pytest.mark.parametrize("exc", ENGINE_ERRORS + ["ScoreError", "BackendError", "TextError", "RenderCancelled"],
                         ids=lambda e: e if isinstance(e, str) else type(e).__name__)
def test_tool_handlers_engine_failures(registered, plugin, tool, exc):
    if isinstance(exc, str):
        exc = getattr(plugin.vocal, exc)("engine says no")
    eng = plugin.engine
    for method, name, args in [("check_score", "chimera_check", {"score": SCORE}),
                               ("check_score", "chimera_render", {"score": SCORE, "wait_s": 20}),
                               ("render_score", "chimera_render", {"score": SCORE, "wait_s": 20}),
                               ("speak", "chimera_speak", {"text": "Hello.", "wait_s": 20})]:
        eng.fail = {method: exc}
        out = tool(registered, name, args, conversation_id="c")
        assert out["success"] is False and out["error"], (method, name, out)
    eng.fail = {"status": exc}
    out = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 20})
    assert out["success"] is True and out["backend"] == "numpy", out


# ------------------------------------------------------------------------------------------------ K-6.8

# K-6.8: longer than wait_s -> running + 128-bit job_id; chimera_job delivers done; a foreign conversation_id
# gets "unknown job" (red proof: 32-bit ids without owner binding)
def test_job_wait(registered, plugin, tool):
    plugin.engine.render_delay = 2.5
    t0 = time.monotonic()
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0.1}, conversation_id="conv-A")
    assert time.monotonic() - t0 < 2.0
    assert r["success"] is True and r["status"] == "running", r
    assert JOB_ID.fullmatch(r["job_id"]), r["job_id"]
    assert 0.0 <= r["progress"] < 1.0 and isinstance(r["eta_s"], (int, float)) and "chimera_job" in r["instruction"]

    other = tool(registered, "chimera_job", {"job_id": r["job_id"], "wait_s": 0.1}, conversation_id="conv-B")
    assert other["success"] is False and "unknown job" in other["error"]

    done = tool(registered, "chimera_job", {"job_id": r["job_id"], "wait_s": 30}, conversation_id="conv-A")
    assert done["success"] is True and done["status"] == "done" and done["job_id"] == r["job_id"]
    assert done["media"] == "MEDIA:" + done["file"] and Path(done["file"]).is_file()

    anon = tool(registered, "chimera_job", {"job_id": r["job_id"]})  # no session: only both-set owners can differ
    assert anon["status"] == "done"
    unknown = tool(registered, "chimera_job", {"job_id": "chm-" + "0" * 32}, conversation_id="conv-A")
    assert unknown["success"] is False and "unknown job" in unknown["error"]

    ids = set()
    plugin.engine.render_delay = 0.0
    for _ in range(20):
        ids.add(tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30}, conversation_id="c")["job_id"])
    assert len(ids) == 20 and all(JOB_ID.fullmatch(i) for i in ids)


def test_job_registry(plugin):
    jobs = plugin.mod("jobs")
    now = [1000.0]
    reg = jobs.JobRegistry(retention_s=7200.0, clock=lambda: now[0])
    gate = threading.Event()

    def work(progress, cancel):
        progress(0.5)
        gate.wait(10)
        return {"success": True, "status": "done", "value": 1}

    st = reg.start("render", "owner-1", work, eta_s=12.0)
    assert JOB_ID.fullmatch(st.id) and st.owner == "owner-1" and st.kind == "render"
    assert st.status in ("queued", "running")
    mid = reg.wait(st.id, "owner-1", 0.2)
    assert mid.status == "running" and mid.progress == pytest.approx(0.5)
    with pytest.raises(KeyError):
        reg.wait(st.id, "owner-2", 0)
    with pytest.raises(KeyError):
        reg.wait("chm-" + "f" * 32, None, 0)
    gate.set()
    end = reg.wait(st.id, None, 10)
    assert end.status == "done" and end.result == {"success": True, "status": "done", "value": 1}
    assert end.finished is not None and end.progress == 1.0

    def boom(progress, cancel):
        raise RuntimeError("render exploded")

    bad = reg.wait(reg.start("speak", None, boom, eta_s=1.0).id, "anyone", 10)
    assert bad.status == "failed" and "render exploded" in bad.error and bad.result is None

    now[0] += 7201.0
    with pytest.raises(KeyError):
        reg.wait(st.id, None, 0)


# K-6.9 (handler part): a deadline below 60 s returns `running` at once
def test_short_deadline_returns_running(registered, plugin, tool):
    plugin.hermes.config["timeouts"] = {"tools": {"concurrent_batch": 50}}
    plugin.engine.render_delay = 0.6
    t0 = time.monotonic()
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30}, conversation_id="c")
    assert time.monotonic() - t0 < 0.5
    assert r["status"] == "running" and JOB_ID.fullmatch(r["job_id"])
    plugin.hermes.config.clear()
    done = tool(registered, "chimera_job", {"job_id": r["job_id"], "wait_s": 30}, conversation_id="c")
    assert done["status"] == "done"


# ------------------------------------------------------------------------------------------------ results

def test_render_done(registered, plugin, tool):
    eng = plugin.engine
    r = tool(registered, "chimera_render", {"score": {**SCORE, "title": "Evening Round!"}, "wait_s": 30},
             conversation_id="c1")
    assert r["success"] is True and r["status"] == "done", r
    assert r["format"] == "wav" and r["file"] == r["wav"] and r["media"] == "MEDIA:" + r["file"]
    p = Path(r["file"])
    assert p.is_file() and p.parent == plugin.hermes.home / "cache" / "audio" / "chimera"
    assert re.fullmatch(r"\d{8}-\d{6}-evening-round\.wav", p.name), p.name
    assert c5fakes.wav_info(p)["magic"] == b"RIFF"
    assert (r["backend"], r["sample_rate"], r["voices"], r["channels"]) == ("numpy", 48000, 1, 1)
    assert r["instruction"] and isinstance(r["warnings"], list) and JOB_ID.fullmatch(r["job_id"])
    call = eng.calls_of("render_score")[-1]
    assert call["out_dir"] == str(p.parent) and call["stem"] == p.stem
    assert isinstance(call["cancel"], threading.Event)

    r2 = tool(registered, "chimera_render", {"score": SCORE, "filename": "My Song", "wait_s": 30})
    assert Path(r2["file"]).name.endswith("-my-song.wav")
    r3 = tool(registered, "chimera_render", {"score": json.dumps(SCORE), "wait_s": 30})
    assert Path(r3["file"]).name.endswith("-test-round.wav") and r3["file"] != r["file"]

    n = len(eng.calls_of("render_score"))
    bad = tool(registered, "chimera_render", {"score": {"bpm": 999, "voices": []}, "wait_s": 30})
    assert bad["success"] is False and {i["path"] for i in bad["issues"]} == {"bpm", "voices"}
    assert all(set(i) >= {"path", "severity", "message"} for i in bad["issues"])
    assert len(eng.calls_of("render_score")) == n, "an invalid score must not be rendered"

    eng.numba_installed, eng.numba_state = True, "ready"
    r4 = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r4["backend"] == "numba"


def test_check_response(registered, plugin, tool):
    eng = plugin.engine
    r = tool(registered, "chimera_check", {"score": SCORE})
    assert r["success"] is True and r["ok"] is True and r["issues"] == []
    assert r["estimate_s"] == {"numba": 1.5, "numpy": 15.0} and r["backend"] == "numpy"
    assert r["fits_one_call"] is True
    for k in ("voices", "seconds", "voice_seconds", "lead_in", "memory_mb", "disk_mb"):
        assert k in r, k
    assert r["voices"][0]["name"] == "lead"
    bad = tool(registered, "chimera_check", {"score": {"voices": []}})
    assert bad["success"] is True and bad["ok"] is False
    assert {i["path"] for i in bad["issues"]} == {"bpm", "voices"}
    assert tool(registered, "chimera_check", {"score": json.dumps(SCORE)})["ok"] is True
    eng.estimate_s = {"numba": 1.5, "numpy": 500.0}
    assert tool(registered, "chimera_check", {"score": SCORE})["fits_one_call"] is False
    eng.numba_installed, eng.numba_state = True, "ready"
    r = tool(registered, "chimera_check", {"score": SCORE})
    assert r["backend"] == "numba" and r["fits_one_call"] is True
    assert eng.calls_of("render_score") == []


def test_speak_tool(registered, plugin, tool):
    eng = plugin.engine
    r = tool(registered, "chimera_speak", {"text": "Hello there.", "voice": "female;pitch=230", "speed": 1.2,
                                           "wait_s": 30}, conversation_id="c")
    assert r["success"] is True and r["status"] == "done", r
    p = Path(r["file"])
    assert r["media"] == "MEDIA:" + r["file"] and p.parent == plugin.hermes.home / "cache" / "audio" / "chimera"
    info = c5fakes.wav_info(r["wav"])
    assert (info["magic"], info["channels"], info["bits"], info["rate"]) == (b"RIFF", 1, 16, 48000)
    call = eng.calls_of("speak")[-1]
    assert c5fakes.FakeEngine.voice_id(call["voice"]) == "female" and call["speed"] == pytest.approx(1.2)
    registered.settings["voice"] = "narrator"
    tool(registered, "chimera_speak", {"text": "Default voice.", "wait_s": 30})
    call = eng.calls_of("speak")[-1]
    assert c5fakes.FakeEngine.voice_id(call["voice"]) == "narrator" and call["speed"] == pytest.approx(0.92)
    eng.fail["speak"] = plugin.vocal.TextError("chimera reads English text; nothing speakable was left after cleanup")
    out = tool(registered, "chimera_speak", {"text": "日本語", "wait_s": 30})
    assert out["success"] is False and "English" in out["error"]


# ------------------------------------------------------------------------------------------------ K-6.12

# K-6.12: compile error / worker crash (SIGSEGV via test hook) during warm-up -> state failed, renders with numpy,
# the Hermes process lives (red proof: exception passed through)
def test_numba_failure_fallback(registered, plugin, tool):
    eng = plugin.engine
    eng.numba_installed, eng.numba_version, eng.numba_state = True, "0.68.1", "cold"
    eng.warm_crash = True
    slash = registered.commands["chimera"]["handler"]
    out = slash("warmup")
    assert isinstance(out, str) and "fail" in out.lower(), out
    assert eng.numba_state == "failed"
    status = slash("status")
    assert "failed" in status and "numpy" in status and "returncode" in status, status
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["success"] is True and r["backend"] == "numpy"
    s = tool(registered, "chimera_speak", {"text": "Still speaking.", "wait_s": 30})
    assert s["success"] is True and s["backend"] == "numpy"

    eng.warm_crash = False
    eng.fail["warmup"] = plugin.vocal.BackendError("numba compile error: TypingError in run_samples")
    out = slash("warmup")
    assert isinstance(out, str) and "compile error" in out
    registered.tts_providers[0].warm()
    eng.fail["selftest"] = plugin.vocal.BackendError("self-test worker crashed")
    out = slash("test")
    assert isinstance(out, str) and "crashed" in out


# Time-limit warning: estimate >= 50 % of Hermes' tool limit -> warning in chimera_check and in the render result
def test_deadline_warning(registered, plugin, tool):
    eng = plugin.engine
    r = tool(registered, "chimera_check", {"score": SCORE})
    assert r["warnings"] == []
    eng.estimate_s = {"numba": 2.0, "numpy": 210.0}
    r = tool(registered, "chimera_check", {"score": SCORE})
    assert len(r["warnings"]) == 1 and "420" in r["warnings"][0] and "chimera-numba" in r["warnings"][0]
    done = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert any("time limit" in w for w in done["warnings"]), done["warnings"]
    eng.estimate_s = {"numba": 2.0, "numpy": 209.0}
    assert tool(registered, "chimera_check", {"score": SCORE})["warnings"] == []
    plugin.hermes.config["timeouts"] = {"tools": {"concurrent_batch": 0}}
    eng.estimate_s = {"numba": 2.0, "numpy": 5000.0}
    assert tool(registered, "chimera_check", {"score": SCORE})["warnings"] == []
    plugin.hermes.config["timeouts"] = {"tools": {"concurrent_batch": 100}}
    eng.numba_installed, eng.numba_state = True, "ready"
    eng.estimate_s = {"numba": 60.0, "numpy": 5000.0}
    w = tool(registered, "chimera_check", {"score": SCORE})["warnings"]
    assert len(w) == 1 and "numba backend" in w[0] and "chimera-numba" not in w[0]
