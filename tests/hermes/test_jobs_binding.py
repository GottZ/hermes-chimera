"""Jobs of the Hermes binding: profile context, caller binding, waiting and cancelling.

* Profile context: a job runs in a copy of the caller's context, so a profile's HERMES_HOME (a context variable in
  Hermes) decides where its audio lands; engine errors are recognised by class name, so a second plugin copy (one per
  profile, sharing the first copy's engine) still maps them. Red proof: plain ``threading.Thread`` and
  ``isinstance`` against the own copy's classes.
* Caller binding: jobs are bound to ``conversation_id`` and ``session_id`` (released Hermes passes only the latter).
  Red proof: binding on ``conversation_id`` only - every session sees every job.
* Waiting: ``wait_s: 0`` returns at once. Red proof: ``requested or default`` (0 counts as "not given").
* Cancelling: ``chimera_job`` with ``cancel: true`` stops a render. Red proof: no way to cancel.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

SCORE = {"title": "Slow", "bpm": 90,
         "voices": [{"name": "lead", "body": "female", "notes": [{"pitch": "C4", "beats": 1, "lyric": "Ah."}]}]}


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


def test_job_runs_in_callers_profile(registered, plugin, tool, tmp_path):
    hc = plugin.hermes.modules.constants
    home_b = tmp_path / "profile-b"
    home_b.mkdir()
    token = hc.set_hermes_home_override(home_b)
    try:
        r = tool(registered, "chimera_speak", {"text": "Profile B speaking.", "wait_s": 30})
        s = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    finally:
        hc.reset_hermes_home_override(token)
    for res in (r, s):
        assert res["success"] is True, res
        assert Path(res["file"]).parent == home_b / "cache" / "audio" / "chimera", res["file"]
    assert not (plugin.hermes.home / "cache" / "audio" / "chimera").exists() or \
        list((plugin.hermes.home / "cache" / "audio" / "chimera").iterdir()) == []


class TextError(ValueError):        # same name as the engine's class, another identity (another plugin copy)
    pass


class ScoreError(ValueError):
    def __init__(self, msg, errors):
        super().__init__(msg)
        self.errors = errors


def test_errors_matched_by_name(registered, plugin, tool):
    eng = plugin.engine
    eng.fail["speak"] = TextError("chimera reads English text; nothing speakable in '...'")
    prov = registered.tts_providers[0]
    with pytest.raises(RuntimeError) as ei:
        prov.synthesize("...", str(plugin.hermes.home / "x.mp3"))
    assert str(ei.value) == "chimera reads English text; nothing speakable in '...'", str(ei.value)
    eng.fail.clear()
    eng.fail["check_score"] = ScoreError("bad", [{"path": "bpm", "severity": "error", "message": "bpm 30..240"}])
    out = tool(registered, "chimera_check", {"score": SCORE})
    assert out["success"] is False and out["issues"] == [{"path": "bpm", "severity": "error",
                                                          "message": "bpm 30..240"}], out


def test_session_binding(registered, plugin, tool):
    plugin.engine.render_delay = 1.0
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1", task_id="t")
    assert r["status"] == "running", r
    jid = r["job_id"]
    other = tool(registered, "chimera_job", {"job_id": jid, "wait_s": 0}, session_id="s2")
    assert other["success"] is False and "unknown job" in other["error"], other
    mine = tool(registered, "chimera_job", {"job_id": jid, "wait_s": 30}, session_id="s1")
    assert mine["status"] == "done", mine
    # a conversation survives a new session id (context compression)
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1", conversation_id="c1")
    same_conv = tool(registered, "chimera_job", {"job_id": r["job_id"], "wait_s": 30}, session_id="s9",
                     conversation_id="c1")
    assert same_conv["status"] == "done", same_conv
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1", conversation_id="c1")
    foreign = tool(registered, "chimera_job", {"job_id": r["job_id"], "wait_s": 0}, session_id="s3",
                   conversation_id="c3")
    assert foreign["success"] is False and "unknown job" in foreign["error"]


def test_wait_zero_returns_at_once(registered, plugin, tool):
    jobs = plugin.mod("jobs")
    assert jobs.wait_budget_s(0) == 0.0 and jobs.wait_budget_s(0.0) == 0.0
    plugin.hermes.config["timeouts"] = {"tools": {"concurrent_batch": 0}}       # unbounded tool time
    assert jobs.wait_budget_s(0) == 0.0 and jobs.wait_budget_s(None) == 1800.0
    plugin.hermes.config.clear()
    plugin.engine.render_delay = 3.0
    t0 = time.monotonic()
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0})
    assert r["status"] == "running" and time.monotonic() - t0 < 1.0, r


def test_cancel_a_running_render(registered, plugin, tool):
    plugin.engine.render_delay = 30.0
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1")
    assert r["status"] == "running"
    foreign = tool(registered, "chimera_job", {"job_id": r["job_id"], "cancel": True}, session_id="s2")
    assert foreign["success"] is False and "unknown job" in foreign["error"]
    t0 = time.monotonic()
    out = tool(registered, "chimera_job", {"job_id": r["job_id"], "cancel": True}, session_id="s1")
    assert out["success"] is False and out["status"] == "cancelled", out
    assert time.monotonic() - t0 < 5.0
    cancel_ev = plugin.engine.calls_of("render_score")[-1]["cancel"]
    assert cancel_ev.is_set()
    bad = tool(registered, "chimera_job", {"job_id": r["job_id"], "cancel": "maybe"}, session_id="s1")
    assert bad["success"] is False and "true or false" in bad["error"]


def test_cancel_in_schema(registered):
    props = registered.tools["chimera_job"]["schema"]["parameters"]["properties"]
    assert props["cancel"]["type"] == "boolean"
