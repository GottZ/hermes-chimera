"""Jobs of the Hermes binding: profile context, caller binding, waiting and cancelling.

* Profile context: a job runs in a copy of the caller's context, so a profile's HERMES_HOME (a context variable in
  Hermes) decides where its audio lands; engine errors are recognised by class name, so a second plugin copy (one per
  profile, sharing the first copy's engine) still maps them. Red proof: plain ``threading.Thread`` and
  ``isinstance`` against the own copy's classes.
* Caller binding: jobs are bound to ``conversation_id`` and ``session_id`` (released Hermes passes only the latter).
  Red proof: binding on ``conversation_id`` only - every session sees every job.
* Waiting: ``wait_s: 0`` returns at once. Red proof: ``requested or default`` (0 counts as "not given").
* Cancelling: ``chimera_job`` with ``cancel: true`` stops a render. Red proof: no way to cancel.
* Context compression: released Hermes passes only ``session_id``, and compression continues the conversation in a
  new session; a job stays reachable from the sessions of its compression chain (Hermes' session store), not from
  any other session. Red proof (R2A-21): "unknown job" after compression.
* Session end: ``on_session_finalize`` (and ``on_session_reset`` with ``old_session_id``) stops the session's
  unfinished renders, also those started before a compression; other sessions' renders keep running.
  ``/chimera cancel`` stops the caller's own renders from Hermes 0.22.6 on (the session id reaches the handler
  as a keyword) and explains the ``chimera_job`` way on older Hermes. Red proof (R2A-13): renders kept running.
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


def _status(registered, tool, jid, sid, wait_s=10.0):
    return tool(registered, "chimera_job", {"job_id": jid, "wait_s": wait_s}, session_id=sid)


def test_job_survives_context_compression(registered, plugin, tool):
    state = plugin.hermes.modules.state
    plugin.engine.render_delay = 0.5
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1", task_id="t")
    jid = r["job_id"]
    assert _status(registered, tool, jid, "s1", 0)["status"] in ("running", "done")
    assert state.OPENED == [], "a caller matching the job directly must not read the session store"
    # compression: s1 ends with reason "compression", s2 continues it, s3 continues s2; u1 is another conversation
    # and n1 a child of a session that did not end by compression
    state.SESSIONS.update({"s1": (None, "compression"), "s2": ("s1", "compression"), "s3": ("s2", None),
                           "u1": (None, None), "n1": ("u1", None)})
    after = _status(registered, tool, jid, "s3", 30)
    assert after["status"] == "done", after
    assert state.OPENED and all(state.OPENED), "the session store is opened read-only"
    for sid in ("u1", "n1", "zz"):
        foreign = _status(registered, tool, jid, sid, 0)
        assert foreign["success"] is False and "unknown job" in foreign["error"], (sid, foreign)
        cancel = tool(registered, "chimera_job", {"job_id": jid, "cancel": True}, session_id=sid)
        assert "unknown job" in cancel["error"], (sid, cancel)
    opened = len(state.OPENED)
    tool(registered, "chimera_job", {"job_id": jid, "cancel": True}, session_id="s3")
    assert len(state.OPENED) == opened + 1, "one tool call reads the session store at most once"


def test_session_end_stops_its_renders(registered, plugin, tool):
    state = plugin.hermes.modules.state
    plugin.engine.render_delay = 30.0
    start = lambda sid: tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id=sid)["job_id"]
    mine, other, early, rebuilt = start("s1"), start("x1"), start("c1"), start("r1")
    registered.fire("on_session_finalize", session_id="s1", platform="cli", reason="session_boundary")
    assert _status(registered, tool, mine, "s1")["status"] == "cancelled"
    assert _status(registered, tool, other, "x1", 0)["status"] == "running"
    # finalize names the session the conversation is in now; a render from before a compression ends too
    state.SESSIONS.update({"c1": (None, "compression"), "c2": ("c1", None)})
    registered.fire("on_session_finalize", session_id="c2", platform="telegram", reason="new_session",
                    old_session_id="c2", new_session_id="c9")
    assert _status(registered, tool, early, "c2")["status"] == "cancelled"
    # on_session_reset: session_id is the session that starts (or is rebuilt) - left alone; old_session_id ended
    registered.fire("on_session_reset", session_id="r1", platform="tui")
    assert _status(registered, tool, rebuilt, "r1", 0)["status"] == "running"
    registered.fire("on_session_reset", session_id="n1", reason="new_session", platform="telegram",
                    old_session_id="r1", new_session_id="n1")
    assert _status(registered, tool, rebuilt, "r1")["status"] == "cancelled"
    assert _status(registered, tool, other, "x1", 0)["status"] == "running"
    registered.fire("on_session_finalize", session_id=None)          # nothing to do, no error
    tool(registered, "chimera_job", {"job_id": other, "cancel": True}, session_id="x1")


def test_session_hooks_without_any_render(registered, plugin):
    registered.fire("on_session_finalize", session_id="s1")
    registered.fire("on_session_reset", session_id="s2", old_session_id="s1")
    assert plugin.engine.calls == [] and plugin.vocal.rt.jobs is None, "a session end must not start the engine"
    assert plugin.hermes.modules.state.OPENED == []


def test_slash_cancel_own_session(registered, plugin, tool):
    import inspect

    slash = registered.commands["chimera"]["handler"]
    plugin.engine.render_delay = 30.0
    a = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s1")["job_id"]
    b = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 0}, session_id="s2")["job_id"]
    # Hermes before 0.22.6 calls handler(raw_args): no session, so no cancel - the message names the other way
    old = slash("cancel")
    assert "0.22.6" in old and "chimera_job" in old and "cancel: true" in old, old
    assert _status(registered, tool, a, "s1", 0)["status"] == "running"
    # Hermes 0.22.6+ passes session_id to a handler that names it as a keyword parameter
    assert inspect.signature(slash).parameters["session_id"].kind is inspect.Parameter.KEYWORD_ONLY
    out = slash("cancel", session_id="s1")
    assert out.startswith("stopping 1 chimera render") and a in out, out
    assert _status(registered, tool, a, "s1")["status"] == "cancelled"
    assert _status(registered, tool, b, "s2", 0)["status"] == "running"
    assert "unknown job" in slash(f"cancel {b}", session_id="s1")
    assert _status(registered, tool, b, "s2", 0)["status"] == "running"
    assert slash(f"cancel {b}", session_id="s2") == f"stopping chimera render {b}"
    assert _status(registered, tool, b, "s2")["status"] == "cancelled"
    assert slash("cancel", session_id="s1") == "no chimera render is running in this conversation"
    assert slash("cancel", session_id="").startswith("no session here")
