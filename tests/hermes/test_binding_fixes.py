"""Smaller fixes of the Hermes binding: live settings, credentials in the install id, provider output next to user
files, the CLI warm-up wait after a failed self-test, and the faithful speech switch (binding part).

Red proofs (9a616be): live settings - the first config wins (no ``apply_settings`` call); install id - only http(s)
sources cleaned; provider output - ``<stem>.wav`` opened with "wb"; CLI warm-up - the CLI waits up to 900 s for
ready|failed while the state stays cold; faithful speech - no switch exists.
"""

from __future__ import annotations

import argparse
import shutil
import threading

import pytest

import c5fakes

SCORE = {"bpm": 90, "voices": [{"body": "female", "notes": [{"pitch": "C4", "beats": 1, "lyric": "Ah."}]}]}


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


# --- live settings --------------------------------------------------------------------------------------------

def test_changed_settings_reach_the_engine(registered, plugin, tool):
    registered.settings.update(backend="auto", workers=2, text_prep=True, max_voices=8)
    tool(registered, "chimera_check", {"score": SCORE})
    first = plugin.engine.calls_of("apply_settings")[-1]["config"]
    assert (first.backend, first.workers, first.text_prep, first.max_voices) == ("auto", 2, True, 8)
    registered.settings.update(backend="numpy", workers=3, text_prep=False, max_voices=2)
    tool(registered, "chimera_check", {"score": SCORE})
    last = plugin.engine.calls_of("apply_settings")[-1]["config"]
    assert (last.backend, last.workers, last.text_prep, last.max_voices) == ("numpy", 3, False, 2)


# --- credentials in the install id ----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,shown", [
    ("ssh://git:SECRET@example.org/o/r.git", "ssh://example.org/o/r.git"),
    ("git+https://user:SECRET@github.com/o/r", "o/r"),
    ("git+https://user:SECRET@gitlab.example/o/r.git?ref=x#accel", "git+https://gitlab.example/o/r.git"),
    ("https://token@github.com/o/r.git#accel", "o/r"),
    ("ssh://SECRET@[::1]:2222/o/r", "ssh://[::1]:2222/o/r"),
    ("me:SECRET@host.example:o/r.git", "host.example:o/r.git"),
    ("git@github.com:o/r.git", "git@github.com:o/r.git"),
    ("file:///srv/plugins/chimera", "file:///srv/plugins/chimera"),
    ("GottZ/hermes-chimera", "GottZ/hermes-chimera"),
])
def test_install_source_has_no_credentials(plugin, raw, shown):
    clean = plugin.mod("commands")._clean_source(raw)
    assert "SECRET" not in clean and "token" not in clean, clean
    assert clean == shown


# --- provider output next to user files -----------------------------------------------------------------------

def test_provider_never_overwrites_a_users_wav(registered, plugin, tmp_path):
    prov = registered.tts_providers[0]
    mine = tmp_path / "story.wav"
    mine.write_bytes(b"my own recording")
    got = prov.synthesize("Once upon a time.", str(tmp_path / "story.mp3"))
    assert mine.read_bytes() == b"my own recording"
    assert got != str(mine) and got.startswith(str(tmp_path / "story.")) and got.endswith(".wav")
    assert c5fakes.wav_info(got)["magic"] == b"RIFF"
    # Hermes' own .wav / .ogg path is used as given
    assert prov.synthesize("Hi.", str(tmp_path / "chunk.wav")) == str(tmp_path / "chunk.wav")
    # a failed synthesis leaves no reserved empty file
    plugin.engine.fail["speak"] = RuntimeError("worker died")
    with pytest.raises(RuntimeError):
        prov.synthesize("Nope.", str(tmp_path / "fail.mp3"))
    assert not (tmp_path / "fail.wav").exists()


# --- CLI warm-up wait -----------------------------------------------------------------------------------------

def _cli(registered, argv):
    parser = argparse.ArgumentParser(prog="hermes chimera")
    registered.cli_commands["chimera"]["setup_fn"](parser)
    return registered.cli_commands["chimera"]["handler_fn"](parser.parse_args(argv))


@pytest.mark.parametrize("state", ["cold", "failed", "ready"])
def test_cli_warmup_returns_when_not_warming(registered, plugin, state):
    eng = plugin.engine
    eng.numba_installed, eng.numba_version, eng.numba_state = True, "0.68.1", state
    eng.numba_error = "selftest failed: 9.9e-03" if state != "ready" else None
    rc = {}
    t = threading.Thread(target=lambda: rc.setdefault("rc", _cli(registered, ["warmup"])), daemon=True)
    t.start()
    t.join(5.0)
    assert not t.is_alive(), f"`hermes chimera warmup` still waiting with state {state}"
    assert rc["rc"] == (1 if state == "failed" else 0)


def test_status_shows_reason_of_a_cold_state(registered, plugin):
    eng = plugin.engine
    eng.numba_installed, eng.numba_state = True, "cold"
    eng.numba_error = "warm-up not finished: timed out waiting for the warm-up lock (retried on the next render)"
    out = registered.commands["chimera"]["handler"]("status")
    assert "timed out waiting for the warm-up lock" in out and "/chimera warmup" in out


# --- faithful speech switch (binding part) --------------------------------------------------------------------

def test_faithful_setting_and_tool_parameter(registered, plugin, tool, tmp_path):
    eng = plugin.engine
    prov = registered.tts_providers[0]
    prov.synthesize("Plain.", str(tmp_path / "a.mp3"))
    assert eng.calls_of("speak")[-1]["faithful"] is False
    registered.settings["faithful"] = True
    prov.synthesize("Faithful.", str(tmp_path / "b.mp3"))
    assert eng.calls_of("speak")[-1]["faithful"] is True
    list(prov.stream("Faithful stream.", format="pcm"))
    assert eng.calls_of("speak_pcm16")[-1]["faithful"] is True
    r = tool(registered, "chimera_speak", {"text": "Setting wins.", "wait_s": 30})
    assert r["faithful"] is True and eng.calls_of("speak")[-1]["faithful"] is True
    r = tool(registered, "chimera_speak", {"text": "Parameter wins.", "faithful": False, "wait_s": 30})
    assert r["faithful"] is False and eng.calls_of("speak")[-1]["faithful"] is False
    schema = registered.tools["chimera_speak"]["schema"]["parameters"]["properties"]
    assert schema["faithful"]["type"] == "boolean"
