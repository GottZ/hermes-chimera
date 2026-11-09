"""/chimera slash command and `hermes chimera` CLI (INTERFACES §12; DESIGN §7)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

import c5fakes


@pytest.fixture
def slash(registered):
    return registered.commands["chimera"]["handler"]


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


def test_status_without_numba(slash, plugin):
    out = slash("status")
    assert out == slash("") == slash("   ")
    low = out.lower()
    assert "numpy" in low and "available" in low
    assert "chimera-numba" in out and "#accel" in out and "--yes-deps" in out and "restart hermes" in low
    assert "capabilities" in low and "plugins" in low
    assert "ffmpeg" in low and "wav" in low
    assert "tts.provider" in out and "hermes config set tts.provider chimera" in out
    assert "voice: male" in out
    assert str(plugin.hermes.home / "cache" / "chimera" / "numba") in out


def test_status_states(slash, plugin, registered, monkeypatch):
    eng = plugin.engine
    eng.numba_installed, eng.numba_version, eng.numba_state = True, "0.68.1", "ready"
    plugin.hermes.config["tts"] = {"provider": "chimera", "voice": "child"}
    out = slash("status")
    assert "0.68.1" in out and "ready" in out and "chimera-numba" not in out
    assert "tts.provider: chimera" in out and "voice: child" in out
    eng.restart_needed = True
    assert "restart hermes" in slash("status").lower()
    eng.restart_needed = False
    eng.numba_state, eng.numba_error = "failed", "selftest mismatch 3e-2"
    out = slash("status")
    assert "failed" in out and "selftest mismatch 3e-2" in out and "numpy" in out
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    assert "isolation" in slash("status")
    eng.fail["status"] = RuntimeError("engine gone")
    out = slash("status")
    assert isinstance(out, str) and "engine gone" in out
    assert plugin.hermes.writes == [] and registered.config_writes == []


def test_voices_and_voice(slash, plugin, registered):
    out = slash("voices")
    for vid in c5fakes.PRESET_TABLE:
        assert vid in out
    out = slash("voice female")
    assert registered.config_writes == [("voice", "female")] and registered.settings["voice"] == "female"
    assert "female" in out and "tts.provider chimera" in out and "Text-to-Speech" in out
    plugin.hermes.config["tts"] = {"provider": "chimera"}
    out = slash("voice female;pitch=230")
    assert registered.settings["voice"] == "female;pitch=230" and "tts.provider chimera" not in out
    out = slash("voice alloy")
    assert "unknown voice" in out.lower() and registered.settings["voice"] == "female;pitch=230"
    assert "male" in out and "elder" in out
    assert "voice" in slash("voice").lower()
    assert plugin.hermes.writes == [], "the plugin must never write Hermes core configuration"


def test_warmup_and_test(slash, plugin):
    eng = plugin.engine
    out = slash("warmup")
    assert eng.calls_of("warmup") and isinstance(out, str) and "chimera-numba" in out
    eng.numba_installed, eng.numba_state = True, "warming"
    out = slash("warmup")
    assert "warming" in out
    out = slash("test")
    assert eng.calls_of("selftest") and "passed" in out
    assert "unknown" in slash("dance").lower() and "status" in slash("dance")


def _cli(registered, argv):
    parser = argparse.ArgumentParser(prog="hermes chimera")
    registered.cli_commands["chimera"]["setup_fn"](parser)
    args = parser.parse_args(argv)
    return registered.cli_commands["chimera"]["handler_fn"](args)


def test_cli(registered, plugin, tmp_path, capsys):
    assert _cli(registered, ["status"]) == 0
    assert "numpy" in capsys.readouterr().out
    assert _cli(registered, ["voices"]) == 0
    assert "narrator" in capsys.readouterr().out
    out_wav = tmp_path / "hi.wav"
    assert _cli(registered, ["say", "Hello", "world.", "--voice", "female", "--out", str(out_wav)]) == 0
    assert c5fakes.wav_info(out_wav)["magic"] == b"RIFF"
    call = plugin.engine.calls_of("speak")[-1]
    assert call["text"] == "Hello world." and c5fakes.FakeEngine.voice_id(call["voice"]) == "female"
    assert _cli(registered, ["say", "x", "--voice", "alloy", "--out", str(tmp_path / "n.wav")]) != 0
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"bpm": 90, "voices": [{"notes": [{"pitch": "C4", "beats": 1}]}]}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"voices": []}), encoding="utf-8")
    capsys.readouterr()
    assert _cli(registered, ["check", str(good)]) == 0
    assert _cli(registered, ["check", str(bad)]) == 1
    assert "bpm" in capsys.readouterr().out
    out = tmp_path / "song.wav"
    assert _cli(registered, ["sing", str(good), "--out", str(out)]) == 0
    assert out.is_file()
    assert _cli(registered, ["sing", str(bad), "--out", str(tmp_path / "x.wav")]) == 1
    assert _cli(registered, ["selftest"]) == 0
    assert _cli(registered, ["warmup"]) == 0
    assert _cli(registered, ["check", str(tmp_path / "missing.json")]) != 0


def _plugin_copy(tmp_path):
    """A copy of the plugin inside a Hermes-like plugins dir: <tmp>/plugins/chimera."""
    import shutil as sh

    root = tmp_path / "plugins" / "chimera"
    root.mkdir(parents=True)
    for name in ("__init__.py", "plugin.yaml"):
        sh.copy2(c5fakes.REPO_ROOT / name, root / name)
    sh.copytree(c5fakes.REPO_ROOT / "chimera_hermes", root / "chimera_hermes",
                ignore=sh.ignore_patterns("__pycache__"))
    return root


# Companion install id: Hermes' install record > the plugin's git origin > the published repository; never credentials
def test_install_source(tmp_path, fake_hermes):
    root = _plugin_copy(tmp_path)
    pkg = c5fakes.load_plugin(root=root)
    try:
        cmd = c5fakes.submodule(pkg, "chimera_hermes.commands")
        assert cmd.install_source() == "GottZ/hermes-chimera"
        hint = cmd.companion_hint()
        assert "'GottZ/hermes-chimera#accel'" in hint and "--yes-deps" in hint and "restart Hermes" in hint

        git = root / ".git"
        git.mkdir()
        (git / "config").write_text('[core]\n\tbare = false\n[remote "upstream"]\n\turl = https://example.org/x/y\n'
                                    '[remote "origin"]\n\turl = https://alice:TOPSECRET@github.com/alice/chimera-fork.git\n'
                                    '\tfetch = +refs/heads/*:refs/remotes/origin/*\n', encoding="utf-8")
        assert cmd.install_source() == "alice/chimera-fork"
        assert "TOPSECRET" not in cmd.companion_hint()

        sidecar = root.parent / ".install-metadata.json"
        sidecar.write_text(json.dumps({"chimera": {"pinned": False, "revision": "0" * 40,
                                                   "source": "https://bob:tok@git.example.com/bob/chimera.git?x=1#accel"},
                                       "other": {"source": "https://github.com/x/other"}}), encoding="utf-8")
        assert cmd.install_source() == "https://git.example.com/bob/chimera.git"
        sidecar.write_text(json.dumps({"chimera": {"source": "file:///srv/plugins/chimera"}}), encoding="utf-8")
        assert cmd.install_source() == "file:///srv/plugins/chimera"
        sidecar.write_text(json.dumps({"chimera": {"source": "https://github.com/GottZ/hermes-chimera.git"}}),
                           encoding="utf-8")
        assert cmd.install_source() == "GottZ/hermes-chimera"
        sidecar.write_text("{broken", encoding="utf-8")
        assert cmd.install_source() == "alice/chimera-fork"
    finally:
        c5fakes.unload_plugin(pkg)


# A running Hermes never switches to the environment an install commits: chimera-numba installed after start means
# "restart Hermes", not "install the companion plugin" (what Hermes itself prints after the install)
def test_status_restart_hint_when_companion_installed(tmp_path, fake_hermes):
    root = _plugin_copy(tmp_path)
    pkg = c5fakes.load_plugin(root=root)
    try:
        cmd = c5fakes.submodule(pkg, "chimera_hermes.commands")
        assert not cmd.companion_on_disk()
        assert "install the companion plugin" in cmd.numba_missing_line()
        comp = root.parent / "chimera-numba"
        comp.mkdir()
        (comp / "plugin.yaml").write_text("manifest_version: 2\nname: chimera-numba\n", encoding="utf-8")
        assert cmd.companion_on_disk()
        line = cmd.numba_missing_line().lower()
        assert "restart hermes" in line and "install the companion plugin" not in line
        (comp / "plugin.yaml").unlink()
        comp.rmdir()
        (root.parent / ".install-metadata.json").write_text(
            json.dumps({"chimera-numba": {"source": "https://github.com/GottZ/hermes-chimera.git#accel"}}), encoding="utf-8")
        assert cmd.companion_on_disk()
        (root.parent / ".install-metadata.json").write_text("{broken", encoding="utf-8")
        assert not cmd.companion_on_disk()
    finally:
        c5fakes.unload_plugin(pkg)
