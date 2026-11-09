"""Housekeeping and file placement of the Hermes binding: numba cache, temporary files, file modes.

* numba cache: the render handlers keep the numba cache folder the engine uses, however old its mtime (numba does
  not touch it on a cache hit); folders of other versions still go after ``keep_days``; without the engine's answer
  the numba cache is left alone. Red proof: ``paths.cleanup(keep_days)`` without ``keep_cache`` (9a616be).
* Temporary files: renders keep their temporary files under ``<HERMES_HOME>/cache/chimera/tmp`` (cleaned after a
  day), and leftovers of interrupted renders in the audio folder (``.chimera-tmp-*``, ``*.wav.part-*``) are removed
  after a day. Red proof: temp folder in the audio folder, never cleaned.
* File modes: chimera's folders are 0700 and its rendered files 0600 (like Hermes' media cache). Red proof: 0644 files,
  default-mode folders.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path

import pytest

SCORE = {"title": "Tidy", "bpm": 90,
         "voices": [{"name": "lead", "body": "female", "notes": [{"pitch": "C4", "beats": 1, "lyric": "Ah."}]}]}
DAY = 86400.0


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


def _age(p: Path, days: float) -> None:
    t = time.time() - days * DAY
    os.utime(p, (t, t))


def _numba_dirs(plugin):
    nb = plugin.hermes.home / "cache" / "chimera" / "numba"
    active, foreign = nb / "0123456789abcdef", nb / "fedcba9876543210"
    for d in (active, foreign):
        (d / "x").mkdir(parents=True)
        _age(d, 30)
    return active, foreign


@pytest.mark.parametrize("tool_name,args", [("chimera_render", {"score": SCORE, "wait_s": 30}),
                                             ("chimera_speak", {"text": "Hello.", "wait_s": 30})])
def test_active_numba_cache_survives(registered, plugin, tool, tool_name, args):
    active, foreign = _numba_dirs(plugin)
    plugin.engine.numba_dir = str(active)
    r = tool(registered, tool_name, args)
    assert r["success"] is True, r
    assert active.is_dir(), "the numba cache in use was deleted"
    assert not foreign.exists(), "an old cache folder of another version must still be removed"


def test_numba_cache_untouched_without_engine_answer(registered, plugin, tool):
    active, foreign = _numba_dirs(plugin)
    plugin.engine.numba_dir = None
    assert tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})["success"] is True
    assert active.is_dir() and foreign.is_dir()


def test_paths_cleanup_numba_needs_keep(plugin):
    paths = plugin.mod("paths")
    active, foreign = _numba_dirs(plugin)
    paths.cleanup(7)
    assert active.is_dir() and foreign.is_dir(), "keep_cache=None leaves the numba cache alone"
    paths.cleanup(7, keep_cache={active.name})
    assert active.is_dir() and not foreign.exists()


def test_render_tmp_dir_is_cleaned_cache(registered, plugin, tool):
    tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    tool(registered, "chimera_speak", {"text": "Hello.", "wait_s": 30})
    want = str(plugin.hermes.home / "cache" / "chimera" / "tmp")
    assert plugin.engine.calls_of("render_score")[-1]["tmp_dir"] == want
    assert plugin.engine.calls_of("speak")[-1]["tmp_dir"] == want


def test_leftovers_of_interrupted_renders_are_removed(plugin):
    paths = plugin.mod("paths")
    audio = paths.audio_dir()
    tmp = paths.tmp_dir(create=True)
    old_dir = audio / ".chimera-tmp-dd3fab06dfca"
    (old_dir).mkdir()
    (old_dir / "voice-0.f32").write_bytes(b"\0" * 64)
    old_part = audio / "20261101-101010-song.wav.part-1a2b3c4d"
    old_part.write_bytes(b"RIFF")
    old_tmp = tmp / ".chimera-tmp-abc123.f32"
    old_tmp.write_bytes(b"\0")
    fresh = audio / ".chimera-tmp-0000aaaa1111"
    fresh.mkdir()
    for p in (old_dir, old_part, old_tmp):
        _age(p, 2)
    paths.cleanup(0)                       # keep_days 0 keeps rendered audio, not temporary leftovers
    assert not old_dir.exists() and not old_part.exists() and not old_tmp.exists()
    assert fresh.is_dir(), "a running render's temp folder (younger than a day) stays"


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_private_modes(registered, plugin, tool):
    old = os.umask(0o022)
    try:
        r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
        s = tool(registered, "chimera_speak", {"text": "Private words.", "wait_s": 30})
    finally:
        os.umask(old)
    home = plugin.hermes.home
    for d in (home / "cache", home / "cache" / "audio", home / "cache" / "audio" / "chimera",
              home / "cache" / "chimera", home / "cache" / "chimera" / "tmp"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, (d, oct(d.stat().st_mode))
    for f in (r["file"], s["file"]):
        assert stat.S_IMODE(os.stat(f).st_mode) == 0o600, (f, oct(os.stat(f).st_mode))


@pytest.mark.parametrize("method,tool_name,args", [
    ("render_score", "chimera_render", {"score": SCORE, "wait_s": 30}),
    ("speak", "chimera_speak", {"text": "Nothing comes of this.", "wait_s": 30})])
def test_failed_render_drops_reserved_file(registered, plugin, tool, method, tool_name, args):
    """The output name is reserved (an empty file) before rendering; a failed render must not leave it."""
    plugin.engine.fail[method] = RuntimeError("worker died twice")
    out = tool(registered, tool_name, args)
    assert out["success"] is False
    audio = plugin.hermes.home / "cache" / "audio" / "chimera"
    left = [p.name for p in audio.iterdir()] if audio.is_dir() else []
    assert left == [], left
