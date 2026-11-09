"""File names, storage, clean-up and ffmpeg (K-6.14, K-6.15)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import c5fakes

STEM = re.compile(r"[a-z0-9-]{1,48}")


# K-6.14: line breaks, spaces, device names, 200 chars, Unicode -> valid stem [a-z0-9-]{1,48} (red proof: raw title)
@pytest.mark.parametrize("raw,expected", [
    ("Evening Round", "evening-round"),
    ("a\nb\rc", "a-b-c"),
    ("  spaced   out  ", "spaced-out"),
    ("Row, Row, Row Your Boat!", "row-row-row-your-boat"),
    ("Ünïcödé Café", "unicode-cafe"),
    ("con", "score"), ("CON", "score"), ("prn", "score"), ("aux", "score"), ("nul", "score"),
    ("com1", "score"), ("COM9", "score"), ("lpt1", "score"), ("Lpt9", "score"),
    ("con.txt", "con-txt"),
    ("日本語", "score"), ("", "score"), ("---", "score"), ("   ", "score"), (None, "score"), (42, "score"),
    ("x" * 200, "x" * 48),
    ("MEDIA:/etc/passwd\nMEDIA:/x", "media-etc-passwd-media-x"),
    ("../../secret", "secret"),
])
def test_slug(plugin, raw, expected):
    s = plugin.mod("paths").slug(raw, "score")
    assert s == expected
    assert STEM.fullmatch(s) and "\n" not in s and "\r" not in s and " " not in s


def test_slug_fuzz(plugin):
    import random

    slug = plugin.mod("paths").slug
    rng = random.Random(14)
    alphabet = "aZ09 -_.\n\r\t/\\:*?\"<>|ÄöüßéñçøœæłþðΩжह😀 \u0000"
    for _ in range(2000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 220)))
        s = slug(raw, "speech")
        assert STEM.fullmatch(s), (raw, s)
        assert s.lower() not in {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
            f"lpt{i}" for i in range(1, 10)}


def test_storage_locations(plugin):
    paths = plugin.mod("paths")
    home = plugin.hermes.home
    assert paths.audio_dir() == home / "cache" / "audio" / "chimera" and paths.audio_dir().is_dir()
    assert paths.cache_dir() == home / "cache" / "chimera"
    assert paths.numba_cache_dir() == home / "cache" / "chimera" / "numba"
    assert paths.tmp_dir() == home / "cache" / "chimera" / "tmp"
    legacy = home / "audio_cache"
    legacy.mkdir()
    (legacy / "old.mp3").write_bytes(b"x")
    assert paths.audio_dir() == legacy / "chimera", "legacy audio_cache with content wins (get_hermes_dir)"
    shutil.rmtree(legacy)
    d1, s1 = paths.new_output_stem("Same name", "score", now=1_700_000_000.0)
    d2, s2 = paths.new_output_stem("Same name", "score", now=1_700_000_000.0)
    assert d1 == d2 == paths.audio_dir() and s1 != s2
    assert re.fullmatch(r"\d{8}-\d{6}-same-name", s1) and s2.startswith(s1)
    assert (d1 / f"{s1}.wav").exists() and (d2 / f"{s2}.wav").exists()


def test_cleanup(plugin):
    paths = plugin.mod("paths")
    d = paths.audio_dir()
    now = time.time()
    old = now - 10 * 86400

    def touch(p: Path, t: float, data=b"x"):
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(data) if not p.suffix == "" else p.mkdir()
        os.utime(p, (t, t))
        return p

    own_old = touch(d / "20261030-101010-old-song.wav", old)
    own_old_mp3 = touch(d / "20261030-101010-old-song.mp3", old)
    own_new = touch(d / "20261109-101010-new-song.wav", now)
    foreign_old = touch(d / "my-notes.wav", old)
    tmp_old = touch(paths.tmp_dir() / "job-1" / "voice0.f32", old)
    os.utime(paths.tmp_dir() / "job-1", (now - 2 * 86400,) * 2)
    tmp_new = touch(paths.tmp_dir() / "job-2.f32", now)
    nb_keep = paths.numba_cache_dir() / "aaaaaaaaaaaaaaaa"
    nb_drop = paths.numba_cache_dir() / "bbbbbbbbbbbbbbbb"
    for p in (nb_keep, nb_drop):
        p.mkdir(parents=True)
        os.utime(p, (old, old))
    n = paths.cleanup(7, now=now, keep_cache={"aaaaaaaaaaaaaaaa"})
    assert not own_old.exists() and not own_old_mp3.exists() and not nb_drop.exists()
    assert own_new.exists() and foreign_old.exists() and nb_keep.exists() and tmp_new.exists()
    assert not (paths.tmp_dir() / "job-1").exists()
    assert n >= 4
    touch(d / "20261030-101010-old-song.wav", old)
    assert paths.cleanup(0, now=now) >= 0 and (d / "20261030-101010-old-song.wav").exists(), "keep_days 0 keeps"
    assert tmp_old.parent.exists() is False


class _Run:
    def __init__(self, rc=0, raise_=None, write=True):
        self.calls = []
        self.rc, self.raise_, self.write = rc, raise_, write

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        if self.raise_ is not None:
            raise self.raise_
        if self.write and self.rc == 0:
            Path(cmd[-1]).write_bytes(b"ID3fake-mp3")
        return subprocess.CompletedProcess(cmd, self.rc, b"", b"encoder libmp3lame not found" if self.rc else b"")


@pytest.fixture
def ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/opt/ff/ffmpeg" if name == "ffmpeg" else None)
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    return run


# K-6.15: stdin=DEVNULL, timeout max(60, 2·seconds), hidden window on Windows; failure/missing ffmpeg -> WAV + warning
# (red proof: exception passed through)
def test_ffmpeg(plugin, ffmpeg, tmp_path, monkeypatch):
    media = plugin.mod("media")
    wav, mp3 = tmp_path / "a.wav", tmp_path / "a.mp3"
    c5fakes.write_wav16(wav, c5fakes.fake_samples("x", 0.1))
    assert media.to_mp3(str(wav), str(mp3), seconds=10.0) is True and mp3.is_file()
    cmd, kw = ffmpeg.calls[-1]
    assert cmd[0] == "/opt/ff/ffmpeg" and cmd[-1] == str(mp3) and str(wav) in cmd
    assert "-y" in cmd and "libmp3lame" in cmd
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["timeout"] == pytest.approx(60.0)
    # hidden window exactly on Windows: CREATE_NO_WINDOW there, no flags elsewhere
    assert kw.get("creationflags", 0) == (0x08000000 if sys.platform == "win32" else 0)
    media.to_mp3(str(wav), str(mp3), seconds=100.0)
    assert ffmpeg.calls[-1][1]["timeout"] == pytest.approx(200.0)
    monkeypatch.setattr(sys, "platform", "win32")
    media.to_mp3(str(wav), str(mp3), seconds=1.0)
    monkeypatch.setattr(sys, "platform", "linux")
    assert ffmpeg.calls[-1][1]["creationflags"] == 0x08000000

    for failing in (_Run(raise_=subprocess.TimeoutExpired("ffmpeg", 60)), _Run(raise_=OSError(8, "Exec format error")),
                    _Run(rc=1), _Run(rc=0, write=False), _Run(raise_=PermissionError("denied"))):
        if mp3.exists():
            mp3.unlink()
        monkeypatch.setattr(subprocess, "run", failing)
        assert media.to_mp3(str(wav), str(mp3), seconds=3.0) is False
        assert not mp3.exists()

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    never = _Run()
    monkeypatch.setattr(subprocess, "run", never)
    assert media.to_mp3(str(wav), str(mp3), seconds=3.0) is False and never.calls == []
    spaced = Path("/x/y z.wav")             # the native form of the path, spaces kept, nothing quoted
    assert media.media_line(spaced) == "MEDIA:" + str(spaced)
    assert sys.platform == "win32" or media.media_line(spaced) == "MEDIA:/x/y z.wav"


SCORE = {"bpm": 90, "voices": [{"notes": [{"pitch": "C4", "beats": 1}]}]}


def test_ffmpeg_in_render(registered, plugin, tool, ffmpeg, monkeypatch):
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["format"] == "mp3" and r["file"].endswith(".mp3") and r["media"] == "MEDIA:" + r["file"]
    assert Path(r["wav"]).is_file() and Path(r["file"]).is_file()
    assert plugin.engine.calls_of("render_score")[-1]["want_mp3"] is False
    monkeypatch.setattr(subprocess, "run", _Run(rc=1))
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["success"] is True and r["format"] == "wav" and r["file"] == r["wav"]
    assert any("ffmpeg" in w.lower() for w in r["warnings"]), r["warnings"]
    monkeypatch.setattr(subprocess, "run", _Run(raise_=subprocess.TimeoutExpired("ffmpeg", 60)))
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["success"] is True and r["format"] == "wav"
    r = tool(registered, "chimera_render", {"score": {**SCORE, "format": "wav"}, "wait_s": 30})
    assert r["format"] == "wav" and not any("ffmpeg" in w.lower() for w in r["warnings"])
    registered.settings["music_format"] = "wav"
    monkeypatch.setattr(subprocess, "run", ffmpeg)
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["format"] == "wav"
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    registered.settings["music_format"] = "mp3"
    r = tool(registered, "chimera_render", {"score": SCORE, "wait_s": 30})
    assert r["format"] == "wav" and any("ffmpeg" in w.lower() for w in r["warnings"])
