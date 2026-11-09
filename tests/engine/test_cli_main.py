"""``python -m chimera_vocal`` (``sing``, ``bench``) and the warm-up marker write under a concurrent reader.

* ``sing --mp3`` is gone (the engine never runs ffmpeg; it always failed), ``bench --seconds`` sets the length of
  the held vowel. Red proof: 9a616be accepts ``--mp3`` and then raises; ``bench`` renders "ah." whatever
  ``--seconds`` says.
* ``replace_retrying`` rides out ``PermissionError`` from ``os.replace`` (Windows, a reader holds the marker open).
  Red proof: a single ``os.replace``.
"""

from __future__ import annotations

import json
import os

import pytest

from chimera_vocal import __main__ as cli
from chimera_vocal.dsp import backend


def test_sing_has_no_mp3_option(tmp_path, capsys):
    score = tmp_path / "s.json"
    score.write_text(json.dumps({"bpm": 90, "voices": [{"notes": [{"pitch": "C4", "beats": 1}]}]}), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        cli.main(["sing", str(score), "--out-dir", str(tmp_path), "--mp3"])
    assert ei.value.code == 2 and "--mp3" in capsys.readouterr().err


def test_bench_seconds(stub_planner, fake_workers, capsys):
    out = {}
    for s in (0.5, 3.0):
        assert cli.main(["bench", "--seconds", str(s), "--workers", "2"]) == 0
        out[s] = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out[0.5]["vowel_s"] == 0.5 and out[3.0]["vowel_s"] == 3.0
    assert out[3.0]["audio_s"] - out[0.5]["audio_s"] == pytest.approx(2.5, abs=0.05)
    assert out[3.0]["rtf"] > 0


def test_say_faithful_option(stub_planner, fake_workers, tmp_path, capsys):
    p = tmp_path / "say.wav"
    assert cli.main(["say", "Dr. Smith, hello.", "--faithful", "--out", str(p), "--workers", "2"]) == 0
    assert p.read_bytes()[:4] == b"RIFF"
    seen = [a for n, a in stub_planner.g2p.calls if n == "to_phrases"]
    assert seen[-1] == "Dr. Smith, hello."


def test_marker_replace_retries_permission_error(tmp_path, monkeypatch):
    src, dst = tmp_path / "m.tmp", tmp_path / "marker.json"
    src.write_text("{}", encoding="utf-8")
    real = os.replace
    fails = {"n": 3}

    def flaky(a, b):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(13, "The process cannot access the file because it is being used")
        return real(a, b)
    monkeypatch.setattr(backend.os, "replace", flaky)
    monkeypatch.setattr(backend, "REPLACE_DELAY_S", 0.001)
    backend.replace_retrying(str(src), str(dst))
    assert dst.read_text(encoding="utf-8") == "{}" and fails["n"] == 0
    fails["n"] = 10 ** 6
    src.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError):
        backend.replace_retrying(str(src), str(dst))


def test_warm_numba_uses_replace_retrying():
    import inspect
    src = inspect.getsource(backend.warm_numba)
    assert "replace_retrying(" in src and "os.replace(" not in src
