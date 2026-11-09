"""Engine API surface (INTERFACES §5) with stub planner + FakeVoice workers: speak, speak_pcm16, check/render_score
end to end (bit-exact against FakeVoice references through compile + mix), status, in-process fallback, errors,
EngineConfig/Audio, and the numpy-free import rule (§1 rule 1, K-6.11 relies on it)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

from chimera_vocal import (Audio, BackendError, BackendStatus, EngineConfig, RenderCancelled, ScoreError,
                           ScoreResult, TextError)
from chimera_vocal.core.wavio import read_wav, wav_bytes
from chimera_vocal.music.compile import compile_score
from chimera_vocal.music.mixer import MixInfo, mix
from chimera_vocal.music.schema import validate_score
from chimera_vocal.presets import PRESETS
from chimera_vocal.speech import assemble, plan_speech
from support import REPO_ROOT
from support.fakes import reference_render

TEXT = "Hello there, my name is Chimera. Can you hear me?"


def small_score(pan: bool = False) -> dict:
    v = lambda body, start, p: {"body": body, "start": start, "pan": p,
                                "notes": [{"pitch": 60 + k, "beats": 1, "lyric": "ta"} for k in range(4)]}
    return {"bpm": 120, "voices": [v("female", 0, -0.5 if pan else 0), v("male", 2, 0.5 if pan else 0)]}


def test_config_defaults_and_rate():
    c = EngineConfig()
    assert (c.backend, c.workers, c.in_process_fallback, c.cache_dir, c.sample_rate, c.text_prep, c.max_voices,
            c.max_score_seconds, c.max_voice_seconds, c.worker_idle_s) == \
        ("auto", 0, True, None, 48000, True, 8, 300.0, 1200.0, 60.0)
    with pytest.raises(ValueError):
        EngineConfig(sample_rate=44100)
    with pytest.raises(ValueError):
        EngineConfig(backend="cuda")


def test_audio():
    x = (np.sin(np.arange(4800) / 10) * 0.5).astype(np.float32)
    a = Audio(x, 48000)
    assert a.duration_s() == pytest.approx(0.1)
    assert a.pcm16_bytes() == wav_bytes(x, 48000)[44:]
    st = Audio(np.stack([x, -x], axis=1), 48000)
    assert st.pcm16_bytes() == wav_bytes(st.samples, 48000)[44:]


def test_audio_to_wav(tmp_path):
    x = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    p = Audio(x, 48000).to_wav(tmp_path / "a.wav")
    assert p == str(tmp_path / "a.wav")
    sr, y = read_wav(p)
    assert sr == 48000 and len(y) == 480


def test_speak(stub_planner, fake_workers, make_engine):
    eng = make_engine(workers=3)
    audio, kernel = eng.speak(TEXT, "female")
    assert kernel == "numpy" and isinstance(audio, Audio) and audio.sample_rate == 48000
    jobs, n_total = plan_speech(TEXT, PRESETS["female"], speed=None, g2p=stub_planner.g2p)
    ref = assemble([reference_render(j) for j in jobs], jobs, n_total)
    assert audio.samples.dtype == np.float32 and np.array_equal(audio.samples, ref)


def test_speak_settle_first_false(stub_planner, fake_workers, make_engine):
    eng = make_engine(workers=2)
    audio, _ = eng.speak(TEXT, "female", settle_first=False)
    jobs, n_total = plan_speech(TEXT, PRESETS["female"], speed=None, g2p=stub_planner.g2p, settle_first=False)
    assert np.array_equal(audio.samples, assemble([reference_render(j) for j in jobs], jobs, n_total))


def test_speak_pcm16_matches_speak(stub_planner, fake_workers, make_engine):
    eng = make_engine(workers=3)
    audio, _ = eng.speak(TEXT, "male")
    pcm = b"".join(eng.speak_pcm16(TEXT, "male"))
    assert pcm == audio.pcm16_bytes()


def test_speak_spec_and_preset_object(stub_planner, fake_workers, make_engine):
    eng = make_engine(workers=2)
    a1, _ = eng.speak("Hello there.", "female;pitch=230")
    from chimera_vocal.presets import parse_voice_spec
    a2, _ = eng.speak("Hello there.", parse_voice_spec("female;pitch=230"))
    assert np.array_equal(a1.samples, a2.samples)


def test_speak_nothing_speakable(stub_planner, fake_workers, make_engine):
    with pytest.raises(TextError):
        make_engine(workers=2).speak("!!!", "male")


def test_speak_cancelled(stub_planner, fake_workers, make_engine):
    ev = threading.Event()
    ev.set()
    with pytest.raises(RenderCancelled):
        make_engine(workers=2).speak(TEXT, "male", cancel=ev)


def test_in_process_fallback(stub_planner, fake_workers, fake_voice, make_engine, monkeypatch):
    monkeypatch.setattr(sys, "executable", os.path.join(str(REPO_ROOT), "no-such-python"))
    eng = make_engine(workers=2, in_process_fallback=True)
    audio, kernel = eng.speak("Hello there.", "male")
    jobs, n_total = plan_speech("Hello there.", PRESETS["male"], speed=None, g2p=stub_planner.g2p)
    assert kernel == "numpy"
    assert np.array_equal(audio.samples, assemble([reference_render(j) for j in jobs], jobs, n_total))


def test_no_worker_no_fallback(stub_planner, fake_workers, fake_voice, make_engine, monkeypatch):
    monkeypatch.setattr(sys, "executable", os.path.join(str(REPO_ROOT), "no-such-python"))
    with pytest.raises(BackendError):
        make_engine(workers=2, in_process_fallback=False).speak("Hello there.", "male")


@pytest.mark.parametrize("pan", [False, True], ids=["mono", "stereo"])
def test_render_score_end_to_end(stub_planner, fake_workers, make_engine, tmp_path, pan):
    eng = make_engine(workers=3)
    prog = []
    res = eng.render_score(small_score(pan), str(tmp_path), "duet", want_mp3=False, progress=prog.append)
    assert isinstance(res, ScoreResult) and isinstance(res.mix, MixInfo)
    assert res.path_mp3 is None and res.kernel == "numpy" and res.voices == 2 and res.sample_rate == 48000
    assert res.channels == (2 if pan else 1) and res.render_s > 0 and isinstance(res.warnings, list)
    assert os.path.dirname(res.path_wav) == str(tmp_path) and os.path.basename(res.path_wav).startswith("duet")
    assert prog and prog == sorted(prog) and 0 <= prog[0] and prog[-1] == pytest.approx(1.0)
    report = validate_score(small_score(pan), g2p=stub_planner.g2p)
    jobs, mixes, n_mix = compile_score(report, g2p=stub_planner.g2p)
    ref, info = mix([reference_render(j) for j in jobs], mixes, n_mix, stereo=pan)
    assert res.seconds == pytest.approx(n_mix / 48000)
    with open(res.path_wav, "rb") as f:
        assert f.read() == wav_bytes(ref, 48000), "engine output = compile + reference renders + mix"
    assert res.mix == info
    assert sorted(os.listdir(tmp_path)) == [os.path.basename(res.path_wav)], "no temporary files left"


def test_render_score_mp3_refused(stub_planner, fake_workers, make_engine, tmp_path):
    """The engine never calls ffmpeg: MP3 is the Hermes adapter's job, want_mp3=True is a caller error."""
    with pytest.raises(ValueError):
        make_engine(workers=2).render_score(small_score(), str(tmp_path), "x", want_mp3=True)
    assert os.listdir(tmp_path) == []


def test_render_score_invalid(stub_planner, fake_workers, make_engine, tmp_path):
    with pytest.raises(ScoreError) as ei:
        make_engine(workers=2).render_score({"bpm": 500, "voices": []}, str(tmp_path), "x", want_mp3=False)
    assert {e.path for e in ei.value.errors} == {"bpm", "voices"}
    assert os.listdir(tmp_path) == []


def test_render_score_cancel_before_start(stub_planner, fake_workers, make_engine, tmp_path):
    ev = threading.Event()
    ev.set()
    with pytest.raises(RenderCancelled):
        make_engine(workers=2).render_score(small_score(), str(tmp_path), "x", want_mp3=False, cancel=ev)
    assert os.listdir(tmp_path) == []


def test_status(fake_workers, make_engine, tmp_path):
    eng = make_engine(workers=3)
    st = eng.status()
    assert isinstance(st, BackendStatus)
    assert st.numpy_ok is True and st.numba_installed is False and st.numba_state == "absent"
    assert st.numba_version is None and st.marker is None and st.restart_needed is False
    from chimera_vocal.music.estimate import DEFAULT_RTF
    assert st.rtf_estimate == DEFAULT_RTF and st.workers == 3
    assert st.cache_dir == str(tmp_path / "numba-cache") or (st.cache_dir or "").startswith(str(tmp_path))


def test_status_numba_installed_never_imported(fake_workers, make_engine, monkeypatch):
    import chimera_vocal.scheduler as sched
    monkeypatch.setattr(sched, "_numba_installed", lambda: True)
    st = make_engine(workers=2).status()
    assert st.numba_installed is True and st.numba_state == "cold"
    assert "numba" not in sys.modules, "the host never imports numba (P1-11)"


def test_warmup_is_non_blocking(fake_workers, make_engine):
    import time
    eng = make_engine(workers=2)
    t0 = time.monotonic()
    eng.warmup()
    assert time.monotonic() - t0 < 0.5


NO_NUMPY = textwrap.dedent(r"""
    import sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("numpy", "numba"):
                raise ImportError("blocked: " + name)
            return None
    sys.meta_path.insert(0, Block())
    sys.path.insert(0, sys.argv[1])
    import chimera_vocal, chimera_vocal.presets, chimera_vocal.runtime, chimera_vocal.scheduler, chimera_vocal.speech
    import chimera_vocal.music.schema, chimera_vocal.music.estimate, chimera_vocal.music.planning
    from chimera_vocal import Engine, BackendError
    from chimera_vocal.presets import list_voices
    assert len(list_voices()) == 12
    eng = Engine()
    st = eng.status()
    assert st.numpy_ok is False, st
    try:
        eng.speak("Hello.", "male")
    except BackendError:
        pass
    else:
        raise AssertionError("speak without numpy must raise BackendError")
    eng.shutdown()
    assert not any(m.split(".")[0] in ("numpy", "numba") for m in sys.modules)
    print("ok")
""")


def test_engine_without_numpy():
    r = subprocess.run([sys.executable, "-I", "-c", NO_NUMPY, str(REPO_ROOT)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-3000:]


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "chimera_vocal", "--help"], capture_output=True, text=True,
                       cwd=str(REPO_ROOT), timeout=60)
    assert r.returncode == 0
    for cmd in ("say", "sing", "check", "selftest", "bench"):
        assert cmd in r.stdout


def test_cli_check_reports_issues(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bpm": 500, "voices": []}), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "chimera_vocal", "check", str(bad)], capture_output=True, text=True,
                       cwd=str(REPO_ROOT), timeout=120)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "bpm" in r.stdout and "voices" in r.stdout
