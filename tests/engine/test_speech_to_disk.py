"""Speech without a memory bound in the host process (REVIEW-2A R2A-02), the faithful switch (REVIEW-2B R2B-02) and
temporary files under a chosen folder (R2A-05), with stub planner + FakeVoice workers.

* ``speak_to_wav`` writes the same bytes as ``speak(...)[0].to_wav`` (overlap-add from the window file in blocks,
  same float32 arithmetic as ``assemble``), ``speak_pcm16`` the same PCM, and both remove their temporary file.
* ``faithful=True`` plans exactly ``chimera render text``: one window, no settled first pitch, no text preparation
  (``plan_speech(split=False, settle_first=False, text_prep=False)``). Red proof: no such switch in 9a616be.
* ``render_score(tmp_dir=…)`` keeps its voice tracks under that folder. Red proof: always ``out_dir``.
* ``write_wav`` encodes block by block: same bytes as ``wav_bytes`` for any length and both widths.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

from chimera_vocal import EngineConfig, SpeechResult
from chimera_vocal.core import wavio
from chimera_vocal.core.wavio import wav_bytes, write_wav
from chimera_vocal.presets import PRESETS
from chimera_vocal.speech import WindowStore, assemble, plan_speech
from support.fakes import reference_render

TEXT = ("One two three, four five six. Seven eight nine, ten eleven twelve. Thirteen fourteen, fifteen sixteen. "
        "Hello there, my name is Chimera. Can you hear me?")


def _leftovers(d) -> list[str]:
    return sorted(n for n in os.listdir(d) if n.startswith(".chimera-tmp-"))


def test_speak_to_wav_equals_speak(stub_planner, fake_workers, make_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(WindowStore.__init__, "__kwdefaults__", {"block": 1000})    # many blocks per window
    eng = make_engine(workers=3)
    audio, kernel = eng.speak(TEXT, "female")
    ref = tmp_path / "ref.wav"
    audio.to_wav(ref)
    out = tmp_path / "out" / "speech.wav"
    tmp = tmp_path / "tmp"
    res = eng.speak_to_wav(TEXT, str(out), "female", tmp_dir=str(tmp))
    assert isinstance(res, SpeechResult) and res.path_wav == str(out) and res.kernel == kernel == "numpy"
    assert res.seconds == pytest.approx(len(audio.samples) / 48000.0) and res.sample_rate == 48000
    jobs, _ = plan_speech(TEXT, PRESETS["female"], speed=None, g2p=stub_planner.g2p)
    assert len(jobs) >= 4, "the text must give several windows"
    assert out.read_bytes() == ref.read_bytes()
    assert _leftovers(tmp) == [] and _leftovers(out.parent) == []


def test_speak_pcm16_equals_speak(stub_planner, fake_workers, make_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(WindowStore.__init__, "__kwdefaults__", {"block": 777})
    eng = make_engine(workers=3)
    audio, _ = eng.speak(TEXT, "male")
    pcm = b"".join(eng.speak_pcm16(TEXT, "male", tmp_dir=str(tmp_path)))
    assert pcm == wav_bytes(audio.samples, 48000)[44:]
    assert _leftovers(tmp_path) == []


def test_speak_to_wav_cancel_and_error_leave_nothing(stub_planner, fake_workers, make_engine, tmp_path):
    from chimera_vocal import RenderCancelled, TextError
    eng = make_engine(workers=2)
    ev = threading.Event()
    ev.set()
    with pytest.raises(RenderCancelled):
        eng.speak_to_wav(TEXT, str(tmp_path / "x.wav"), "male", cancel=ev, tmp_dir=str(tmp_path))
    with pytest.raises(TextError):
        eng.speak_to_wav("...", str(tmp_path / "y.wav"), "male", tmp_dir=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == []


@pytest.mark.parametrize("faithful", [False, True])
def test_faithful_plans_render_text(stub_planner, fake_workers, make_engine, faithful):
    """faithful=True renders plan_speech(split=False, settle_first=False, text_prep=False) and nothing else."""
    eng = make_engine(workers=3)
    text = "Dr. Smith paid $5, then left. It was 3 o'clock; nobody saw him."
    audio, _ = eng.speak(text, "child", faithful=faithful)
    if faithful:
        jobs, n_total = plan_speech(text, PRESETS["child"], speed=None, g2p=stub_planner.g2p, text_prep=False,
                                    split=False, settle_first=False)
        assert len(jobs) == 1 and jobs[0].settle_f0 is False and jobs[0].start_block == 0
    else:
        jobs, n_total = plan_speech(text, PRESETS["child"], speed=None, g2p=stub_planner.g2p)
        assert len(jobs) > 1
    assert np.array_equal(audio.samples, assemble([reference_render(j) for j in jobs], jobs, n_total))


def test_faithful_ignores_text_prep_setting(stub_planner, fake_workers, make_engine):
    eng = make_engine(workers=2, text_prep=True)
    text = "Dr. Smith paid $5."
    eng.speak(text, "male", faithful=True)
    seen = [args for name, args in stub_planner.g2p.calls if name == "to_phrases"]
    assert seen and seen[-1] == text, "faithful mode must read the text as given"


def test_render_score_tmp_dir(stub_planner, fake_workers, make_engine, tmp_path):
    eng = make_engine(workers=3)
    score = {"bpm": 120, "voices": [{"body": "female", "notes": [{"pitch": 60 + k, "beats": 1, "lyric": "ta"}
                                                                 for k in range(4)]}]}
    tmp = tmp_path / "tmp"
    out = tmp_path / "out"
    seen = []

    def progress(p):
        if tmp.is_dir():
            seen.extend(n for n in os.listdir(tmp) if n.startswith(".chimera-tmp-"))
    res = eng.render_score(score, str(out), "s", progress=progress, tmp_dir=str(tmp))
    assert os.path.isfile(res.path_wav) and seen, "the voice tracks were not under tmp_dir"
    assert _leftovers(out) == [] and _leftovers(tmp) == []


@pytest.mark.parametrize("n", [0, 1, 65535, 65536, 65537, 200001])
@pytest.mark.parametrize("bits", [16, 32])
def test_write_wav_blockwise_bytes(tmp_path, n, bits, monkeypatch):
    rng = np.random.default_rng(n)
    x = (rng.standard_normal(n) * 0.6).astype(np.float32)
    special = np.array([np.nan, np.inf, -np.inf, 1.5, -1.5, 0.5 / 32767, -0.5 / 32767], np.float32)
    for i, v in zip(range(0, n, max(1, n // len(special))), special):
        x[i] = v
    p = tmp_path / "a.wav"
    for sig in (x, np.stack([x, -x], axis=1)):
        write_wav(p, sig, 48000, bits=bits)
        assert p.read_bytes() == wav_bytes(sig, 48000, bits=bits)
    ref = wav_bytes(x, 48000, bits=bits)
    monkeypatch.setattr(wavio, "ENCODE_BLOCK", 3)          # bytes do not depend on the block size
    assert wav_bytes(x, 48000, bits=bits) == ref
    write_wav(p, x, 48000, bits=bits)
    assert p.read_bytes() == ref


def test_apply_settings_takes_effect(fake_workers, make_engine):
    """R2A-04 (engine part): the user-facing settings change the running engine; cache_dir stays."""
    eng = make_engine(workers=2, backend="auto", text_prep=True, max_voices=8)
    cache = eng.cache_base
    eng.apply_settings(EngineConfig(backend="numpy", workers=3, text_prep=False, max_voices=2,
                                    cache_dir="/somewhere/else"))
    assert (eng.config.backend, eng.config.workers, eng.config.text_prep, eng.config.max_voices) == \
        ("numpy", 3, False, 2)
    assert eng.scheduler.backend == "numpy" and eng.workers == 3 and eng.scheduler.workers == 3
    assert (eng.scheduler.music_cap, eng.scheduler.total_cap) == (2, 3)
    assert eng.cache_base == cache and eng.config.cache_dir == cache
    report = eng.check_score({"bpm": 100, "voices": [{"notes": [{"pitch": "C4", "beats": 1}]}] * 3})
    assert not report.ok and any("limit of 2" in i.message for i in report.issues)
