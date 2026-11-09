"""K-5.9 test_plan_speech_windows: window bounds, start blocks, seeds per window (``Seeds.for_window``) and gesture
ownership against a phrase-split measurement on the native template (data/e1a_phrase_split.json: 8 texts ×
bodies, 46 windows of the chosen split rule: window k from t0_k − 0.05 s to end_k + 0.15 s, settled first pitch);
``split=False`` = one job of exactly the ``chimera render text`` length. The stub planner reproduces the measured
phrase timing (t0 chain from 0.05, end_k − t0_k per phrase, pauses 0.18/0.35)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chimera_vocal import TextError
from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.jsmath import js_round
from chimera_vocal.core.types import Phrase, PhraseInput, PlanOptions, Seeds, Segment, Intonation
from chimera_vocal.music.compile import RenderJob
from chimera_vocal.presets import PRESETS, parse_voice_spec
from chimera_vocal.speech import assemble, plan_speech

DATA = json.loads((Path(__file__).parent / "data" / "e1a_phrase_split.json").read_text(encoding="utf-8"))
JOBS = DATA["jobs"]


def prime_durations(planner, job: dict) -> list[Phrase]:
    phrases = planner.g2p.to_phrases(job["text"])
    assert len(phrases) == len(job["phrases"]), "stub G2P splits the measured text into the same phrases"
    for ph, ref in zip(phrases, job["phrases"]):
        assert ph.pause == ref["pause"] or ph is phrases[-1]
        planner.durations["".join(s.ipa for s in ph.segments)] = ref["end"] - ref["t0"]
    return phrases


def phrase_gestures(planner, phrases: list[Phrase], opts: PlanOptions) -> list[list]:
    """Gestures the stub plans for each phrase on the common clock (t0 chain of plan_phrases)."""
    out, t = [], opts.t0
    for p in phrases:
        o = PlanOptions(opts.body, t, opts.rate, opts.f0, opts.intonation, opts.accent, opts.open_end, opts.sung)
        plan = planner.plan_phrases([PhraseInput(p.segments, p.intonation, p.pause)], o)
        out.append(plan.gestures)
        t = plan.end + p.pause
    return out


@pytest.mark.parametrize("job", JOBS, ids=[j["tag"] for j in JOBS])
def test_windows_against_e1a(stub_planner, job):
    phrases = prime_durations(stub_planner, job)
    preset = PRESETS[job["body"]]
    stub_planner.calls.clear()
    jobs, n_total = plan_speech(job["text"], preset, speed=None, g2p=stub_planner.g2p)
    assert n_total == job["n_total"], "total length = chimera render text length js_round(fs·(end + 0.3))"
    assert len(jobs) == len(job["windows"])
    calls = stub_planner.calls_of("plan_phrases")
    assert calls, "plan_speech plans through plan_phrases"
    opts = calls[0][1]
    assert opts.t0 == 0.05 and opts.f0 == preset.speech.f0 and opts.body == preset.body
    assert opts.rate in (None, 1.0) and opts.accent is None
    expected_g = phrase_gestures(stub_planner, phrases, PlanOptions(preset.body, 0.05, opts.rate, opts.f0))
    for k, (rj, (w, e)) in enumerate(zip(jobs, job["windows"])):
        assert isinstance(rj, RenderJob)
        assert 64 * rj.start_block == w, f"window {k}: start {64 * rj.start_block} != W {w}"
        assert rj.n_out == e - w, f"window {k}: length {rj.n_out} != E − W {e - w}"
        assert rj.settle_f0 is True, "settle_first=True (TTS default) and k >= 1"
        assert rj.seeds == Seeds.for_window(Seeds(), k)
        assert rj.source == preset.source.patched(f0=preset.speech.f0)
        assert rj.body == preset.body and rj.fs_out == 48000 and rj.oversample == 2
        assert list(rj.gestures) == expected_g[k], f"window {k} must carry exactly the gestures of phrase {k}"
        assert list(rj.commands) == [] and rj.artic is None


def test_render_text_mode(stub_planner):
    job = JOBS[0]
    phrases = prime_durations(stub_planner, job)
    preset = PRESETS["male"]
    jobs, n_total = plan_speech(job["text"], preset, speed=None, g2p=stub_planner.g2p, split=False,
                                settle_first=False)
    assert len(jobs) == 1
    rj = jobs[0]
    plan_end = job["phrases"][-1]["end"]
    assert n_total == rj.n_out == js_round(48000 * (plan_end + 0.3)) == job["n_total"]
    assert (rj.start_block, rj.settle_f0, rj.seeds) == (0, False, Seeds())
    assert rj.source == preset.source.patched(f0=112.0)
    all_g = [g for gs in phrase_gestures(stub_planner, phrases, PlanOptions(preset.body, 0.05, None, 112.0))
             for g in gs]
    assert list(rj.gestures) == all_g


def test_settle_first_false_keeps_first_window_fresh(stub_planner):
    job = JOBS[1]
    prime_durations(stub_planner, job)
    jobs, _ = plan_speech(job["text"], PRESETS["female"], speed=None, g2p=stub_planner.g2p, settle_first=False)
    assert [j.settle_f0 for j in jobs] == [False] + [True] * (len(jobs) - 1)


class ShortPauseG2P:
    """Three phrases; the first boundary has a pause below 0.18 s and must not split."""

    def to_phrases(self, text: str) -> list[Phrase]:
        seg = [Segment("a", stress=1), Segment("b", stress=0)]
        return [Phrase(list(seg), Intonation.STATEMENT, 0.1, ["ab"]),
                Phrase(list(seg), Intonation.STATEMENT, 0.35, ["ab"]),
                Phrase(list(seg), Intonation.STATEMENT, 0.35, ["ab"])]


def test_short_pause_does_not_split(stub_planner):
    jobs, _ = plan_speech("ab ab ab", PRESETS["male"], speed=None, g2p=ShortPauseG2P(), text_prep=False)
    assert len(jobs) == 2
    assert jobs[0].start_block == 0
    labels0 = [g.label for g in jobs[0].gestures]
    assert labels0.count("breath") == 2, "the first window holds the gestures of phrases 0 and 1"


def test_speed_and_preset_rate(stub_planner):
    plan_speech("Hello there.", PRESETS["teen"], speed=None, g2p=stub_planner.g2p)
    assert stub_planner.calls_of("plan_phrases")[-1][1].rate == 1.05
    plan_speech("Hello there.", PRESETS["teen"], speed=1.3, g2p=stub_planner.g2p)
    assert stub_planner.calls_of("plan_phrases")[-1][1].rate == 1.3, "speed is the effective rate (see resolve_speed)"
    plan_speech("Hello there.", PRESETS["narrator"], speed=None, g2p=stub_planner.g2p)
    o = stub_planner.calls_of("plan_phrases")[-1][1]
    assert o.accent == 0.12 and o.f0 == 104.0


def test_text_prep_switch(stub_planner):
    plan_speech("Dr. Smith, 3 cats.", PRESETS["male"], speed=None, g2p=stub_planner.g2p, text_prep=True)
    assert stub_planner.calls_of("prepare_text") == ["Dr. Smith, 3 cats."]
    stub_planner.calls.clear()
    plan_speech("Dr. Smith, 3 cats.", PRESETS["male"], speed=None, g2p=stub_planner.g2p, text_prep=False)
    assert stub_planner.calls_of("prepare_text") == []


@pytest.mark.parametrize("text", ["", "   ", "!!! ...", "123"])
def test_nothing_speakable(stub_planner, text):
    with pytest.raises(TextError):
        plan_speech(text, PRESETS["male"], speed=None, g2p=stub_planner.g2p)


def test_preset_seed(stub_planner):
    job = JOBS[0]
    prime_durations(stub_planner, job)
    p = parse_voice_spec("male;seed=5")
    jobs, _ = plan_speech(job["text"], p, speed=None, g2p=stub_planner.g2p)
    base = Seeds.for_voice(0, 5)
    assert [j.seeds for j in jobs] == [Seeds.for_window(base, k) for k in range(len(jobs))]
    assert len({j.seeds for j in jobs}) == len(jobs)


def test_speaking_f0_default_for_blend(stub_planner):
    from chimera_vocal.core.body import default_f0
    plan_speech("Hello there.", PRESETS["teen"], speed=None, g2p=stub_planner.g2p)
    o = stub_planner.calls_of("plan_phrases")[-1][1]
    assert o.f0 == pytest.approx(default_f0(BodyMix(0.5, 0.5)), rel=1e-15) and o.body == BodyMix(0.5, 0.5)


def _job(k: int, start_block: int, n: int) -> RenderJob:
    return RenderJob(id=f"w{k}", body=BodyMix(), seeds=Seeds(), n_out=n, start_block=start_block)


def test_assemble_overlap_add():
    rng = np.random.default_rng(3)
    jobs = [_job(0, 0, 1000), _job(1, 10, 900), _job(2, 20, 300)]   # offsets 0, 640, 1280
    pieces = [rng.standard_normal(j.n_out).astype(np.float32) for j in jobs]
    out = assemble(pieces, jobs, 1500)
    ref = np.zeros(1500, np.float32)
    for p, j in zip(pieces, jobs):
        o = 64 * j.start_block
        n = min(len(p), 1500 - o)
        ref[o:o + n] = ref[o:o + n] + p[:n]
    assert out.dtype == np.float32 and out.shape == (1500,)
    assert np.array_equal(out, ref), "float32 overlap-add in job order"
    short = assemble(pieces, jobs, 1200)
    assert short.shape == (1200,) and np.array_equal(short, ref[:1200])
    long = assemble(pieces, jobs, 2000)
    assert long.shape == (2000,) and np.all(long[1580:] == 0)
