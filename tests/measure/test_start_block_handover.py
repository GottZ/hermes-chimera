"""Start-block handover (supports K-4.4 and K-4.10): the start block reaches the real voice exactly, without
rendering. For every voice of several scores and every phrase window of a speech plan, the voice the production factory
builds (``compile.voice_factory()`` -> ``_human_voice`` -> ``HumanVoice(start_block=...)``) has the clock of a
continuous voice after ``start_block`` blocks (repeated addition, K-3.15), the mix places it at 64·start_block, and a
score voice's start block, pre-roll and length follow the per-voice start rule (start_block = max(0,
floor((t_first - 0.30)·48000/64)), 64·start_block + n_out = js_round(48000·(end + 0.4))).
Red proofs: start block ±1 in ``_human_voice``; pre-roll -0.1 s; n_out = mix length."""

from __future__ import annotations

import math

import pytest

from chimera_vocal.core.jsmath import js_round
from support.fakes import real_engine_available

pytestmark = pytest.mark.skipif(not real_engine_available(), reason="needs the real engine")
DT = 128.0 / 96000.0          # control block at 48 kHz, oversample 2 (= 64 / 48 000 s)
PRE_ROLL = 0.30               # pre-roll of the start-block rule (design value, not the module constant)
TAIL = 0.4                    # render.rs:561

ROW = [("D4", "Row,", 1), ("D4", "row,", 1), ("D4", "row", 1), ("E4", "your", 1), ("F#4", "boat", 1)]
SCORES = {
    "ta-canon-120": {"bpm": 120, "voices": [{"body": "female", "start": s, "seed": 0,
                                             "notes": [{"pitch": p, "beats": 1, "lyric": "ta"} for p in (62, 64, 65, 67)]}
                                            for s in (0, 8, 16)]},
    "row-canon-84": {"bpm": 84, "voices": [{"body": b, "start": s,
                                            "notes": [{"pitch": p, "beats": n, "lyric": l} for p, l, n in ROW]}
                                           for b, s in (("female", 0), ("male", 8), ("female", 16))]},
    "odd-entries-97": {"bpm": 97, "voices": [
        {"body": "child", "start": 0, "notes": [{"pitch": 67, "beats": 2, "lyric": "strength"}]},
        {"body": "male", "start": 3.5, "notes": [{"pitch": 48, "beats": 1.5, "lyric": "ah"}]},
        {"body": "female", "start": 0, "notes": [{"rest": 5.25}, {"pitch": 64, "beats": 1, "lyric": "splash"}]},
        {"body": "female=0.5", "start": 7.75, "notes": [{"say": "Hello there.", "beats": 4}]}]},
}


def continuous_clock(k: int) -> float:
    c = 0.0
    for _ in range(k):
        c += DT
    return c


@pytest.fixture
def real():
    from chimera_vocal.music.compile import set_voice_factory, voice_factory
    from chimera_vocal.music.planning import planner_api, set_planner_api
    set_planner_api(None)
    prev = set_voice_factory(None)
    try:
        yield planner_api().default_g2p(), voice_factory()
    finally:
        set_voice_factory(prev)


@pytest.mark.parametrize("name", sorted(SCORES))
def test_score_voices_get_their_start_block(real, name):
    from chimera_vocal.music.compile import compile_normalized, compile_score
    from chimera_vocal.music.schema import validate_score
    g2p, factory = real
    report = validate_score(SCORES[name], g2p=g2p)
    assert report.ok, report.issues
    plans = compile_normalized(report.normalized, g2p=g2p)
    jobs, mixes, n_mix = compile_score(report, g2p=g2p)
    assert any(j.start_block > 0 for j in jobs), "the score has late voices"
    assert n_mix == max(m.offset + j.n_out for j, m in zip(jobs, mixes))
    for vp, job, vm in zip(plans, jobs, mixes):
        t_first = min(g.t0 - g.ramp_in for g in job.gestures)
        sb = max(0, math.floor((t_first - PRE_ROLL) * 48000 / 64))
        assert job.start_block == sb, f"{job.id}: start block {job.start_block}, the start-block rule gives {sb}"
        # the voice starts at least the pre-roll before its first gesture and less than one block earlier than that
        if sb:
            assert t_first - PRE_ROLL - DT < 64 * sb / 48000 <= t_first - PRE_ROLL, job.id
        assert vm.offset == 64 * job.start_block, f"{job.id}: placement"
        assert job.settle_f0 is (job.start_block > 0), f"{job.id}: settle"
        assert 64 * job.start_block + job.n_out == js_round(48000 * (vp.end + TAIL)), f"{job.id}: length"
        v = factory(job, None, None)
        assert v.clock == continuous_clock(job.start_block), \
            f"{job.id}: voice clock {v.clock!r}, continuous voice after {job.start_block} blocks {continuous_clock(job.start_block)!r}"


@pytest.mark.parametrize("body,settle_first", [("male", False), ("female", True), ("child", True)])
def test_speech_windows_get_their_start_block(real, body, settle_first):
    from chimera_vocal.presets import PRESETS
    from chimera_vocal.speech import plan_speech
    g2p, factory = real
    jobs, n_total = plan_speech("The quick brown fox. It jumps over the lazy dog, and then it rests. Good night!",
                                PRESETS[body], speed=None, g2p=g2p, text_prep=False, split=True,
                                settle_first=settle_first)
    assert len(jobs) >= 3 and all(j.start_block > 0 for j in jobs[1:])
    for k, job in enumerate(jobs):
        assert job.settle_f0 is (k > 0 or settle_first)
        v = factory(job, None, None)
        assert v.clock == continuous_clock(job.start_block), f"window {k}: clock {v.clock!r}"
    assert 64 * jobs[-1].start_block + jobs[-1].n_out == n_total
