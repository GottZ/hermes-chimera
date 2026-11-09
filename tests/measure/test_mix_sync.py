"""K-4.4 test_mix_sync: three voices singing /t/ syllables on the same beats have their onsets on the
same mix sample (0 samples apart); canon entries lie exactly start·60/bpm·48 000 samples later; a voice rendered from
start_block > 0 lands where it belongs. Measured on the mix timeline (each voice alone through mix()).

FakeVoice: every check is sample-exact. Real engine: a threshold onset is the first loud glottal pulse after /t/, and
the glottal oscillator runs through the silence before an entry, so the template itself puts the canon entries
109/122 samples after start·60/bpm·48 000. The real engine is therefore checked against the template, exactly where
the template defines the result:
- plan (gestures of voice k shifted by exactly start·60/bpm), start block (``spec_start_block``) and placement
  (offset 64·start_block): exact;
- template mode (start_block 0, no settle - what ``render notes`` does) against native renders of the same voices
  (``data/canon-native.json``): length and every onset exact, the samples around the first onset T3;
- a late voice (rendered from its own start block with a fresh glottis) against the same job from block 0: the
  control track (gestures, clock, articulators, tract, noise sources, glottis parameters) row for row at the same
  absolute blocks - bit-equal except the glide f0 (<= 1e-12 rel); its acoustic onsets within one glottal period of the
  native onsets (the glottal phase at the entry is the one acoustic difference a late start allows).
Red proofs: offset applied twice; start block handed to the voice ±1 block; pre-roll -0.1 s; plan shifted one block."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from support.fakes import real_engine_available
from support.pitch import hz, onsets

BPM = 120   # 1 beat = 0.5 s = 24 000 samples
PITCHES = (62, 64, 65, 67)
NATIVE = json.loads((Path(__file__).resolve().parent / "data" / "canon-native.json").read_text(encoding="utf-8"))
ENTRY = {e["start_beats"]: e for e in NATIVE["entries"]}


def ta_voice(start: int, body: str = "female") -> dict:
    return {"body": body, "start": start, "seed": 0,
            "notes": [{"pitch": p, "beats": 1, "lyric": "ta"} for p in PITCHES]}


def ons(x) -> list[int]:
    on = onsets(x, threshold=0.2, quiet=4800)
    assert on, "no onset found"
    return on


def first_onsets(pipeline, score: dict) -> tuple[list[int], list, int]:
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(score)
    return [ons(pipeline.placed(t, vm, n_mix))[0] for t, vm in zip(tracks, mixes)], jobs, n_mix


def assert_native_region(x: np.ndarray, entry: dict, label: str) -> None:
    """T3 on the stored native samples around the first onset (max |Δ| <= 1e-6 × native peak)."""
    ref = np.asarray(entry["region"], dtype=np.float64).astype(np.float32).astype(np.float64)
    a = entry["region_start"]
    got = np.asarray(x[a:a + ref.size], dtype=np.float32).astype(np.float64)
    assert got.size == ref.size, f"{label}: render ends before the native region"
    d = float(np.max(np.abs(got - ref))) / entry["peak"]
    assert d <= 1e-6, f"{label}: samples around the first onset differ from the native render by {d:.3e} x peak"


def assert_within_period(got: list[int], entry: dict, label: str) -> None:
    """Every onset of a late voice within one glottal period (of that note's pitch) of the native onset."""
    nat = entry["onsets"]
    assert len(got) == len(nat), f"{label}: {len(got)} onsets, native {len(nat)}"
    for k, (a, b, p) in enumerate(zip(got, nat, PITCHES)):
        period = 48000 / hz(p)
        assert abs(a - b) <= period, f"{label}: onset {k} at {a}, native {b} (> one period {period:.0f})"


def test_same_beats_same_sample(pipeline):
    outs, jobs, _ = first_onsets(pipeline, {"bpm": BPM, "voices": [ta_voice(0), ta_voice(0), ta_voice(0)]})
    assert outs[0] == outs[1] == outs[2], f"onsets {outs}"


def test_canon_entries_exact(pipeline):
    starts = [0, 8, 16]
    report, jobs, mixes, n_mix, tracks = pipeline.tracks({"bpm": BPM, "voices": [ta_voice(s) for s in starts]})
    assert jobs[1].start_block > 0 and jobs[2].start_block > jobs[1].start_block, "late voices start late"
    for j, vm in zip(jobs, mixes):
        assert j.start_block == pipeline.spec_start_block(j), f"{j.id}: start block vs spec_start_block"
        assert vm.offset == 64 * j.start_block, f"{j.id}: placement"
        assert j.settle_f0 is (j.start_block > 0)
    placed = [pipeline.placed(t, vm, n_mix) for t, vm in zip(tracks, mixes)]
    if pipeline.kind == "fake":
        outs = [ons(x)[0] for x in placed]
        for s, o in zip(starts[1:], outs[1:]):
            assert o - outs[0] == s * 60 * 48000 // BPM, f"entry at beat {s}: {o - outs[0]} samples after voice 0"
        return
    for s, j in zip(starts[1:], jobs[1:]):
        dt = [g.t0 - g0.t0 for g, g0 in zip(j.gestures, jobs[0].gestures)]
        assert len(j.gestures) == len(jobs[0].gestures)
        assert max(abs(d - s * 60 / BPM) for d in dt) <= 1e-9, f"entry at beat {s}: plan not shifted exactly"
    assert report.lead_in == ENTRY[0]["t0"], "native data were rendered with this lead-in"
    # voice 0 starts at block 0: it is the template's render, sample for sample
    assert jobs[0].start_block == 0 and jobs[0].n_out == ENTRY[0]["n"]
    assert ons(placed[0]) == ENTRY[0]["onsets"], "voice 0 onsets vs the native render"
    assert_native_region(placed[0], ENTRY[0], "voice 0")
    for s, j, x in zip(starts[1:], jobs[1:], placed[1:]):
        bad = pipeline.late_track_mismatches(j)
        assert not bad, f"entry at beat {s}: control track vs the same voice from block 0: {bad}"
        assert_within_period(ons(x), ENTRY[s], f"entry at beat {s}")


@pytest.mark.skipif(not real_engine_available(), reason="needs the real engine")
@pytest.mark.parametrize("kernel", [pytest.param("numpy", marks=pytest.mark.slow),
                                    pytest.param("numba", marks=pytest.mark.numba)])
def test_canon_entries_match_template(kernel, make_scheduler, monkeypatch):
    """Template mode (start_block 0, no settle) of all three canon voices through a real render worker = the native
    renders: length, every onset (so the entries +192 109 / +384 122 samples) and the samples around the first onset."""
    import threading

    from chimera_vocal.music.compile import compile_score, set_voice_factory
    from chimera_vocal.music.planning import planner_api, set_planner_api
    from chimera_vocal.music.schema import validate_score
    from chimera_vocal.scheduler import MemorySink, Request
    monkeypatch.delenv("CHIMERA_WORKER_VOICE_FACTORY", raising=False)
    set_planner_api(None)
    set_voice_factory(None)
    g2p = planner_api().default_g2p()
    report = validate_score({"bpm": BPM, "voices": [ta_voice(s) for s in (0, 8, 16)]}, g2p=g2p)
    jobs, mixes, n_mix = compile_score(report, g2p=g2p)
    assert report.lead_in == ENTRY[0]["t0"]
    sched = make_scheduler(workers=3)
    zero = [dataclasses.replace(j, id=f"{j.id}-z", start_block=0, settle_f0=False, n_out=j.n_out + 64 * j.start_block)
            for j in jobs]
    sinks = [MemorySink(j.n_out) for j in zero]
    futs = [sched.submit(Request(j.id, "music", kernel), j, sink=s, cancel=threading.Event())
            for j, s in zip(zero, sinks)]
    for f in futs:
        assert f.result(timeout=3600).kernel == kernel
    first = []
    for s, j, sink in zip((0, 8, 16), zero, sinks):
        x, e = sink.data, ENTRY[s]
        assert len(x) == e["n"], f"beat {s}: length {len(x)} vs native {e['n']}"
        assert ons(x) == e["onsets"], f"beat {s}: onsets {ons(x)} vs native {e['onsets']}"
        assert_native_region(x, e, f"beat {s}")
        first.append(ons(x)[0])
    assert [o - first[0] for o in first] == [0, 192109, 384122], "the template's own canon entries"


def test_late_voice_sample_exact(pipeline):
    """The voice entering at beat 16 rendered from its start block against the same voice from block 0: FakeVoice
    puts the first onset on the same sample; the real engine has the same control track at the same absolute blocks
    and its onsets within one glottal period of the template's (the render from block 0 = the native render, see
    test_canon_entries_match_template)."""
    report, jobs, mixes, n_mix, tracks = pipeline.tracks({"bpm": BPM, "voices": [ta_voice(0), ta_voice(16)]})
    late, vm = jobs[1], mixes[1]
    assert late.start_block > 0 and vm.offset == 64 * late.start_block
    assert late.start_block == pipeline.spec_start_block(late)
    a = ons(pipeline.placed(tracks[1], vm, n_mix))
    if pipeline.kind == "real":
        assert report.lead_in == ENTRY[0]["t0"], "native data were rendered with this lead-in"
        bad = pipeline.late_track_mismatches(late)
        assert not bad, f"control track vs the same voice from block 0: {bad}"
        assert_within_period(a, ENTRY[16], "late voice")
        return
    full = dataclasses.replace(late, start_block=0, settle_f0=False, n_out=late.n_out + 64 * late.start_block)
    b = ons(pipeline.render(full))
    assert a[0] == b[0], f"late render onset {a[0]} vs render from 0: {b[0]}"
