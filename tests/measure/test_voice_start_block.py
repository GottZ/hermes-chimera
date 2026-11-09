"""K-4.10 test_voice_start_block (TESTPLAN L4, P1-06, N-7): in a canon (entries at beats 0/8/16, 84 bpm) the third
voice renders at least 11 s less than from sample 0, and its result matches rendering it from 0.

The canon sings "Row, row, row your boat ..." one beat per note with the last note held three beats, so the default
vibrato (35 ct, it starts 0.22 s into a vowel) is fully there for 1.3 s. Checks:
- structure (fake + real): start block by the P1-06 formula, placement, settle, >= 11 s less;
- control track (real): the late voice's TrackChunk rows equal the same job's rows from block 0 at the same absolute
  blocks, bit for bit except the glide f0 (<= 1e-12 rel) - exact, phase-independent, catches a start block or pre-roll
  off by one block;
- audio with the default vibrato: FakeVoice within the K-4.8b band; real engine within the template's spread against
  itself (``data/k410-band.json``: the from-0 render with 12 other seed sets vs the job's seeds, maximum per metric):
  envelope p95, f0 p90, onset f0 p90, third-octave RMS, voicing disagreement, and the vibrato's extent and rate on the
  held note (phase-independent) - the vibrato's phase is part of the random realisation (N-7), its depth and rate are
  not;
- without vibrato (real, slow, additional case): the K-4.8b speech band.
Red proofs: n_out = mix length; start block ±1 / pre-roll -0.1 s; no settle_pitch; no vibrato in late voices."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from support.pitch import compare_split, vibrato_stats

BPM = 84
BEAT = 60 / BPM
TUNE = [("D4", "Row,", 1), ("D4", "row,", 1), ("D4", "row", 1), ("E4", "your", 1), ("F#4", "boat", 1),
        ("F#4", "gent-", 1), ("E4", "ly", 1), ("F#4", "down", 1), ("G4", "the", 1), ("A4", "stream.", 3)]
BAND = json.loads((Path(__file__).resolve().parent / "data" / "k410-band.json").read_text(encoding="utf-8"))
K48B = {"env_p95_db": 2.5, "f0_p90_ct": 18.0, "f0_onset_p90_ct": 20.0, "third_rms_db": 0.3, "voicing_disagree": 0.032}


def canon(vibrato: float | None = None) -> dict:
    notes = [{"pitch": p, "beats": b, "lyric": l} for p, l, b in TUNE]
    extra = {} if vibrato is None else {"vibrato": vibrato}
    return {"bpm": BPM, "voices": [{"body": b, "start": s, "notes": notes, **extra}
                                   for b, s in (("female", 0), ("male", 8), ("female", 16))]}


def late_and_ref(pipeline, vibrato: float | None):
    """(report, late job, render of the late job placed on the mix timeline, render of the same job from block 0)."""
    report, jobs, mixes, n_mix = pipeline.compile(canon(vibrato))
    late, vm = jobs[2], mixes[2]
    full_n = late.n_out + 64 * late.start_block
    got = pipeline.placed(pipeline.render(late), vm, n_mix)[:full_n]
    ref = pipeline.render(dataclasses.replace(late, start_block=0, settle_f0=False, n_out=full_n))
    return report, late, got, ref


def held_span(report) -> list[tuple[float, float]]:
    t_held = report.lead_in + (16 + 9) * BEAT
    a, b = BAND["span_rel_held_note_s"]
    return [(t_held + a, t_held + b)]


def test_voice_start_block_structure(pipeline):
    report, jobs, mixes, n_mix = pipeline.compile(canon())
    late, vm = jobs[2], mixes[2]
    assert late.start_block == pipeline.spec_start_block(late) > 0, "start block by the P1-06 formula"
    assert vm.offset == 64 * late.start_block and late.settle_f0 is True
    assert n_mix == max(m.offset + j.n_out for j, m in zip(jobs, mixes))
    # voice 3 sings voice 1's tune (same body) 16 beats later: rendered from 0 it would be about
    # n_out(voice 1) + 16·60/84 s long; from its start block it must be at least 11 s shorter
    from_zero = jobs[0].n_out + 16 * BEAT * 48000
    assert (from_zero - late.n_out) / 48000 >= 11.0 - 1e-3, "voice 3 renders at least 11 s less"


def test_voice_start_block_control_track(pipeline):
    if pipeline.kind == "fake":
        pytest.skip("the control track is the real engine's (FakeVoice has none)")
    report, jobs, mixes, n_mix = pipeline.compile(canon())
    bad = pipeline.late_track_mismatches(jobs[2])
    assert not bad, f"late voice vs the same voice from block 0: {bad}"


def test_voice_start_block(pipeline):
    """Default vibrato: FakeVoice within the K-4.8b band; real engine within the template's spread against itself."""
    report, late, got, ref = late_and_ref(pipeline, None)
    t_first = report.lead_in + 16 * BEAT
    m = compare_split(ref, got, onsets_s=[t_first], bands="active40" if pipeline.kind == "fake" else "all")
    print(m)
    if pipeline.kind == "fake":
        for k, lim in K48B.items():
            assert getattr(m, k) <= lim, (k, m)
        return
    limits = {k: max(d[k] for d in BAND["draws"]) for k in K48B}
    over = {k: (getattr(m, k), lim) for k, lim in limits.items() if getattr(m, k) > lim}
    assert not over, f"outside the template's spread (value, band max): {over}"
    span = held_span(report)
    vr, vg = vibrato_stats(ref, span), vibrato_stats(got, span)
    print(f"vibrato ref {vr} late {vg}")
    assert vr.extent_ct > 25, f"the held note carries the default vibrato in the render from 0: {vr}"
    d_ext, d_rate = abs(vg.extent_ct - vr.extent_ct), abs(vg.rate_hz - vr.rate_hz)
    lim_ext = max(d["vib_extent_delta_ct"] for d in BAND["draws"])
    lim_rate = max(d["vib_rate_delta_hz"] for d in BAND["draws"])
    assert d_ext <= lim_ext, f"vibrato extent {vg.extent_ct:.2f} vs {vr.extent_ct:.2f} ct (band {lim_ext:.2f})"
    assert d_rate <= lim_rate, f"vibrato rate {vg.rate_hz:.3f} vs {vr.rate_hz:.3f} Hz (band {lim_rate:.3f})"


@pytest.mark.slow
def test_voice_start_block_no_vibrato(pipeline):
    """Additional case: without vibrato the late voice stays within the K-4.8b speech band."""
    if pipeline.kind == "fake":
        pytest.skip("FakeVoice is covered by test_voice_start_block")
    report, late, got, ref = late_and_ref(pipeline, 0.0)
    m = compare_split(ref, got, onsets_s=[report.lead_in + 16 * BEAT], bands="all")
    print(m)
    for k, lim in K48B.items():
        assert getattr(m, k) <= lim, (k, m)
