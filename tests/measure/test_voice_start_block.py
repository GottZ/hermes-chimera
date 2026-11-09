"""K-4.10 test_voice_start_block (TESTPLAN L4, P1-06): in a canon (entries at beats 0/8/16, 84 bpm) the third voice
renders at least 11 s less than from sample 0, and its result matches rendering it from 0 within the K-4.8b band
(envelope p95 <= 2.5 dB, f0 p90 <= 18 ct, onset f0 p90 <= 20 ct, third-octave RMS <= 0.3 dB).
Red proof: n_out = mix length."""

from __future__ import annotations

import dataclasses

import pytest

from support.pitch import compare_split

BPM = 84


def canon(vibrato: float | None = None) -> dict:
    tune = [("D4", "Row,"), ("D4", "row,"), ("D4", "row"), ("E4", "your"), ("F#4", "boat"), ("F#4", "gent-"),
            ("E4", "ly"), ("F#4", "down"), ("G4", "the"), ("A4", "stream.")]
    notes = [{"pitch": p, "beats": 1, "lyric": l} for p, l in tune]
    extra = {} if vibrato is None else {"vibrato": vibrato}
    return {"bpm": BPM, "voices": [{"body": b, "start": s, "notes": notes, **extra}
                                   for b, s in (("female", 0), ("male", 8), ("female", 16))]}


def test_voice_start_block(pipeline):
    # real engine without vibrato: the vibrato's phase is part of the random realisation - the template rendered with
    # six other seed sets differs from itself by f0 p90 58 ct (max 62) at the default 35 ct vibrato, so the speech
    # band (18 ct) cannot judge it; without vibrato the start-block render measures 11 ct against the render from 0
    report, jobs, mixes, n_mix = pipeline.compile(canon(0 if pipeline.kind == "real" else None))
    late, vm = jobs[2], mixes[2]
    assert vm.offset == 64 * late.start_block and late.settle_f0 is True
    # voice 3 sings voice 1's tune (same body) 16 beats later: rendered from 0 it would be about
    # n_out(voice 1) + 16·60/84 s long; from its start block it must be at least 11 s shorter
    from_zero = jobs[0].n_out + 16 * 60 / BPM * 48000
    assert (from_zero - late.n_out) / 48000 >= 11.0 - 1e-3, "voice 3 renders at least 11 s less"
    full_n = late.n_out + 64 * late.start_block
    got = pipeline.placed(pipeline.render(late), vm, n_mix)[:full_n]
    ref = pipeline.render(dataclasses.replace(late, start_block=0, settle_f0=False, n_out=full_n))
    t_first = report.lead_in + 16 * 60 / BPM
    m = compare_split(ref, got, onsets_s=[t_first], bands="active40" if pipeline.kind == "fake" else "all")
    print(m)
    assert m.env_p95_db <= 2.5 and m.f0_p90_ct <= 18 and m.f0_onset_p90_ct <= 20 and m.third_rms_db <= 0.3, m
