"""K-4.8 test_phrase_split_equivalence: speech rendered in phrase windows (``chimera_vocal.speech``: planned once,
split at pauses, one fresh voice per window, settled f0 for k >= 1) equals the continuous voice (a) bit-exactly
before W_1 and (b) overall within the template's own spread (the template against itself with another random
phase, from the phrase-split study on the native template, ``tests/engine/data/e1a_phrase_split.json``):
envelope p95 <= 2.5 dB, f0 p90 <= 18 ct, f0 onset p90 <= 20 ct, third-octave RMS <= 0.3 dB, voicing disagreement
<= 0.032 (the study's band maximum; f0 is measured over frames voiced in both, so lost voicing shows only there;
measured 0.006-0.019 on the real engine). FakeVoice with the study's phrase timing always; the real engine also
with child and rate 1.3. Red proof: no settle_f0 for k >= 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chimera_vocal.presets import PRESETS
from chimera_vocal.speech import assemble, plan_speech
from support.pitch import compare_split

DATA = json.loads((Path(__file__).resolve().parents[1] / "engine" / "data" / "e1a_phrase_split.json")
                  .read_text(encoding="utf-8"))
CASES = [("t1-male", "male", None), ("t3-female", "female", None)]
S3_EXTRA = [("t2-female", "child", None), ("t4-male", "male", 1.3)]


def prime(pipeline, tag: str) -> dict:
    job = next(j for j in DATA["jobs"] if j["tag"] == tag)
    if pipeline.planner is not None:
        phrases = pipeline.g2p.to_phrases(job["text"])
        for ph, ref in zip(phrases, job["phrases"]):
            pipeline.planner.durations["".join(s.ipa for s in ph.segments)] = ref["end"] - ref["t0"]
    return job


@pytest.mark.parametrize("tag,body,speed", CASES + S3_EXTRA, ids=[c[0] + "-" + c[1] for c in CASES + S3_EXTRA])
def test_phrase_split_equivalence(pipeline, tag, body, speed):
    if pipeline.kind == "fake" and (tag, body, speed) in S3_EXTRA:
        pytest.skip("extra cases (child, rate 1.3) measure the real engine")
    job = prime(pipeline, tag)
    preset = PRESETS[body]
    cont_jobs, n_total = plan_speech(job["text"], preset, speed=speed, g2p=pipeline.g2p, text_prep=False,
                                     split=False, settle_first=False)
    split_jobs, n2 = plan_speech(job["text"], preset, speed=speed, g2p=pipeline.g2p, text_prep=False, split=True,
                                 settle_first=False)
    assert n2 == n_total and len(split_jobs) >= 2
    cont = pipeline.render(cont_jobs[0])
    split = assemble([pipeline.render(j) for j in split_jobs], split_jobs, n_total)
    w1 = 64 * split_jobs[1].start_block
    assert np.array_equal(split[:w1], cont[:w1]), "(a) samples before W_1 equal the continuous voice"
    onsets_s = [64 * j.start_block / 48000 + 0.05 for j in split_jobs[1:]]
    # the real engine is measured over all 21 third-octave bands (the template study's definition); FakeVoice only
    # over its active bands - its harmonic spectrum leaves the low bands nearly empty (window leakage only)
    bands = "active40" if pipeline.kind == "fake" else "all"
    m = compare_split(cont, split, onsets_s=onsets_s, fmin=60, fmax=500, bands=bands)
    print(f"{tag}/{body}: {m}")
    assert m.env_p95_db <= 2.5, m
    assert m.f0_p90_ct <= 18.0, m
    assert m.f0_onset_p90_ct <= 20.0, m
    assert m.third_rms_db <= 0.3, m
    assert m.voicing_disagree <= 0.032, m
