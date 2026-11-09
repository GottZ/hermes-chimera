"""K-4.1 test_sung_f0: a held sung note hits its pitch: median f0 within ±10 ct of the note
(octave-checked, tests/support/pitch.py) for MIDI 36, 48, 57, 64, 72, 84 per body. FakeVoice always (all notes);
real engine (numpy: three notes per body à 1 s; ``slow``: all). Red proof: midi_to_hz with 432 Hz."""

from __future__ import annotations

import pytest

from support.pitch import cents, hz, sung_f0

ALL = [36, 48, 57, 64, 72, 84]
QUICK = {"male": [48, 57, 64], "female": [57, 64, 72], "child": [57, 64, 72]}
CASES = [(b, m) for b in ("male", "female", "child") for m in ALL]


def one_note(body: str, midi: float) -> dict:
    return {"bpm": 60, "voices": [{"body": body, "vibrato": 0, "notes": [{"pitch": midi, "beats": 1, "lyric": "ah"}]}]}


@pytest.mark.parametrize("body,midi", CASES, ids=[f"{b}-{m}" for b, m in CASES])
def test_sung_f0(pipeline, request, body, midi):
    if pipeline.kind == "real" and midi not in QUICK[body] and not (
            request.config.getoption("--run-slow") or __import__("os").environ.get("CHIMERA_RUN_SLOW") == "1"):
        pytest.skip("slow on the real engine: --run-slow")
    report, jobs, mixes, n_mix, tracks = pipeline.tracks(one_note(body, midi))
    t_beat = report.lead_in
    seg = pipeline.placed(tracks[0], mixes[0], n_mix)[int((t_beat + 0.3) * 48000):int((t_beat + 0.8) * 48000)]
    f = sung_f0(seg, 48000, hz(midi))
    assert abs(cents(f, hz(midi))) <= 10.0, f"{body} MIDI {midi}: {f:.2f} Hz = {cents(f, hz(midi)):+.1f} ct"
