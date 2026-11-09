"""K-5.2 test_compile_matches_render_notes (TESTPLAN L5): a score equivalent to a ``chimera render notes`` input
compiles to exactly the job the CLI renders (render.rs:481-575): same source, same gestures (plan_song per phrase,
phrases split at rests, phrase start = time of its first note, t accumulated note by note), same length
js_round(48000·(end + 0.4)), start_block 0, template seeds. Red proof: split phrases at phraseEnd instead of rests.

Inputs: copies of the template-CLI inputs used for the reference renders (``data/render-notes-*.json`` =
lab/refs/ref-audio/inputs/*.json, recon/ref-audio/manifest.md; sha256 equal). The ``stub`` planner runs in S1/S2,
the ``real`` planner (C1) once the clusters are merged (S3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.jsmath import js_max, js_round, midi_to_hz
from chimera_vocal.core.types import Seeds, SongOptions, SourceParams, SungNote
from chimera_vocal.music.compile import RenderJob, compile_score
from chimera_vocal.music.planning import planner_api, set_planner_api
from chimera_vocal.music.schema import validate_score
from support.fakes import real_engine_available

DATA = Path(__file__).parent / "data"
CASES = [("melody-lyrics", "male"), ("melody-lyrics", "female"), ("melody-lyrics", "child"),
         ("notes-ah", "male"), ("notes-ah", "female"), ("duet-upper", "female"), ("duet-lower", "male")]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load(name: str) -> dict:
    return json.loads((DATA / f"render-notes-{name}.json").read_text(encoding="utf-8"))


def score_from_render_notes(j: dict, body: str, *, names: bool = False) -> dict:
    """The score form of a render-notes input: template lyrics string, one voice, lead_in = t0 (default 0.3)."""
    notes = []
    for midi, beats in j["notes"]:
        if midi > 0:
            m = int(midi)
            pitch = f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" if names and m == midi else midi
            notes.append({"pitch": pitch, "beats": beats})
        else:
            notes.append({"rest": beats})
    return {"bpm": j["bpm"], "lead_in": j.get("t0", 0.3),
            "voices": [{"body": body, "lyrics": j["lyrics"], "vibrato": j.get("vibrato", 35.0), "notes": notes}]}


def reference_render_notes(j: dict, body: str, api, g2p):
    """render.rs:481-575 line by line: (source, gestures, n, plan_song calls)."""
    bpm = float(j["bpm"])
    vibrato = float(j.get("vibrato", 35.0))
    raw = j["notes"]
    syl = api.lyrics_to_syllables(j["lyrics"], g2p)
    assert len(syl) == sum(1 for n in raw if n[0] > 0)
    t0 = float(j.get("t0", 0.3))
    beat = 60.0 / bpm
    first = next(n for n in raw if n[0] > 0)
    source = SourceParams().patched(f0=midi_to_hz(first[0]), voiced=True, pressure=9.0, tension=0.55)
    gestures, calls = [], []
    state = {"end": 0.0}
    phrase: list = []

    def flush(start: float) -> None:
        if not phrase:
            return
        opts = SongOptions(BodyMix.parse(body), bpm, start, vibrato)
        plan = api.plan_song(list(phrase), opts)
        calls.append((list(phrase), opts))
        gestures.extend(plan.gestures)
        state["end"] = js_max(state["end"], plan.end)
        phrase.clear()

    t, k, start = t0, 0, t0
    for midi, beats in raw:
        if midi > 0:
            if not phrase:
                start = t
            phrase.append(SungNote(float(midi), float(beats), syl[k]))
            k += 1
        else:
            flush(start)
        t += beats * beat
    flush(start)
    return source, gestures, js_round(48000.0 * (state["end"] + 0.4)), calls


@pytest.fixture(params=["stub", "real"])
def planner(request):
    if request.param == "stub":
        return request.getfixturevalue("stub_planner")
    if not real_engine_available():
        pytest.skip("real planner (C1) not merged yet - runs in S3")
    set_planner_api(None)
    return None


@pytest.mark.parametrize("names", [False, True], ids=["midi", "note-names"])
@pytest.mark.parametrize("name,body", CASES, ids=[f"{n}-{b}" for n, b in CASES])
def test_compile_matches_render_notes(planner, name, body, names):
    api = planner_api()
    g2p = api.default_g2p()
    j = load(name)
    report = validate_score(score_from_render_notes(j, body, names=names), g2p=g2p)
    assert report.ok, report.issues
    assert report.lead_in == j.get("t0", 0.3), "lead_in must not be raised for these inputs"

    if planner is not None:
        planner.calls.clear()
    jobs, mixes, n_mix = compile_score(report, g2p=g2p)
    got_calls = list(planner.calls_of("plan_song")) if planner is not None else None
    source, gestures, n, calls = reference_render_notes(j, body, api, g2p)

    assert len(jobs) == 1 and len(mixes) == 1
    job = jobs[0]
    assert isinstance(job, RenderJob)
    assert job.source == source
    assert list(job.gestures) == gestures
    assert job.n_out == n and n_mix == n
    assert (job.start_block, job.settle_f0, job.fs_out, job.oversample) == (0, False, 48000, 2)
    assert job.seeds == Seeds(), "voice 0 with seed null renders with the template seeds"
    assert job.body == BodyMix.parse(body)
    assert job.artic is None and list(job.commands) == []
    assert mixes[0].offset == 0 and mixes[0].gain_db == 0 and mixes[0].pan == 0 and mixes[0].gain_env is None
    if got_calls is not None:
        assert [(list(n_), o) for n_, o in got_calls] == calls, "plan_song calls differ from render notes"


def test_render_job_json_roundtrip(stub_planner):
    api = planner_api()
    report = validate_score(score_from_render_notes(load("melody-lyrics"), "female"), g2p=api.default_g2p())
    jobs, _, _ = compile_score(report, g2p=api.default_g2p())
    d = jobs[0].to_json()
    assert json.loads(json.dumps(d)) == d
    assert RenderJob.from_json(d) == jobs[0]
