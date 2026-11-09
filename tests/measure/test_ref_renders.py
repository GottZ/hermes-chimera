"""K-4.3 test_ref_renders (TESTPLAN L4, S3 mandatory, ``lab``, both kernels): the port renders the template's
reference files (``$CHIMERA_LAB_REFS/ref-audio``, native CLI renders of HEAD 092b16ce12b9) in the template-faithful mode (``settle_first=False``,
``start_block = 0``) to T3 over the WHOLE file (max |Δ| <= 1e-6 × peak, length exact). Renders go through a real
render worker (RenderScheduler) for both kernels. Red proof: tail 0.4 s instead of 0.3 s for ``text``."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.jsmath import js_round, midi_to_hz
from chimera_vocal.core.types import Seeds, SongOptions, SourceParams, SungNote, Syllable
from chimera_vocal.core.wavio import read_wav
from support import REPO_ROOT, assert_t3, golden
from support.fakes import real_engine_available

pytestmark = [pytest.mark.lab,
              pytest.mark.skipif(not real_engine_available(), reason="real engine (C1–C3) - S3")]
DATA = Path(__file__).resolve().parents[1] / "music" / "data"
SPEECH_TEXT = "Hello there, my name is Chimera. Can you hear me?"
VOWELS = ["i", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "u", "ʌ", "ɝ"]
VOWEL_F0 = {"male": 110.0, "female": 205.0, "child": 260.0}


@pytest.fixture
def worker_render(kernel_name, make_scheduler, monkeypatch):
    """Render one job in a real worker with the requested kernel."""
    monkeypatch.delenv("CHIMERA_WORKER_VOICE_FACTORY", raising=False)
    from chimera_vocal.music.planning import set_planner_api
    from chimera_vocal.music.compile import set_voice_factory
    set_planner_api(None)
    set_voice_factory(None)
    sched = make_scheduler(workers=2)

    def render(job):
        from chimera_vocal.scheduler import MemorySink, Request
        sink = MemorySink(job.n_out)
        stats = sched.submit(Request(job.id, "music", kernel_name), job, sink=sink,
                             cancel=threading.Event()).result(timeout=3600)
        assert stats.kernel == kernel_name
        return sink.data
    return render


def ref(lab_refs, rel: str):
    sr, x = read_wav(lab_refs / "ref-audio" / rel)
    assert sr == 48000
    return x


@pytest.mark.parametrize("body", ["male", "female", "child"])
def test_text(lab_refs, worker_render, body):
    from chimera_vocal.music.planning import planner_api
    from chimera_vocal.presets import PRESETS
    from chimera_vocal.speech import plan_speech
    jobs, n_total = plan_speech(SPEECH_TEXT, PRESETS[body], speed=None, g2p=planner_api().default_g2p(),
                                text_prep=False, split=False, settle_first=False)
    assert len(jobs) == 1 and n_total == jobs[0].n_out
    assert_t3(ref(lab_refs, f"text-{body}.wav"), worker_render(jobs[0]), f"text-{body}")


@pytest.mark.parametrize("body", ["male", "female", "child"])
def test_vowels(lab_refs, worker_render, body):
    """render.rs:785-877: source f0, per vowel 0.6 s: articulation (tip_free) and gate 1, gate 0 from 0.48 s."""
    from chimera_vocal.music.compile import Command, RenderJob
    table = json.loads((REPO_ROOT / "chimera_vocal" / "data" / "vowel_targets.json").read_text(encoding="utf-8"))
    n_seg = js_round(48000 * 0.6)
    cmds = []
    for k, ipa in enumerate(VOWELS):
        cmds.append(Command(k * n_seg, "artic", {"patch": table[body][ipa]["artic"], "tip_free": True}))
        cmds.append(Command(k * n_seg, "gate", {"level": 1.0}))
        cmds.append(Command(k * n_seg + 23040, "gate", {"level": 0.0}))   # first 128-block with local >= 0.48 s
    job = RenderJob(id=f"vowels-{body}", body=BodyMix.parse(body), seeds=Seeds(),
                    source=SourceParams().patched(f0=VOWEL_F0[body]), commands=cmds, n_out=js_round(48000 * 6.0))
    assert_t3(ref(lab_refs, f"vowels-{body}.wav"), worker_render(job), f"vowels-{body}")


SONGS = {"made-of-air": "Made of air", "twinkle-twinkle": "Twinkle, twinkle"}


@pytest.mark.parametrize("body", ["male", "female", "child"])
@pytest.mark.parametrize("slug", list(SONGS))
def test_songs(lab_refs, worker_render, body, slug):
    """render.rs song: notes of the demo song (golden planning.json, same octave per body), t0 0.3 (the golden
    case uses 0.35), pressure 9, tension 0.55, vibrato 35, n = js_round(fs·(end + 0.4)). [Hypothese: the golden
    notes equal DEMO_SONGS with the body's octave - the length check below decides first.]"""
    from chimera_vocal.music.compile import RenderJob
    from chimera_vocal.music.planning import planner_api
    case = next(s for s in golden("planning")["songs"] if s["name"] == f"{SONGS[slug]} ({body})")
    notes = [SungNote.from_json(n) for n in case["notes"]]
    opts = SongOptions(BodyMix.parse(body), float(case["opts"]["bpm"]), 0.3, 35.0)
    plan = planner_api().plan_song(notes, opts)
    job = RenderJob(id=f"{slug}-{body}", body=BodyMix.parse(body), seeds=Seeds(),
                    source=SourceParams().patched(f0=midi_to_hz(notes[0].midi), voiced=True, pressure=9.0,
                                                  tension=0.55),
                    gestures=list(plan.gestures), n_out=js_round(48000 * (plan.end + 0.4)))
    reference = ref(lab_refs, f"song-{body}/{slug}-{body}.wav")
    assert job.n_out == len(reference), "golden notes differ from the CLI demo song"
    assert_t3(reference, worker_render(job), f"{slug}-{body}")


NOTES = [("melody-lyrics", "male"), ("melody-lyrics", "female"), ("melody-lyrics", "child"),
         ("notes-ah", "male"), ("notes-ah", "female"), ("duet-upper", "female"), ("duet-lower", "male")]


@pytest.mark.parametrize("name,body", NOTES, ids=[f"{n}-{b}" for n, b in NOTES])
def test_render_notes(lab_refs, worker_render, name, body):
    from chimera_vocal.music.compile import compile_score
    from chimera_vocal.music.planning import planner_api
    from chimera_vocal.music.schema import validate_score
    j = json.loads((DATA / f"render-notes-{name}.json").read_text(encoding="utf-8"))
    notes = [{"pitch": m, "beats": b} if m > 0 else {"rest": b} for m, b in j["notes"]]
    score = {"bpm": j["bpm"], "lead_in": j.get("t0", 0.3), "normalize": False,
             "voices": [{"body": body, "lyrics": j["lyrics"], "vibrato": j.get("vibrato", 35), "notes": notes}]}
    g2p = planner_api().default_g2p()
    jobs, _, _ = compile_score(validate_score(score, g2p=g2p), g2p=g2p)
    out = f"{name}-{body}.wav" if not name.startswith("duet") else f"{name}-{body}.wav"
    assert_t3(ref(lab_refs, out), worker_render(jobs[0]), f"{name}-{body}")
