"""K-1.7 — singing behaviour (port of crates/chimera/tests/sing_fast.rs): every vowel lands on its beat (± 1e-6 s)
however short the note, keeps >= 20 ms, consonant pile-ups slip < 8 ms per note, roomy notes keep their
consonants, onset consonants carry the new pitch, a delayed cluster closure stays inside a short sung stop.
This is what makes several voices synchronous by construction.

Red proof (TESTPLAN): put the onset consonants after the beat instead of before it.
"""

import pytest

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.jsmath import midi_to_hz
from chimera_vocal.core.types import SongOptions, SungNote
from chimera_vocal.plan.sing import plan_song
from chimera_vocal.text.english import EnglishG2P
from chimera_vocal.text.syllables import lyrics_to_syllables

SCAT = ("bah\tB AA1\nbee\tB IY1\nbop\tB AA1 P\ndee\tD IY1\ndoo\tD UW1\nscoo\tS K UW1\nskee\tS K IY1\n"
        "strut\tS T R AH1 T\n")
MALE = BodyMix.parse("male")


def g2p() -> EnglishG2P:
    g = EnglishG2P()
    g.load_dict(SCAT)
    return g


def sing(lyrics: str, beats: float, bpm: float):
    """The melody alternates C4 and E4, ``beats`` long each; first beat at 0.5 s."""
    syl = lyrics_to_syllables(lyrics, g2p())
    notes = [SungNote(60.0 if i % 2 == 0 else 64.0, beats, s) for i, s in enumerate(syl)]
    times = [0.5 + i * beats * 60.0 / bpm for i in range(len(notes))]
    plan = plan_song(notes, SongOptions(MALE, bpm, 0.5, 0.0))
    return notes, times, plan


def vowel_onsets(segs) -> list[float]:
    """The first vowel segment of each syllable (a diphthong's second part follows its first)."""
    return [s.t0 for i, s in enumerate(segs) if s.kind == "vowel" and (i == 0 or segs[i - 1].kind != "vowel")]


RUNS = ["scoo bah dee bop scoo bah dee bop", "skee dee bop bah skee dee bop bah",
        "bah bee doo bop bah bee doo bop", "strut strut strut strut"]
TEMPI = [(184.0, 0.25), (184.0, 1.0 / 3.0), (240.0, 0.25), (100.0, 0.5)]


@pytest.mark.parametrize("lyrics", RUNS)
def test_fast_runs_land_on_the_beat(lyrics):
    for bpm, beats in TEMPI:
        if lyrics.startswith("strut") and bpm == 240.0:
            continue      # four consonants in a 62 ms note: see test_too_many_consonants_degrade_gently
        _, times, plan = sing(lyrics, beats, bpm)
        got = vowel_onsets(plan.segments)
        assert len(got) == len(times), f"{lyrics!r} at {bpm} bpm"
        for k, (g, want) in enumerate(zip(got, times)):
            assert abs(g - want) < 1e-6, f"{lyrics!r}, {beats} beats at {bpm} bpm: vowel {k} at {g:.4f} s, beat {want:.4f} s"
        for s in plan.segments:
            if s.kind == "vowel":
                assert s.t1 - s.t0 >= 0.02 - 1e-9, f"{lyrics!r}: a {s.t1 - s.t0:.3f} s vowel"


def test_too_many_consonants_degrade_gently():
    _, times, plan = sing("strut strut strut strut", 0.25, 240.0)
    onsets = vowel_onsets(plan.segments)
    assert len(onsets) == len(times)
    for k, (g, want) in enumerate(zip(onsets, times)):
        assert g - want < 0.008 * k + 1e-6, f"vowel {k} {(g - want) * 1000:.1f} ms late"


def test_notes_with_room_keep_their_consonants():
    _, _, plan = sing("bah dee bah dee", 1.0, 100.0)
    d = next(s for s in plan.segments if s.ipa == "d")
    assert d.t1 - d.t0 > 0.04, f"d squeezed to {d.t1 - d.t0:.3f} s"


def test_onset_consonants_carry_the_new_pitch():
    notes, _, plan = sing("bah dee bah dee", 1.0 / 3.0, 184.0)
    d = next(s for s in plan.segments if s.ipa == "d")
    hz = midi_to_hz(notes[1].midi)
    assert any(abs(g.t0 - d.t0) < 1e-9 and g.targets.get("f0") == hz for g in plan.gestures), \
        f"no pitch gesture for {hz:.1f} Hz starting on the 'd' at {d.t0:.4f} s"


def test_a_delayed_cluster_closure_stays_inside_a_short_sung_stop():
    g = EnglishG2P()
    g.load_dict("harped\tHH AA1 R P T\nah\tAA1\n")
    syl = lyrics_to_syllables("harped ah harped ah", g)
    notes = [SungNote(60.0, 0.25, s) for s in syl]
    plan = plan_song(notes, SongOptions(MALE, 180.0, 0.5, 0.0))
    ts = [s for i, s in enumerate(plan.segments) if s.ipa == "t" and i > 0 and plan.segments[i - 1].ipa == "p"]
    assert ts
    for t in ts:
        for gst in plan.gestures:
            if gst.label == "t" and t.t0 - 0.05 <= gst.t0 <= t.t1:
                assert gst.t0 <= gst.t1, f"a /t/ gesture runs backwards: {gst.t0:.4f} -> {gst.t1:.4f}"
                assert gst.t1 <= t.t1 + 1e-9, f"a /t/ gesture ends at {gst.t1:.4f} s, after its /t/ ({t.t0:.4f}-{t.t1:.4f} s)"
