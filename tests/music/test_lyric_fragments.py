"""K-5.3 test_lyric_fragments: per-note lyrics (the ``lyric`` field of a score note).

Rule under test (``chimera_vocal.music.compile``, from the template's syllables.rs:180-220 semantics): fragments
ending in "-" continue the word on the next note; the joined word ("twin-" + "kle" -> "twinkle") is looked up as ONE
dictionary word when its syllable count equals the number of fragments; otherwise the fragments are kept as typed
syllables ("ti-ger", the template's typed-hyphen form). A lone "-" holds the previous vowel (melisma, syllable text
"~"); trailing , . ; : ! ? end a phrase. Without any lyric and without ``lyrics`` every sounding note sings "ah"
(web-app rule singing.ts)."""

from __future__ import annotations

import re

import pytest

from chimera_vocal.music.compile import compile_score
from chimera_vocal.music.schema import validate_score
from support.fakes import STUB_DICTIONARY, StubG2P, StubPlanner


def voice_score(lyrics: list[str | None], *, pitches=None) -> dict:
    pitches = pitches or [60 + i for i in range(len(lyrics))]
    notes = []
    for p, l in zip(pitches, lyrics):
        n = {"pitch": p, "beats": 1}
        if l is not None:
            n["lyric"] = l
        notes.append(n)
    return {"bpm": 100, "voices": [{"body": "female", "notes": notes}]}


def sung(planner: StubPlanner) -> list:
    """SungNotes of the last compile (all plan_song calls in order)."""
    out = []
    for notes, _opts in planner.calls_of("plan_song"):
        out.extend(notes)
    return out


def compiled(planner: StubPlanner, score: dict):
    report = validate_score(score, g2p=planner.g2p)
    assert report.ok, report.issues
    planner.calls.clear()
    compile_score(report, g2p=planner.g2p)
    return report, sung(planner), planner.calls_of("lyrics_to_syllables")


def test_dictionary_word(stub_planner):
    report, notes, texts = compiled(stub_planner, voice_score(["twin-", "kle", "lit-", "tle", "star."]))
    assert any(re.search(r"\btwinkle\b", t) for t in texts), texts
    assert not any("twin-" in t or "twin kle" in t for t in texts), texts
    assert [n.syl.text for n in notes] == ["twin", "kle", "lit", "tle", "star"]
    # the pronunciation is the dictionary word's ("twɪŋ" "kəl"), not the spelled fragments ("twin" "kle")
    assert [n.syl.nucleus.ipa for n in notes] == ["ɪ", "ə", "ɪ", "ə", "ɑ"]
    assert [n.syl.word_end for n in notes] == [False, True, False, True, True]
    assert notes[-1].syl.phrase_end is True
    assert report.voices[0].syllables == ["twin", "kle", "lit", "tle", "star"]


def test_hyphen_word_matching_count(stub_planner):
    _, notes, texts = compiled(stub_planner, voice_score(["ti-", "ger"]))
    assert any(re.search(r"\btiger\b", t) for t in texts), texts
    assert [n.syl.text for n in notes] == ["ti", "ger"]
    assert [n.syl.nucleus.ipa for n in notes] == ["a", "ɚ"], "dictionary pronunciation of 'tiger'"


@pytest.fixture
def one_syllable_tiger():
    """A dictionary in which "tiger" has ONE syllable: the fragments must stay typed ("ti-ger")."""
    from chimera_vocal.music.planning import set_planner_api
    d = dict(STUB_DICTIONARY)
    d["tiger"] = ["taɪgɚ"]
    p = StubPlanner(StubG2P(d))
    prev = set_planner_api(p.api())
    yield p
    set_planner_api(prev)


def test_hyphen_word_other_count(one_syllable_tiger):
    p = one_syllable_tiger
    _, notes, texts = compiled(p, voice_score(["ti-", "ger"]))
    assert any("ti-ger" in t for t in texts), texts
    assert [n.syl.text for n in notes] == ["ti", "ger"]
    assert [n.syl.nucleus.ipa for n in notes] == ["i", "e"], "typed fragments, spelled (not the dictionary word)"


def test_melisma(stub_planner):
    report, notes, _ = compiled(stub_planner, voice_score(["Row,", "row", "-", "-", "boat."]))
    assert [n.syl.text for n in notes] == ["Row", "row", "~", "~", "boat"]
    assert len(stub_planner.calls_of("plan_song")) == 1, "phrases split only at rests; punctuation breathes inside plan_song"
    assert [bool(n.syl.melisma) for n in notes] == [False, False, True, True, False]
    assert notes[2].syl.nucleus == notes[1].syl.nucleus == notes[3].syl.nucleus
    assert notes[0].syl.phrase_end is True
    assert report.voices[0].syllables == ["Row", "row", "~", "~", "boat"]


def test_default_ah(stub_planner):
    report, notes, texts = compiled(stub_planner, voice_score([None, None, None]))
    assert [n.syl.text.lower() for n in notes] == ["ah", "ah", "ah"]
    assert report.voices[0].syllables == ["ah", "ah", "ah"]


def test_per_note_equals_lyrics_string(stub_planner):
    pitches = [60, 62, 64, 65, 67, 67, 65, 64]
    per_note = voice_score(["hel-", "lo", "my", "friend", "sing", "with", "me", "now"], pitches=pitches)
    as_string = {"bpm": 100, "voices": [{"body": "female", "lyrics": "hello my friend sing with me now",
                                         "notes": [{"pitch": p, "beats": 1} for p in pitches]}]}
    jobs = []
    for s in (per_note, as_string):
        r = validate_score(s, g2p=stub_planner.g2p)
        assert r.ok, r.issues
        jobs.append(compile_score(r, g2p=stub_planner.g2p)[0][0])
    assert jobs[0].gestures == jobs[1].gestures and jobs[0].n_out == jobs[1].n_out
