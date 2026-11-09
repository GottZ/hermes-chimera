"""K-5.1 test_score_validation (TESTPLAN L5): every faulty score yields exactly one error issue at the expected JSON
path, with a one-sentence message that says what to write instead (INTERFACES §8.2). Red proof: error path without
the list index (e.g. "voices.notes.pitch").

Also: note-name grammar (§8.1), defaults in ``normalized``, voice summaries, range warnings (§8.2)."""

from __future__ import annotations

import copy
import json
import re

import pytest

from chimera_vocal.music.schema import (NATURAL_RANGE_MIDI, ScoreReport, midi_name, parse_body, parse_pitch,
                                        validate_score)
from chimera_vocal.core.body import BodyMix

FIX_WORDS = re.compile(
    r"\b(use|give|write|set|remove|raise|lower|shorten|add|choose|split|put|keep|make|drop|replace|reduce|"
    r"increase|pick|move|start|end|delete|try|spell|rename|leave)\b", re.IGNORECASE)


def base_score() -> dict:
    return {
        "title": "Test round",
        "bpm": 96,
        "voices": [
            {"name": "soprano", "body": "female", "notes": [
                {"pitch": "E5", "beats": 1, "lyric": "Row,"}, {"pitch": "D5", "beats": 1, "lyric": "row"},
                {"pitch": "C5", "beats": 1, "lyric": "row"}, {"pitch": "D5", "beats": 1, "lyric": "your"},
                {"pitch": "E5", "beats": 2, "lyric": "boat."}]},
            {"name": "alto", "body": "female", "notes": [
                {"pitch": "C5", "beats": 1, "lyric": "Row,"}, {"pitch": "B4", "beats": 1, "lyric": "row"},
                {"pitch": "A4", "beats": 1, "lyric": "row"}, {"pitch": "B4", "beats": 1, "lyric": "your"},
                {"pitch": "C5", "beats": 2, "lyric": "boat."}]},
            {"name": "bass", "body": "male", "notes": [
                {"pitch": "C3", "beats": 1, "lyric": "Row,"}, {"pitch": "G2", "beats": 1, "lyric": "row"},
                {"pitch": "A2", "beats": 1, "lyric": "row"}, {"pitch": "G2", "beats": 1, "lyric": "your"},
                {"pitch": "C3", "beats": 2, "lyric": "boat."}]},
        ],
    }


def _set(path: list, value):
    def f(s):
        cur = s
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
        return s
    return f


def _del(path: list):
    def f(s):
        cur = s
        for k in path[:-1]:
            cur = cur[k]
        del cur[path[-1]]
        return s
    return f


def _apply(*fs):
    def f(s):
        for g in fs:
            s = g(s)
        return s
    return f


def _notes(i: int, notes: list):
    return _set(["voices", i, "notes"], notes)


SAY_TOO_LONG = [{"pitch": "C3", "beats": 1, "lyric": "Row,"},
                {"say": "and now, all together!", "beats": 2},
                {"pitch": "C3", "beats": 1, "lyric": "row."}]

# (id, mutation, expected error path)
CASES = [
    # score root
    ("root-list", lambda s: [s], "score"),
    ("root-bad-json", lambda s: "{not json", "score"),
    ("root-number", lambda s: 42, "score"),
    ("unknown-top-key", _set(["tempo"], 120), "tempo"),
    ("title-too-long", _set(["title"], "x" * 121), "title"),
    ("title-not-string", _set(["title"], ["a"]), "title"),
    # tempo and global options
    ("bpm-missing", _del(["bpm"]), "bpm"),
    ("bpm-text", _set(["bpm"], "fast"), "bpm"),
    ("bpm-bool", _set(["bpm"], True), "bpm"),
    ("bpm-low", _set(["bpm"], 20), "bpm"),
    ("bpm-high", _set(["bpm"], 300), "bpm"),
    ("lead-in-negative", _set(["lead_in"], -0.1), "lead_in"),
    ("lead-in-high", _set(["lead_in"], 3), "lead_in"),
    ("normalize-text", _set(["normalize"], "yes"), "normalize"),
    ("stereo-word", _set(["stereo"], "both"), "stereo"),
    ("format-ogg", _set(["format"], "ogg"), "format"),
    # voices list
    ("voices-missing", _del(["voices"]), "voices"),
    ("voices-empty", _set(["voices"], []), "voices"),
    ("voices-not-list", _set(["voices"], {"a": 1}), "voices"),
    ("voices-too-many", lambda s: _set(["voices"], [copy.deepcopy(s["voices"][2]) for _ in range(9)])(s), "voices"),
    ("voice-not-object", _set(["voices", 1], "soprano"), "voices[1]"),
    # voice fields
    ("voice-unknown-key", _set(["voices", 0, "instrument"], "flute"), "voices[0].instrument"),
    ("voice-name-long", _set(["voices", 0, "name"], "n" * 41), "voices[0].name"),
    ("body-unknown", _set(["voices", 0, "body"], "robot"), "voices[0].body"),
    ("body-mix-range", _set(["voices", 0, "body"], {"female": 1.5}), "voices[0].body.female"),
    ("body-mix-key", _set(["voices", 0, "body"], {"female": 0.5, "teen": 1}), "voices[0].body.teen"),
    ("body-string-mix-bad", _set(["voices", 0, "body"], "female=2,child=0"), "voices[0].body"),
    ("start-negative", _set(["voices", 1, "start"], -1), "voices[1].start"),
    ("volume-high", _set(["voices", 0, "volume"], 20), "voices[0].volume"),
    ("pan-range", _set(["voices", 0, "pan"], 2), "voices[0].pan"),
    ("breath-range", _set(["voices", 2, "breath"], 30), "voices[2].breath"),
    ("tension-range", _set(["voices", 0, "tension"], 1.5), "voices[0].tension"),
    ("vibrato-range", _set(["voices", 0, "vibrato"], 200), "voices[0].vibrato"),
    ("vibrato-rate-range", _set(["voices", 0, "vibrato_rate"], 10), "voices[0].vibrato_rate"),
    ("breathiness-range", _set(["voices", 0, "breathiness"], -0.1), "voices[0].breathiness"),
    ("brightness-text", _set(["voices", 0, "brightness"], "bright"), "voices[0].brightness"),
    ("ring-range", _set(["voices", 0, "ring"], 2), "voices[0].ring"),
    ("seed-negative", _set(["voices", 0, "seed"], -1), "voices[0].seed"),
    ("seed-too-big", _set(["voices", 0, "seed"], 2 ** 32), "voices[0].seed"),
    ("seed-fraction", _set(["voices", 0, "seed"], 1.5), "voices[0].seed"),
    ("notes-missing", _del(["voices", 1, "notes"]), "voices[1].notes"),
    ("notes-empty", _set(["voices", 0, "notes"], []), "voices[0].notes"),
    ("notes-too-many", _notes(0, [{"pitch": "C4", "beats": 0.1, "lyric": "ah"}] * 2001), "voices[0].notes"),
    # note items
    ("note-not-object", _set(["voices", 0, "notes", 2], "C4"), "voices[0].notes[2]"),
    ("pitch-name-H", _set(["voices", 0, "notes", 3, "pitch"], "H4"), "voices[0].notes[3].pitch"),
    ("pitch-no-octave", _set(["voices", 0, "notes", 3, "pitch"], "C"), "voices[0].notes[3].pitch"),
    ("pitch-midi-high", _set(["voices", 0, "notes", 2, "pitch"], 110), "voices[0].notes[2].pitch"),
    ("pitch-name-high", _set(["voices", 0, "notes", 2, "pitch"], "C9"), "voices[0].notes[2].pitch"),
    ("pitch-midi-low", _set(["voices", 2, "notes", 1, "pitch"], 20), "voices[2].notes[1].pitch"),
    ("pitch-bool", _set(["voices", 1, "notes", 0, "pitch"], True), "voices[1].notes[0].pitch"),
    ("beats-zero", _set(["voices", 0, "notes", 1, "beats"], 0), "voices[0].notes[1].beats"),
    ("beats-text", _set(["voices", 1, "notes", 4, "beats"], "two"), "voices[1].notes[4].beats"),
    ("beats-missing", _del(["voices", 0, "notes", 1, "beats"]), "voices[0].notes[1].beats"),
    ("pitch-and-rest", _set(["voices", 0, "notes", 1, "rest"], 1), "voices[0].notes[1]"),
    ("neither-pitch-rest-say", _set(["voices", 0, "notes", 1], {"beats": 1, "lyric": "row"}), "voices[0].notes[1]"),
    ("note-unknown-key", _set(["voices", 0, "notes", 1, "velocity"], 90), "voices[0].notes[1].velocity"),
    ("rest-negative", _set(["voices", 0, "notes", 1], {"rest": -1}), "voices[0].notes[1].rest"),
    ("rest-with-lyric", _set(["voices", 0, "notes", 1], {"rest": 1, "lyric": "row"}), "voices[0].notes[1].lyric"),
    ("dyn-word", _set(["voices", 0, "notes", 2, "dyn"], "loud"), "voices[0].notes[2].dyn"),
    ("lyric-number", _set(["voices", 0, "notes", 1, "lyric"], 5), "voices[0].notes[1].lyric"),
    ("lyric-missing-one", _del(["voices", 0, "notes", 2, "lyric"]), "voices[0].notes[2].lyric"),
    ("melisma-first", _set(["voices", 1, "notes", 0, "lyric"], "-"), "voices[1].notes[0].lyric"),
    ("lyrics-and-lyric", _set(["voices", 0, "lyrics"], "row row row your boat"), "voices[0].lyrics"),
    ("lyrics-count", lambda s: _apply(
        _set(["voices", 2, "lyrics"], "row row your boat"),
        *[_del(["voices", 2, "notes", k, "lyric"]) for k in range(5)])(s), "voices[2]"),
    # spoken slots
    ("say-too-long", _notes(2, SAY_TOO_LONG), "voices[2].notes[1]"),
    ("say-unspeakable", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."}, {"say": "!!!", "beats": 2}]),
     "voices[2].notes[1].say"),
    ("say-empty", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."}, {"say": "", "beats": 2}]),
     "voices[2].notes[1].say"),
    ("say-with-pitch", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."},
                                  {"say": "hi", "pitch": "C3", "beats": 2}]), "voices[2].notes[1]"),
    ("say-rate", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."}, {"say": "hi", "beats": 4, "rate": 3}]),
     "voices[2].notes[1].rate"),
    ("say-hz", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."}, {"say": "hi", "beats": 4, "hz": 1000}]),
     "voices[2].notes[1].hz"),
    ("say-no-beats", _notes(2, [{"pitch": "C3", "beats": 1, "lyric": "row."}, {"say": "hi"}]),
     "voices[2].notes[1].beats"),
    # size limits (bpm 30: 1 beat = 2 s)
    ("mix-too-long", lambda s: _apply(_set(["bpm"], 30), _set(["voices"], [
        {"body": "male", "notes": [{"pitch": "C3", "beats": 4, "lyric": "ah"}] * 40}]))(s), "score"),
    ("voice-seconds-too-many", lambda s: _apply(_set(["bpm"], 30), _set(["voices"], [
        {"body": "male", "notes": [{"pitch": "C3", "beats": 4, "lyric": "ah"}] * 30} for _ in range(6)]))(s),
     "score"),
]


@pytest.mark.parametrize("case_id,mutate,path", CASES, ids=[c[0] for c in CASES])
def test_score_validation(stub_planner, case_id, mutate, path):
    score = mutate(base_score())
    report = validate_score(score, g2p=stub_planner.g2p)
    errors = [i for i in report.issues if i.severity == "error"]
    assert not report.ok, f"{case_id}: report.ok although the score is invalid"
    assert [e.path for e in errors] == [path], f"{case_id}: error paths {[e.path for e in errors]} != [{path!r}]"
    msg = errors[0].message
    assert msg and "\n" not in msg and len(msg) <= 400, f"{case_id}: message not one sentence: {msg!r}"
    assert FIX_WORDS.search(msg), f"{case_id}: message does not say what to write instead: {msg!r}"


def test_case_count():
    assert len(CASES) >= 40


def test_valid_score_and_defaults(stub_planner):
    report = validate_score(base_score(), g2p=stub_planner.g2p)
    assert isinstance(report, ScoreReport)
    assert report.ok, [i for i in report.issues if i.severity == "error"]
    n = report.normalized
    assert n["bpm"] == 96 and n["lead_in"] == pytest.approx(0.3) and n["normalize"] is True
    assert n["stereo"] == "auto" and n["format"] == "auto"
    v = n["voices"][2]
    assert v["name"] == "bass" and v["start"] == 0 and v["volume"] == 0 and v["pan"] == 0
    assert v["breath"] == 9 and v["tension"] == 0.55 and v["vibrato"] == 35 and v["vibrato_rate"] == 6.0
    assert v["breathiness"] == 0.15 and v["brightness"] == 0.25 and v["roughness"] == 0.1
    assert v["ring"] == 0 and v["seed"] is None
    assert [x["pitch"] for x in v["notes"]] == [48.0, 43.0, 45.0, 43.0, 48.0]
    assert all(isinstance(x["pitch"], float) for x in v["notes"])
    unnamed = copy.deepcopy(base_score())
    del unnamed["voices"][1]["name"]
    assert validate_score(unnamed, g2p=stub_planner.g2p).normalized["voices"][1]["name"] == "voice1"


def test_json_string_input(stub_planner):
    s = json.dumps(base_score())
    assert validate_score(s, g2p=stub_planner.g2p).ok


def test_voice_summary(stub_planner):
    score = base_score()
    score["voices"][0]["notes"] = [
        {"pitch": "C4", "beats": 1, "lyric": "twin-"}, {"pitch": "C4", "beats": 1, "lyric": "kle"},
        {"pitch": "G4", "beats": 1, "lyric": "-"}, {"rest": 1}, {"pitch": "A4", "beats": 2, "lyric": "star."}]
    report = validate_score(score, g2p=stub_planner.g2p)
    assert report.ok, report.issues
    s = report.voices[0]
    assert (s.name, s.body, s.items, s.sounding, s.spoken) == ("soprano", "female", 5, 4, 0)
    assert (s.first_beat, s.end_beat) == (0, 6)
    assert (s.lowest, s.highest) == ("C4", "A4")
    assert s.syllables == ["twin", "kle", "~", "star"]
    assert s.seconds > 6 * 60 / 96
    # VoiceSummary.seconds = the voice's RENDERED length (start_block … end + 0.4 s), Σ = voice_seconds
    assert report.voice_seconds == pytest.approx(sum(v.seconds for v in report.voices), rel=1e-12)


def test_natural_range_warning(stub_planner):
    score = base_score()
    score["voices"][2]["notes"][4]["pitch"] = "C6"
    report = validate_score(score, g2p=stub_planner.g2p)
    assert report.ok
    warns = [i for i in report.issues if i.severity == "warning" and i.path == "voices[2].notes[4].pitch"]
    assert len(warns) == 1 and "male" in warns[0].message


@pytest.mark.parametrize("pitch,warn", [(80, True), (75, False), (44, True), (47, False)])
def test_natural_range_blend(stub_planner, pitch, warn):
    # {female 0.5}: weights (0.5, 0.5, 0) -> bounds 45.5 .. 76.5
    score = base_score()
    score["voices"][1]["body"] = {"female": 0.5}
    score["voices"][1]["notes"][2]["pitch"] = pitch
    report = validate_score(score, g2p=stub_planner.g2p)
    paths = [i.path for i in report.issues if i.severity == "warning"]
    assert ("voices[1].notes[2].pitch" in paths) is warn
    assert NATURAL_RANGE_MIDI["female"] == (53, 84)


@pytest.mark.parametrize("text,midi", [
    ("C4", 60.0), ("c4", 60.0), ("C#5", 73.0), ("Db5", 73.0), ("Bb3", 58.0), ("B##3", 61.0), ("Cbb4", 58.0),
    ("A4", 69.0), ("A4+15", 69.15), ("A4-30", 68.7), ("E1", 28.0), ("E7", 100.0), (64, 64.0), (60.5, 60.5),
])
def test_parse_pitch(text, midi):
    assert parse_pitch(text) == pytest.approx(midi, abs=1e-12)


@pytest.mark.parametrize("bad", ["H4", "C", "4", "C#", "Cx4", "A4+", "", "do4", None, True, [60]])
def test_parse_pitch_invalid(bad):
    with pytest.raises(ValueError):
        parse_pitch(bad)


@pytest.mark.parametrize("midi,name", [(60, "C4"), (61, "C#4"), (58, "A#3"), (69.15, "A4+15"), (28, "E1")])
def test_midi_name(midi, name):
    assert midi_name(midi) == name


@pytest.mark.parametrize("spec,mix", [
    ("male", BodyMix(0.0, 0.0)), ("female", BodyMix(1.0, 0.0)), ("child", BodyMix(0.0, 1.0)),
    ({"female": 0.5}, BodyMix(0.5, 0.0)), ({"female": 0.5, "child": 0.2}, BodyMix(0.5, 0.2)),
    ("female=0.5,child=0.2", BodyMix(0.5, 0.2)), ("child=1", BodyMix(0.0, 1.0)),
])
def test_parse_body(spec, mix):
    assert parse_body(spec) == mix
