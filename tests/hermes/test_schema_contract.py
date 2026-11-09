"""Every value the tool schemas offer passes the engine's validator.

The schemas type ``stereo`` as a string enum and ``pitch`` as a string ("or a MIDI number as text"), because some
model providers enforce declared types. Red proof: the 9a616be validator accepts only the booleans for ``stereo``
and only note names for string pitches. Uses the real validator (``chimera_vocal.music.schema``) and the real
G2P (cmudict)."""

from __future__ import annotations

import copy

import pytest

import c5fakes

from chimera_vocal.music.planning import planner_api
from chimera_vocal.music.schema import validate_score

BASE = {"bpm": 90, "voices": [{"body": "female", "notes": [{"pitch": "C4", "beats": 1, "lyric": "la"}]}]}


@pytest.fixture(scope="module")
def schemas():
    pkg = c5fakes.load_plugin()
    try:
        yield c5fakes.submodule(pkg, "chimera_hermes.tools").SCHEMAS
    finally:
        c5fakes.unload_plugin(pkg)


def _score_props(schemas):
    return schemas["chimera_render"]["parameters"]["properties"]["score"]["properties"]


def _check(score):
    return validate_score(score, g2p=planner_api().default_g2p())


def test_every_top_level_enum_value_validates(schemas):
    props = _score_props(schemas)
    tried = 0
    for key, spec in props.items():
        for value in spec.get("enum", []):
            s = copy.deepcopy(BASE)
            s[key] = value
            r = _check(s)
            assert r.ok, (key, value, [(i.path, i.message) for i in r.issues if i.severity == "error"])
            tried += 1
    assert tried >= 6


@pytest.mark.parametrize("value,stereo", [("true", True), ("false", False), (True, True), (False, False)])
def test_stereo_strings_normalize(value, stereo):
    s = copy.deepcopy(BASE)
    s["stereo"] = value
    r = _check(s)
    assert r.ok and r.normalized["stereo"] is stereo


def test_dyn_enum_validates(schemas):
    note = schemas["chimera_render"]["parameters"]["properties"]["score"]["properties"]["voices"]["items"]
    for dyn in note["properties"]["notes"]["items"]["properties"]["dyn"]["enum"]:
        s = copy.deepcopy(BASE)
        s["voices"][0]["notes"][0]["dyn"] = dyn
        assert _check(s).ok, dyn


@pytest.mark.parametrize("pitch,midi", [("C4", 60.0), ("F#3", 54.0), ("Bb4", 70.0), ("60", 60.0), ("60.5", 60.5)])
def test_pitch_examples_from_the_schema(schemas, pitch, midi):
    desc = _score_props(schemas)["voices"]["items"]["properties"]["notes"]["items"]["properties"]["pitch"]
    assert "MIDI number as text" in desc["description"]
    s = copy.deepcopy(BASE)
    s["voices"][0]["notes"][0]["pitch"] = pitch
    r = _check(s)
    assert r.ok, [(i.path, i.message) for i in r.issues]
    assert r.normalized["voices"][0]["notes"][0]["pitch"] == pytest.approx(midi)


def test_body_example_from_the_schema():
    s = copy.deepcopy(BASE)
    s["voices"][0]["body"] = "female=0.5,child=0.2"
    assert _check(s).ok
