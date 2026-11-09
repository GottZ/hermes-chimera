"""Serde round trips of the shared data types against goldens and vectors (frozen C0 API for C1/C2/C4)."""

import pytest

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.keys import ARTIC_KEYS, CONTROL_KEYS, N_ARTIC, N_CONTROL, SOURCE_KEYS, control_index
from chimera_vocal.core.types import (Gesture, Intonation, Phrase, PhraseInput, Plan, PlanOptions, Segment,
                                      SongOptions, SourceParams, SungNote, Syllable)
from support import golden, vector


def test_keys_match_template():
    # anatomy/human.rs ArticKey::NAMES, control/score.rs SOURCE_NAMES
    assert ARTIC_KEYS == ("jaw", "tbX", "tbY", "ttX", "ttY", "ttCurl", "root", "lipAp", "lipProt", "lipWidth",
                          "labiodental", "velum", "larynx", "epilarynx", "ventricular", "lateral", "dorsum")
    assert SOURCE_KEYS == ("abduction", "closure", "breath", "f0", "vibrato")
    assert len(CONTROL_KEYS) == N_CONTROL == 22 and N_ARTIC == 17
    assert control_index("f0") == 20 and control_index("jaw") == 0 and control_index("nope") is None


def _plans_in_planning():
    g = golden("planning")
    for u in g["utterances"]:
        yield u["name"], u["plan"]
    for s in g.get("songs", []):
        if "plan" in s:
            yield s.get("name", "song"), s["plan"]


def test_plan_roundtrip_golden_planning():
    n = 0
    for name, p in _plans_in_planning():
        assert Plan.from_json(p).to_json() == p, name
        n += 1
    assert n >= 98


@pytest.mark.parametrize("name", ["plan_phrases", "plan_song"])
def test_roundtrip_vectors(name):
    for c in vector(name).get("cases", [vector(name)]):
        assert Plan.from_json(c["plan"]).to_json() == c["plan"]
        if "phrases" in c:
            assert [PhraseInput.from_json(x).to_json() for x in c["phrases"]] == c["phrases"]
            assert PlanOptions.from_json(c["options"]).to_json() == c["options"]
        if "notes" in c:
            assert [SungNote.from_json(x).to_json() for x in c["notes"]] == c["notes"]
            assert [Syllable.from_json(x).to_json() for x in c["syllables"]] == c["syllables"]
            assert SongOptions.from_json(c["options"]).to_json() == c["options"]


def test_phrase_roundtrip_g2p():
    for c in vector("g2p")["cases"]:
        for key in ("dict", "noDict"):
            assert [Phrase.from_json(x).to_json() for x in c[key]] == c[key]


def test_gesture_targets_and_label():
    d = {"t0": 0, "t1": 1, "rampIn": 0.1, "rampOut": 0.2, "strength": 1,
         "targets": {"f0": 120.0, "bogus": 1.0, "jaw": 0.5}, "label": "x" * 30}
    g = Gesture.from_json(d)
    assert list(g.targets) == ["jaw", "f0"]                 # ControlKey order, unknown dropped
    assert g.label == "x" * 24                              # Label::CAP bytes
    assert Gesture.from_json({**d, "label": "ä" * 13}).label == "ä" * 12   # cut at a char boundary
    assert g.target_index() == [(0, 0.5), (20, 120.0)]


def test_source_params():
    s = SourceParams()
    assert s.to_json()["vibratoDepth"] == 0.0 and s.brightness == 0.25
    assert SourceParams.from_json({"f0": 200.0, "vibratoRate": 5.0}) == s.patched(f0=200.0, vibrato_rate=5.0)
    with pytest.raises(TypeError):
        s.patched(nope=1.0)


def test_enums_and_options():
    assert Intonation("question") is Intonation.QUESTION
    o = PlanOptions(BodyMix.parse("female"), 0.05, f0=205.0)
    assert o.to_json() == {"body": {"female": 1.0, "child": 0.0}, "t0": 0.05, "f0": 205.0}
    assert Segment.from_json({"ipa": "a", "wordEnd": True}).word_end is True
