"""K-5.7 test_presets (TESTPLAN L5, INTERFACES §7): all preset ids parse, the voice-spec grammar, and
male/female/child equal the parameters of ``chimera render text … <body>`` (f0 112/205/250, default source,
render.rs:135-141)."""

from __future__ import annotations

import pytest

from chimera_vocal.core.body import BodyMix, default_f0
from chimera_vocal.core.types import SourceParams
from chimera_vocal.presets import DEFAULT_VOICE, PRESETS, SpeechStyle, VoicePreset, list_voices, parse_voice_spec

IDS = ["male", "female", "child", "androgynous", "teen", "male-deep", "female-bright", "narrator", "whisper",
       "male-whisper", "robot", "elder"]
D = SourceParams()
# id: (body, speaking f0, rate, accent, source)
TABLE = {
    "male": (BodyMix(0, 0), 112.0, 1.0, None, D),
    "female": (BodyMix(1, 0), 205.0, 1.0, None, D),
    "child": (BodyMix(0, 1), 250.0, 1.0, None, D),
    "androgynous": (BodyMix(0.5, 0), default_f0(BodyMix(0.5, 0)), 1.0, None, D),
    "teen": (BodyMix(0.5, 0.5), default_f0(BodyMix(0.5, 0.5)), 1.05, None, D),
    "male-deep": (BodyMix(0, 0), 92.0, 0.95, None, D.patched(tension=0.4, brightness=0.15)),
    "female-bright": (BodyMix(1, 0), 225.0, 1.0, None, D.patched(tension=0.55, brightness=0.45)),
    "narrator": (BodyMix(0, 0), 104.0, 0.92, 0.12, D),
    "whisper": (BodyMix(1, 0), 205.0, 0.95, None, D.patched(voiced=False)),
    "male-whisper": (BodyMix(0, 0), 112.0, 0.95, None, D.patched(voiced=False)),
    "robot": (BodyMix(0, 0), 110.0, 1.0, 0.0, D.patched(drift=0.0, roughness=0.0)),
    "elder": (BodyMix(0, 0), 118.0, 0.9, None, D.patched(roughness=0.35, breathiness=0.3, tension=0.35)),
}
GENDER = {"male": "male", "male-deep": "male", "narrator": "male", "male-whisper": "male", "robot": "male",
          "elder": "male", "female": "female", "female-bright": "female", "whisper": "female",
          "child": "neutral", "androgynous": "neutral", "teen": "neutral"}


def speaking_f0(p: VoicePreset) -> float:
    return p.speech.f0 if p.speech.f0 is not None else default_f0(p.body)


def test_ids_and_order():
    assert list(PRESETS) == IDS
    assert DEFAULT_VOICE == "male"


@pytest.mark.parametrize("pid", IDS)
def test_preset_table(pid):
    p = PRESETS[pid]
    body, f0, rate, accent, src = TABLE[pid]
    assert isinstance(p, VoicePreset) and isinstance(p.speech, SpeechStyle)
    assert p.id == pid and p.display and p.description
    assert p.body == body
    assert speaking_f0(p) == pytest.approx(f0, rel=1e-12)
    assert p.speech.rate == rate and p.speech.accent == accent
    assert p.source == src
    assert p.gender == GENDER[pid]
    assert p.seed is None


@pytest.mark.parametrize("pid", IDS)
def test_every_id_parses(pid):
    assert parse_voice_spec(pid) == PRESETS[pid]


@pytest.mark.parametrize("pid,f0", [("male", 112.0), ("female", 205.0), ("child", 250.0)])
def test_cli_parameters(pid, f0):
    p = PRESETS[pid]
    assert p.body.preset() == pid and p.speech.f0 == f0
    assert p.source == SourceParams() and p.speech.rate == 1.0 and p.speech.accent is None


def test_list_voices():
    lv = list_voices()
    assert [v["id"] for v in lv] == IDS
    for v in lv:
        assert set(v) == {"id", "display", "language", "gender"}
        assert v["language"] == "en" and v["display"] == PRESETS[v["id"]].display
        assert v["gender"] == GENDER[v["id"]]


def test_spec_pitch_rate():
    p = parse_voice_spec("female;pitch=230;rate=1.1")
    assert p.body == BodyMix(1, 0) and p.speech.f0 == 230.0 and p.speech.rate == 1.1
    assert p.source == PRESETS["female"].source


def test_spec_body_sets_default_f0():
    p = parse_voice_spec("male;body=0.3,0")
    assert p.body == BodyMix(0.3, 0.0)
    assert speaking_f0(p) == pytest.approx(default_f0(BodyMix(0.3, 0.0)), rel=1e-12)
    q = parse_voice_spec("male;body=0.3,0.2;pitch=140")
    assert q.body == BodyMix(0.3, 0.2) and q.speech.f0 == 140.0


def test_spec_semitones():
    assert speaking_f0(parse_voice_spec("female;semitones=12")) == pytest.approx(410.0, rel=1e-12)
    assert speaking_f0(parse_voice_spec("male;semitones=-12")) == pytest.approx(56.0, rel=1e-12)


def test_spec_source_options():
    p = parse_voice_spec("male;breathiness=0.4;tension=0.6;brightness=0.3;roughness=0.2;accent=0.2")
    assert p.source == D.patched(breathiness=0.4, tension=0.6, brightness=0.3, roughness=0.2)
    assert p.speech.accent == 0.2
    assert parse_voice_spec("male;whisper=true").source.voiced is False
    assert parse_voice_spec("whisper;whisper=false").source.voiced is True


def test_spec_seed():
    assert parse_voice_spec("male;seed=42").seed == 42
    assert parse_voice_spec("male;seed=4294967295").seed == 4294967295
    assert parse_voice_spec("male;seed=0").seed == 0


def test_spec_tolerates_case_and_spaces():
    assert parse_voice_spec(" Female ; pitch = 230 ") == parse_voice_spec("female;pitch=230")


@pytest.mark.parametrize("spec", [
    "unknown", "male;pitch=abc", "male;pitch=700", "male;pitch=40", "male;rate=2", "male;rate=0.4",
    "male;semitones=25", "male;accent=0.5", "male;breathiness=1.5", "male;whisper=yes", "male;seed=-1",
    "male;seed=4294967296", "male;seed=1.5", "male;foo=1", "male;pitch", "male;body=2,0", "male;body=0.5",
    ";pitch=200", "male;pitch=200;semitones=", 
])
def test_spec_invalid(spec):
    assert parse_voice_spec(spec) is None


@pytest.mark.parametrize("spec", [None, ""])
def test_spec_empty(spec):
    assert parse_voice_spec(spec) is None
