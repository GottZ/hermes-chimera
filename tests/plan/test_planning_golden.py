"""K-1.5 — planners against the template's golden plans (golden/planning.json; T1 structure/strings, T1n numbers):
98 plan_utterance, 24 Klatt duration sets, 4 plan_phrases, 7 plan_song, 23 live notes (on/off/melisma).
Gesture ORDER is part of the contract: the voice mixes overlapping gestures in list order (docs/rust-engine.md).

Red proof (TESTPLAN): emit the gestures sorted by channel.
Plus K-1.5b (mixed bodies, not covered by the goldens): vowel_artic between presets is ArticPatch::blend
of the presets' articulations (planner.rs:74-91, anatomy/human.rs:218-237), bit-exact.
"""

import pytest

from chimera_vocal.core.body import BODY_MIX_CHILD, BODY_MIX_FEMALE, BODY_MIX_MALE, BodyMix
from chimera_vocal.core.keys import ARTIC_KEYS
from chimera_vocal.core.types import PhraseInput, PlanOptions, Segment, SongOptions, SungNote, Syllable
from chimera_vocal.plan.durations import SegInfo, klatt_durations
from chimera_vocal.plan.planner import plan_phrases, plan_utterance, vowel_artic
from chimera_vocal.plan.sing import MelismaOptions, NoteOnOptions, plan_melisma, plan_note_off, plan_note_on, plan_song
from support import golden


def plan_json(p) -> dict:
    return p.to_json()


def info_by_identity(segs, infos):
    """The template's callback receives the caller's segment; find its index by identity (golden_planning.rs:120)."""
    def info(s):
        for j, x in enumerate(segs):
            if x is s:
                return infos[j]
        raise AssertionError("klatt_durations must hand the caller's Segment objects to the info callback")
    return info


def test_golden_utterances(plan_compare):
    us = golden("planning")["utterances"]
    assert len(us) == 98
    c = plan_compare("utterances")
    for u in us:
        segs = [Segment.from_json(s) for s in u["segments"]]
        c.check(f"utterance {u['name']!r}", u["plan"], plan_json(plan_utterance(segs, PlanOptions.from_json(u["opts"]))))
    print(c.report())


def test_golden_klatt(plan_compare):
    ks = golden("planning")["klatt"]
    assert len(ks) == 24
    c = plan_compare("klatt")
    for i, k in enumerate(ks):
        segs = [Segment.from_json(s) for s in k["segments"]]
        infos = [SegInfo.from_json(x) for x in k["infos"]]
        c.check(f"klatt {i}", k["durs"], list(klatt_durations(segs, info_by_identity(segs, infos))))
    print(c.report())


def test_golden_phrases(plan_compare):
    ps = golden("planning")["phrases"]
    assert len(ps) == 4
    c = plan_compare("phrases")
    for i, p in enumerate(ps):
        phrases = [PhraseInput.from_json(x) for x in p["phrases"]]
        c.check(f"phrases {i}", p["plan"], plan_json(plan_phrases(phrases, PlanOptions.from_json(p["opts"]))))
    print(c.report())


def test_golden_songs(plan_compare):
    ss = golden("planning")["songs"]
    assert len(ss) == 7
    c = plan_compare("songs")
    for s in ss:
        notes = [SungNote.from_json(n) for n in s["notes"]]
        c.check(f"song {s['name']!r}", s["plan"], plan_json(plan_song(notes, SongOptions.from_json(s["opts"]))))
    print(c.report())


def test_golden_live_notes(plan_compare):
    ns = golden("planning")["notes"]
    assert len(ns) == 23
    c = plan_compare("notes")
    legato = 0
    for i, n in enumerate(ns):
        kind = n["kind"]
        if kind == "on":
            o = NoteOnOptions.from_json(n["opts"])
            legato += o.legato_from is not None
            got = plan_note_on(Syllable.from_json(n["syl"]), n["midi"], o)
        elif kind == "off":
            got = plan_note_off(Syllable.from_json(n["syl"]), n["midi"], BodyMix.parse(n["body"]))
        elif kind == "melisma":
            got = plan_melisma(n["midi"], MelismaOptions.from_json(n["opts"]))
        else:
            raise AssertionError(kind)
        c.check(f"note {i} ({kind})", n["gestures"], [g.to_json() for g in got])
    assert legato > 0
    print(c.report())


# --- K-1.5b: mixed bodies --------------------------------------------------------------------------------

VOWELS = ("i", "ɪ", "e", "ɛ", "æ", "a", "ɑ", "ɒ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɐ", "ɝ", "y", "ø", "œ", "ɨ",
          "ɯ", "ɤ")
PRESETS = {"male": BODY_MIX_MALE, "female": BODY_MIX_FEMALE, "child": BODY_MIX_CHILD}
MIXES = [BodyMix(0.5, 0.0), BodyMix(0.0, 0.5), BodyMix(0.3, 0.4), BodyMix(1.0, 0.25), BodyMix(0.123, 0.987)]


def blend_reference(ipa: str, mix: BodyMix):
    """ArticPatch::blend over the presets with weight > 0: per key sum(w*x)/sum(w), male-female-child order."""
    parts = []
    for name, w in zip(("male", "female", "child"), mix.weights()):
        if w > 0.0:
            p = vowel_artic(ipa, PRESETS[name])
            if p is None:
                return None
            parts.append((w, p))
    out = {}
    for k in ARTIC_KEYS:
        s = ws = 0.0
        for w, p in parts:
            if k in p:
                s = s + w * p[k]
                ws = ws + w
        if ws > 0.0:
            out[k] = s / ws
    return out


@pytest.mark.parametrize("mix", MIXES, ids=lambda m: f"f{m.female}c{m.child}")
def test_vowel_artic_blends_between_presets(mix):
    bad = []
    for v in VOWELS:
        want, got = blend_reference(v, mix), vowel_artic(v, mix)
        if want is None or got is None:
            if want is not got:
                bad.append(f"{v}: {want} != {got}")
            continue
        if got != want:
            bad.append(f"{v}: {got} != {want}")
    assert not bad, "\n".join(bad[:8])


def test_vowel_artic_presets_and_unknown():
    assert vowel_artic("i", BodyMix(0.7, 1.0)) == vowel_artic("i", BODY_MIX_CHILD)    # child weight 1 = preset
    assert vowel_artic("i", BodyMix(1.0, 0.0)) == vowel_artic("i", BODY_MIX_FEMALE)
    for mix in (BODY_MIX_MALE, BodyMix(0.5, 0.5)):
        assert vowel_artic("ʉ", mix) is None          # tokenised vowel without calibration (R1-P §2)
        assert vowel_artic("aɪ", mix) is None         # diphthongs are expanded before vowel_artic
    a = vowel_artic("ɝ", BODY_MIX_MALE)
    assert a["ttCurl"] > 0.05 and "ttX" in a and "ttY" in a     # rhotic: tip position included
    assert "ttX" not in vowel_artic("i", BODY_MIX_MALE)
