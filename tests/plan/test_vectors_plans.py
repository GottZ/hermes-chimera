"""K-1.6 — the chain text -> phrases -> plan and lyrics -> syllables -> song plan against vectors exported from
the template (tests/vectors/{g2p,plan_phrases,plan_song}.json; recon/R1-work/vectors/src/main.rs):
``render text`` male (t0 0.05, f0 112), "Twinkle little" on 4 notes at 100 bpm (male, vibrato 35).

Red proof (TESTPLAN): ignore ``PlanOptions.f0``.
"""

import functools

from chimera_vocal.core.types import PhraseInput, PlanOptions, SongOptions, SungNote
from chimera_vocal.text.english import EnglishG2P
from chimera_vocal.text.syllables import lyrics_to_syllables
from chimera_vocal.plan.planner import plan_phrases
from chimera_vocal.plan.sing import plan_song
from support import vector


@functools.lru_cache(maxsize=None)
def g2ps() -> tuple[EnglishG2P, EnglishG2P]:
    return EnglishG2P.default(), EnglishG2P()


def stress_ipa(phrase) -> str:
    return "".join(("ˈ" if s.stress == 1 else "") + s.ipa for s in phrase.segments)


def test_vectors_g2p(plan_compare):
    d, nd = g2ps()
    c = plan_compare("g2p vectors")
    cases = vector("g2p")["cases"]
    assert len(cases) == 3
    for v in cases:
        ph = d.to_phrases(v["text"])
        c.check(f"{v['text']!r} dict", v["dict"], [p.to_json() for p in ph])
        c.check(f"{v['text']!r} no dict", v["noDict"], [p.to_json() for p in nd.to_phrases(v["text"])])
        c.check(f"{v['text']!r} ipa", v["ipa"], [stress_ipa(p) for p in ph])
    print(c.report())


def test_vectors_plan_phrases(plan_compare):
    d, _ = g2ps()
    c = plan_compare("plan_phrases vectors")
    cases = vector("plan_phrases")["cases"]
    assert len(cases) == 3
    for v in cases:
        inputs = [PhraseInput(p.segments, p.intonation, p.pause) for p in d.to_phrases(v["text"])]
        c.check(f"{v['text']!r} phrases", v["phrases"], [p.to_json() for p in inputs])
        opts = PlanOptions.from_json(v["options"])
        assert opts.f0 == 112.0 and opts.t0 == 0.05
        c.check(f"{v['text']!r} plan", v["plan"], plan_phrases(inputs, opts).to_json())
    print(c.report())


def test_vectors_plan_song(plan_compare):
    d, _ = g2ps()
    v = vector("plan_song")
    c = plan_compare("plan_song vector")
    syl = lyrics_to_syllables(v["lyrics"], d)
    c.check("syllables", v["syllables"], [s.to_json() for s in syl])
    notes = [SungNote(n["midi"], n["beats"], s) for n, s in zip(v["notes"], syl)]
    c.check("notes", v["notes"], [n.to_json() for n in notes])
    c.check("plan", v["plan"], plan_song(notes, SongOptions.from_json(v["options"])).to_json())
    print(c.report())
