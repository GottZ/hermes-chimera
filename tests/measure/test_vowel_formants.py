"""K-4.2 test_vowel_formants (``lab``, real engine): held /ɑ i u/ per body - LPC (order 12 at 12 kHz) F1/F2 of
the port within ±3 % of the same LPC on the native render (ref-audio/engine/sustain-<v>-<body>.wav: fresh voice,
f0 = default_f0(body), vowel_artic(ipa, body) with tip_free, breath 1.2 s on + 0.3 s off, via the template's Engine).
Red proof: the vowel table of another body."""

from __future__ import annotations

import pytest

from chimera_vocal.core.body import BodyMix, default_f0
from chimera_vocal.core.types import Seeds, SourceParams
from chimera_vocal.core.wavio import read_wav
from support.fakes import real_engine_available
from support.pitch import lpc_formants

pytestmark = [pytest.mark.lab,
              pytest.mark.skipif(not real_engine_available(), reason="needs the real engine")]
VOWELS = {"a": "ɑ", "i": "i", "u": "u"}
BODIES = ["male", "female", "child"]


def sustain_job(body: str, ipa: str):
    from chimera_vocal.music.compile import Command, RenderJob
    from chimera_vocal.plan.planner import vowel_artic
    mix = BodyMix.parse(body)
    patch = vowel_artic(ipa, mix)
    return RenderJob(id=f"sustain-{ipa}-{body}", body=mix, seeds=Seeds(),
                     source=SourceParams().patched(f0=default_f0(mix)),
                     commands=[Command(0, "artic", {"patch": dict(patch), "tip_free": True}),
                               Command(0, "gate", {"level": 1.0}), Command(57600, "gate", {"level": 0.0})],
                     n_out=72000)


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("tag", list(VOWELS))
def test_vowel_formants(lab_refs, body, tag):
    from chimera_vocal.music.compile import render_job
    sr, native = read_wav(lab_refs / "ref-audio" / "engine" / f"sustain-{tag}-{body}.wav")
    port = render_job(sustain_job(body, VOWELS[tag]), None, None)
    c = int(0.8 * 48000)
    fn = lpc_formants(native[c:c + 1536], 48000)
    fp = lpc_formants(port[c:c + 1536].astype("float64"), 48000)
    for k in range(2):
        assert abs(fp[k] - fn[k]) / fn[k] <= 0.03, f"{body} /{VOWELS[tag]}/ F{k + 1}: port {fp[k]:.0f} native {fn[k]:.0f}"
