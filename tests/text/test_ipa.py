"""K-1.1 — tokenize_ipa against the template's 501 golden cases (T1 exact; golden/phonetics-ipa.json).

Red proof: leave out the NFC step.
"""

from chimera_vocal.text.ipa import tokenize_ipa
from support import golden


def test_golden_ipa(mismatches):
    cases = golden("phonetics-ipa")
    assert len(cases) == 501
    m = mismatches("tokenize_ipa")
    for c in cases:
        m.check(repr(c["input"]), c["segments"], tokenize_ipa(c["input"]))
    m.report()


def test_ipa_nfc_composes_before_lookup(mismatches):
    """NFC first (ipa.rs:32): a decomposed a+U+0303 becomes the unknown 'ã' and vanishes with its base;
    a decomposed ɛ+U+0303 has no precomposed form and keeps its base."""
    m = mismatches("tokenize_ipa NFC")
    m.check("a + combining tilde", [], tokenize_ipa("ã"))
    m.check("e + combining acute", [], tokenize_ipa("é"))
    m.check("open e + combining tilde", [{"ipa": "ɛ", "stress": 0}], tokenize_ipa("ɛ̃"))
    m.report()
