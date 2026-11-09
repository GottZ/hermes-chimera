"""Phoneme inventory: IPA symbols as articulatory specifications for the human model (port of
phonetics/inventory.rs).

Coordinates are template space (cm) and apply to every body; the anatomy scales them. Fricative constriction
heights were calibrated by the template against target constriction areas; those calibrated values
(``data/consonant_targets.json``) override the defaults here per body, and blend between bodies.
"""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from ..core.body import BODY_NAMES, BodyMix
from ..core.keys import ARTIC_KEYS
from ..core.types import ArticPatch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Manner(enum.Enum):
    """Manner of articulation (serde lowercase names of the template)."""

    VOWEL = "vowel"
    STOP = "stop"
    NASAL = "nasal"
    FRICATIVE = "fricative"
    AFFRICATE = "affricate"
    APPROXIMANT = "approximant"
    LATERAL = "lateral"
    TAP = "tap"
    TRILL = "trill"
    GLOTTAL = "glottal"


class Voicing(enum.Enum):
    VOICED = "voiced"
    VOICELESS = "voiceless"


class Glottal(enum.Enum):
    """Glottal-only segments: [h] open, [ʔ] stop, [ɦ] breathy."""

    OPEN = "open"
    STOP = "stop"
    BREATHY = "breathy"


@dataclass(frozen=True)
class PhonemeDef:
    """One phoneme (inventory.rs:62-89). ``targets`` is read-only by convention (keys in ARTIC_KEYS order)."""

    ipa: str
    name: str
    manner: Manner
    voicing: Voicing
    place: str
    targets: ArticPatch
    dominant: tuple[str, ...]          # articulators this segment controls strongly (others coarticulate)
    nasal: bool | None
    dur: float                         # default duration (s)
    area: float | None = None          # target constriction area (cm², adult male) of the calibration
    cal_key: str | None = None         # the coordinate the calibration adjusts
    release: str | None = None         # affricate: the fricative release
    glottal: Glottal | None = None


def _c(ipa: str, name: str, manner: str, voicing: str, place: str, targets: dict, dominant: tuple, dur: float, *,
       nasal: bool | None = None, area: float | None = None, cal_key: str | None = None, release: str | None = None,
       glottal: str | None = None) -> PhonemeDef:
    return PhonemeDef(ipa, name, Manner[manner], Voicing[voicing], place, targets, dominant, nasal, dur, area, cal_key,
                      release, Glottal[glottal] if glottal else None)


# Table order and values as in the template; place templates expanded (bilabial lipAp -0.25, alveolar stop
# ttX 6.25/ttY 9.68, postalveolar 5.75/10.0, retroflex 5.45/10.05/curl 0.95, palatal tbX 3.75/tbY 8.45, velar
# tbX 2.3/tbY 8.35/jaw 0.7 -- aimed past contact so it seals wherever the vowel puts the tongue body -- uvular
# tbX 1.55/tbY 7.6).
_BASE: tuple[PhonemeDef, ...] = (
    _c('p', 'voiceless bilabial stop', 'STOP', 'VOICELESS', 'bilabial', {'lipAp': -0.25}, ('lipAp',), 0.085),
    _c('b', 'voiced bilabial stop', 'STOP', 'VOICED', 'bilabial', {'lipAp': -0.25}, ('lipAp',), 0.07),
    _c('t', 'voiceless alveolar stop', 'STOP', 'VOICELESS', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.075),
    _c('d', 'voiced alveolar stop', 'STOP', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.06),
    _c('ʈ', 'voiceless retroflex stop', 'STOP', 'VOICELESS', 'retroflex', {'ttX': 5.45, 'ttY': 10.05, 'ttCurl': 0.95}, ('ttX', 'ttY', 'ttCurl'), 0.075),
    _c('ɖ', 'voiced retroflex stop', 'STOP', 'VOICED', 'retroflex', {'ttX': 5.45, 'ttY': 10.05, 'ttCurl': 0.95}, ('ttX', 'ttY', 'ttCurl'), 0.065),
    _c('c', 'voiceless palatal stop', 'STOP', 'VOICELESS', 'palatal', {'tbX': 3.75, 'tbY': 8.45}, ('tbX', 'tbY'), 0.08),
    _c('ɟ', 'voiced palatal stop', 'STOP', 'VOICED', 'palatal', {'tbX': 3.75, 'tbY': 8.45}, ('tbX', 'tbY'), 0.07),
    _c('k', 'voiceless velar stop', 'STOP', 'VOICELESS', 'velar', {'jaw': 0.7, 'tbX': 2.3, 'tbY': 8.35}, ('tbY',), 0.08),
    _c('g', 'voiced velar stop', 'STOP', 'VOICED', 'velar', {'jaw': 0.7, 'tbX': 2.3, 'tbY': 8.35}, ('tbY',), 0.07),
    _c('q', 'voiceless uvular stop', 'STOP', 'VOICELESS', 'uvular', {'tbX': 1.55, 'tbY': 7.6}, ('tbX', 'tbY'), 0.085),
    _c('ɢ', 'voiced uvular stop', 'STOP', 'VOICED', 'uvular', {'tbX': 1.55, 'tbY': 7.6}, ('tbX', 'tbY'), 0.075),
    _c('ʔ', 'glottal stop', 'GLOTTAL', 'VOICELESS', 'glottal', {}, (), 0.06, glottal='STOP'),
    _c('m', 'bilabial nasal', 'NASAL', 'VOICED', 'bilabial', {'lipAp': -0.25}, ('lipAp',), 0.075, nasal=True),
    _c('ɱ', 'labiodental nasal', 'NASAL', 'VOICED', 'labiodental', {'lipAp': -0.1, 'labiodental': 1.0}, ('labiodental', 'lipAp'), 0.075, nasal=True),
    _c('n', 'alveolar nasal', 'NASAL', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.07, nasal=True),
    _c('ɳ', 'retroflex nasal', 'NASAL', 'VOICED', 'retroflex', {'ttX': 5.45, 'ttY': 10.05, 'ttCurl': 0.95}, ('ttX', 'ttY', 'ttCurl'), 0.07, nasal=True),
    _c('ɲ', 'palatal nasal', 'NASAL', 'VOICED', 'palatal', {'tbX': 3.75, 'tbY': 8.45}, ('tbX', 'tbY'), 0.075, nasal=True),
    _c('ŋ', 'velar nasal', 'NASAL', 'VOICED', 'velar', {'jaw': 0.7, 'tbX': 2.3, 'tbY': 8.35}, ('tbY',), 0.08, nasal=True),
    _c('ɴ', 'uvular nasal', 'NASAL', 'VOICED', 'uvular', {'tbX': 1.55, 'tbY': 7.6}, ('tbX', 'tbY'), 0.08, nasal=True),
    _c('ɸ', 'voiceless bilabial fricative', 'FRICATIVE', 'VOICELESS', 'bilabial', {'lipAp': 0.08}, ('lipAp',), 0.11, area=0.12, cal_key='lipAp'),
    _c('β', 'voiced bilabial fricative', 'FRICATIVE', 'VOICED', 'bilabial', {'lipAp': 0.1}, ('lipAp',), 0.09, area=0.15, cal_key='lipAp'),
    _c('f', 'voiceless labiodental fricative', 'FRICATIVE', 'VOICELESS', 'labiodental', {'lipAp': 0.05, 'labiodental': 1.0}, ('labiodental', 'lipAp'), 0.11, area=0.11, cal_key='lipAp'),
    _c('v', 'voiced labiodental fricative', 'FRICATIVE', 'VOICED', 'labiodental', {'lipAp': 0.06, 'labiodental': 1.0}, ('labiodental', 'lipAp'), 0.08, area=0.13, cal_key='lipAp'),
    _c('θ', 'voiceless dental fricative', 'FRICATIVE', 'VOICELESS', 'dental', {'ttX': 6.85, 'ttY': 8.5}, ('ttX', 'ttY'), 0.11, area=0.15, cal_key='ttY'),
    _c('ð', 'voiced dental fricative', 'FRICATIVE', 'VOICED', 'dental', {'ttX': 6.85, 'ttY': 8.5}, ('ttX', 'ttY'), 0.07, area=0.17, cal_key='ttY'),
    _c('s', 'voiceless alveolar fricative', 'FRICATIVE', 'VOICELESS', 'alveolar', {'jaw': 0.3, 'ttX': 6.3, 'ttY': 9.15}, ('ttX', 'ttY', 'jaw'), 0.12, area=0.085, cal_key='ttY'),
    _c('z', 'voiced alveolar fricative', 'FRICATIVE', 'VOICED', 'alveolar', {'jaw': 0.3, 'ttX': 6.3, 'ttY': 9.15}, ('ttX', 'ttY', 'jaw'), 0.09, area=0.1, cal_key='ttY'),
    _c('ʃ', 'voiceless postalveolar fricative', 'FRICATIVE', 'VOICELESS', 'postalveolar', {'jaw': 0.35, 'ttX': 5.7, 'ttY': 9.5, 'lipProt': 0.75, 'lipWidth': 0.2}, ('ttX', 'ttY', 'jaw', 'lipProt', 'lipWidth'), 0.12, area=0.2, cal_key='ttY'),
    _c('ʒ', 'voiced postalveolar fricative', 'FRICATIVE', 'VOICED', 'postalveolar', {'jaw': 0.35, 'ttX': 5.7, 'ttY': 9.5, 'lipProt': 0.75, 'lipWidth': 0.2}, ('ttX', 'ttY', 'jaw', 'lipProt', 'lipWidth'), 0.09, area=0.22, cal_key='ttY'),
    _c('ʂ', 'voiceless retroflex fricative', 'FRICATIVE', 'VOICELESS', 'retroflex', {'ttX': 5.45, 'ttY': 9.7, 'ttCurl': 0.9}, ('ttX', 'ttY', 'ttCurl'), 0.12, area=0.15, cal_key='ttY'),
    _c('ʐ', 'voiced retroflex fricative', 'FRICATIVE', 'VOICED', 'retroflex', {'ttX': 5.45, 'ttY': 9.7, 'ttCurl': 0.9}, ('ttX', 'ttY', 'ttCurl'), 0.09, area=0.17, cal_key='ttY'),
    _c('ç', 'voiceless palatal fricative', 'FRICATIVE', 'VOICELESS', 'palatal', {'tbX': 3.75, 'tbY': 8.25}, ('tbX', 'tbY'), 0.12, area=0.14, cal_key='tbY'),
    _c('ʝ', 'voiced palatal fricative', 'FRICATIVE', 'VOICED', 'palatal', {'tbX': 3.75, 'tbY': 8.25}, ('tbX', 'tbY'), 0.09, area=0.16, cal_key='tbY'),
    _c('x', 'voiceless velar fricative', 'FRICATIVE', 'VOICELESS', 'velar', {'tbX': 2.3, 'tbY': 7.7}, ('tbY',), 0.12, area=0.18, cal_key='tbY'),
    _c('ɣ', 'voiced velar fricative', 'FRICATIVE', 'VOICED', 'velar', {'tbX': 2.3, 'tbY': 7.7}, ('tbY',), 0.09, area=0.2, cal_key='tbY'),
    _c('χ', 'voiceless uvular fricative', 'FRICATIVE', 'VOICELESS', 'uvular', {'tbX': 1.55, 'tbY': 7.6}, ('tbX', 'tbY'), 0.12, area=0.2, cal_key='tbY'),
    _c('ʁ', 'voiced uvular fricative', 'FRICATIVE', 'VOICED', 'uvular', {'tbX': 1.55, 'tbY': 7.6}, ('tbX', 'tbY'), 0.09, area=0.22, cal_key='tbY'),
    _c('ħ', 'voiceless pharyngeal fricative', 'FRICATIVE', 'VOICELESS', 'pharyngeal', {'tbX': 1.3, 'tbY': 6.2, 'root': -0.95}, ('root', 'tbX', 'tbY'), 0.12, area=0.25, cal_key='tbX'),
    _c('ʕ', 'voiced pharyngeal fricative', 'FRICATIVE', 'VOICED', 'pharyngeal', {'tbX': 1.3, 'tbY': 6.2, 'root': -0.95}, ('root', 'tbX', 'tbY'), 0.09, area=0.3, cal_key='tbX'),
    _c('h', 'voiceless glottal fricative', 'GLOTTAL', 'VOICELESS', 'glottal', {}, (), 0.09, glottal='OPEN'),
    _c('ɦ', 'voiced glottal fricative', 'GLOTTAL', 'VOICED', 'glottal', {}, (), 0.08, glottal='BREATHY'),
    _c('ɬ', 'voiceless lateral fricative', 'FRICATIVE', 'VOICELESS', 'alveolar', {'ttX': 6.25, 'ttY': 9.68, 'lateral': 0.35}, ('ttX', 'ttY', 'lateral'), 0.12),
    _c('ɮ', 'voiced lateral fricative', 'FRICATIVE', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68, 'lateral': 0.45}, ('ttX', 'ttY', 'lateral'), 0.09),
    _c('ts', 'voiceless alveolar affricate', 'AFFRICATE', 'VOICELESS', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.14, release='s'),
    _c('dz', 'voiced alveolar affricate', 'AFFRICATE', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.12, release='z'),
    _c('tʃ', 'voiceless postalveolar affricate', 'AFFRICATE', 'VOICELESS', 'postalveolar', {'ttX': 5.75, 'ttY': 10.0, 'lipProt': 0.75, 'lipWidth': 0.2}, ('ttX', 'ttY', 'lipProt', 'lipWidth'), 0.15, release='ʃ'),
    _c('dʒ', 'voiced postalveolar affricate', 'AFFRICATE', 'VOICED', 'postalveolar', {'ttX': 5.75, 'ttY': 10.0, 'lipProt': 0.75, 'lipWidth': 0.2}, ('ttX', 'ttY', 'lipProt', 'lipWidth'), 0.13, release='ʒ'),
    _c('pf', 'voiceless labiodental affricate', 'AFFRICATE', 'VOICELESS', 'labiodental', {'lipAp': -0.25}, ('lipAp',), 0.14, release='f'),
    _c('w', 'labial-velar approximant', 'APPROXIMANT', 'VOICED', 'labial-velar', {'tbX': 2.45, 'tbY': 7.45, 'lipAp': 0.28, 'lipProt': 1.0, 'lipWidth': 0.05}, ('lipAp', 'lipProt', 'lipWidth', 'tbX', 'tbY'), 0.06),
    _c('ʋ', 'labiodental approximant', 'APPROXIMANT', 'VOICED', 'labiodental', {'lipAp': 0.25, 'labiodental': 0.9}, ('labiodental', 'lipAp'), 0.06),
    _c('ɹ', 'alveolar approximant', 'APPROXIMANT', 'VOICED', 'postalveolar', {'tbX': 2.3, 'tbY': 7.1, 'ttX': 5.35, 'ttY': 9.35, 'ttCurl': 0.85, 'root': -0.6, 'lipProt': 0.5, 'lipWidth': 0.3}, ('ttX', 'ttY', 'ttCurl', 'lipProt', 'lipWidth', 'root'), 0.065),
    _c('ɻ', 'retroflex approximant', 'APPROXIMANT', 'VOICED', 'retroflex', {'ttX': 5.4, 'ttY': 9.55, 'ttCurl': 1.0}, ('ttX', 'ttY', 'ttCurl'), 0.065),
    _c('j', 'palatal approximant', 'APPROXIMANT', 'VOICED', 'palatal', {'jaw': 0.25, 'tbX': 3.8, 'tbY': 7.95}, ('tbX', 'tbY'), 0.055),
    _c('ɰ', 'velar approximant', 'APPROXIMANT', 'VOICED', 'velar', {'tbX': 2.35, 'tbY': 7.45}, ('tbX', 'tbY'), 0.06),
    _c('l', 'alveolar lateral approximant', 'LATERAL', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68, 'lateral': 1.0}, ('ttX', 'ttY', 'lateral'), 0.065),
    _c('ɫ', 'velarised lateral', 'LATERAL', 'VOICED', 'alveolar', {'tbX': 1.9, 'tbY': 6.8, 'ttX': 6.25, 'ttY': 9.68, 'lateral': 1.0}, ('ttX', 'ttY', 'lateral', 'tbX', 'tbY'), 0.075),
    _c('ɭ', 'retroflex lateral', 'LATERAL', 'VOICED', 'retroflex', {'ttX': 5.45, 'ttY': 10.05, 'ttCurl': 0.95, 'lateral': 1.0}, ('ttX', 'ttY', 'ttCurl', 'lateral'), 0.065),
    _c('ʎ', 'palatal lateral', 'LATERAL', 'VOICED', 'palatal', {'tbX': 3.75, 'tbY': 8.45, 'lateral': 1.0}, ('tbX', 'tbY', 'lateral'), 0.07),
    _c('ɾ', 'alveolar tap', 'TAP', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.68}, ('ttX', 'ttY'), 0.025),
    _c('ɽ', 'retroflex flap', 'TAP', 'VOICED', 'retroflex', {'ttX': 5.45, 'ttY': 10.05, 'ttCurl': 0.95}, ('ttX', 'ttY', 'ttCurl'), 0.03),
    _c('r', 'alveolar trill', 'TRILL', 'VOICED', 'alveolar', {'ttX': 6.25, 'ttY': 9.45}, ('ttX', 'ttY'), 0.11),
    _c('ʀ', 'uvular trill', 'TRILL', 'VOICED', 'uvular', {'tbX': 1.55, 'tbY': 7.85}, ('tbX', 'tbY'), 0.11),
    _c('ʙ', 'bilabial trill', 'TRILL', 'VOICED', 'bilabial', {'lipAp': 0.02}, ('lipAp',), 0.12),
)
# a later duplicate would win (the template builds a map); the table has none
_BY_IPA: dict[str, PhonemeDef] = {p.ipa: p for p in _BASE}
SYMBOLS: tuple[str, ...] = tuple(p.ipa for p in _BASE)

_cal: dict | None = None
_cal_lock = threading.Lock()


def calibrated_consonants() -> dict:
    """``data/consonant_targets.json``: body -> IPA -> {key: value} (loaded once, values as float)."""
    global _cal
    if _cal is None:
        with _cal_lock:
            if _cal is None:
                raw = json.loads((DATA_DIR / "consonant_targets.json").read_text(encoding="utf-8"))
                _cal = {b: {ipa: {k: float(v) for k, v in t.items()} for ipa, t in tab.items()} for b, tab in raw.items()}
    return _cal


def blend(parts) -> ArticPatch:
    """Each key's weighted mean over the parts that have it; parts weighing 0 or less are left out
    (anatomy/human.rs ArticPatch::blend; same operation order, so bit-identical)."""
    out: ArticPatch = {}
    for k in ARTIC_KEYS:
        s = ws = 0.0
        for w, p in parts:
            if w > 0.0 and k in p:
                s = s + w * p[k]
                ws = ws + w
        if ws > 0.0:
            out[k] = s / ws
    return out


def _targets_for(p: PhonemeDef, body_name: str) -> ArticPatch:
    cal = calibrated_consonants().get(body_name, {}).get(p.ipa)
    if cal is None:
        return p.targets
    merged = dict(p.targets)
    merged.update(cal)
    return {k: merged[k] for k in ARTIC_KEYS if k in merged}


def phoneme(ipa: str, body: BodyMix) -> PhonemeDef | None:
    """Definition with calibrated targets for ``body``; between presets the calibrated targets blend by the
    body's weights (inventory.rs:247-263)."""
    p = _BY_IPA.get(ipa)
    if p is None:
        return None
    body = BodyMix.parse(body)
    preset = body.preset()
    if preset is not None:
        targets = _targets_for(p, preset)
    else:
        targets = blend([(w, _targets_for(p, name)) for name, w in zip(BODY_NAMES, body.weights())])
    return p if targets is p.targets else replace(p, targets=targets)


def consonants() -> tuple[PhonemeDef, ...]:
    """All definitions in table order."""
    return _BASE
