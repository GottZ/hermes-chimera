"""Segment sequence -> gestural plan (port of phonetics/planner.rs).

Timing and phonation are articulatory: aspiration is a glottal opening gesture peaking at the release (voice
onset time emerges), nasality is velum lowering with anticipation, coarticulation comes from overlapping vowel
gestures beneath consonant gestures. The ORDER of the emitted gestures is part of the result: the voice mixes
overlapping gestures in list order, so a different order changes the floating-point sums.
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from typing import Sequence

from ..core.body import BODY_NAMES, BodyMix
from ..core.jsmath import js_max, js_min, js_round
from ..core.keys import ARTIC_KEYS
from ..core.types import ArticPatch, Gesture, Intonation, Plan, PlannedSegment, PhraseInput, PlanOptions, Segment
from .durations import klatt_durations
from .inventory import DATA_DIR, Glottal, Manner, PhonemeDef, Voicing, blend, phoneme

EXHALE_LABEL = "exhale"     # the end-of-breath-group abduction (a live note can drop it when the next starts)

# when a sung phrase's breath reaches full drive, after the plan start (its first segment starts at 0.06)
_BREATH_SUNG_AT = 0.06
_VOWEL_KEYS = frozenset(("tbX", "tbY", "jaw", "lipAp", "lipProt", "lipWidth", "root", "larynx", "ttCurl"))

# Diphthongs and r-coloured vowels expand to vowel sequences. American /eɪ oʊ/ glide toward [i u] (Hillenbrand
# et al. 1995: /e/ F2 rises ~150 Hz, /o/ F1 falls with F2 steady); with [ɪ ʊ] as targets they barely moved and
# sounded like the steady [eː oː] of Indian English. The other offglides undershoot toward [ɪ ʊ].
_EXPAND: dict[str, tuple[str, ...]] = {"aɪ": ("a", "ɪ"), "aʊ": ("a", "ʊ"), "ɔɪ": ("ɔ", "ɪ"), "eɪ": ("e", "i"),
                                       "oʊ": ("o", "u"), "ɚ": ("ɝ",)}

_vowels: dict | None = None
_vowels_lock = threading.Lock()


def vowel_table() -> dict:
    """``data/vowel_targets.json``: body -> IPA -> calibrated articulation (only ``artic`` is read)."""
    global _vowels
    if _vowels is None:
        with _vowels_lock:
            if _vowels is None:
                raw = json.loads((DATA_DIR / "vowel_targets.json").read_text(encoding="utf-8"))
                _vowels = {b: {ipa: {k: float(x) for k, x in v["artic"].items() if k in ARTIC_KEYS}
                               for ipa, v in tab.items()} for b, tab in raw.items()}
    return _vowels


def expand(ipa: str) -> tuple[str, ...] | None:
    return _EXPAND.get(ipa)


def diphthong_parts(ipa: str) -> tuple[str, str] | None:
    """The two vowel targets of a diphthong ("aɪ" -> ("a", "ɪ")), or None."""
    p = _EXPAND.get(ipa)
    return (p[0], p[1]) if p is not None and len(p) == 2 else None


def is_vowel(ipa: str, body: BodyMix) -> bool:
    t = vowel_table()
    body = BodyMix.parse(body)
    return any(w > 0.0 and ipa in t.get(b, {}) for b, w in zip(BODY_NAMES, body.weights())) or ipa in _EXPAND


def _preset_vowel_artic(ipa: str, body_name: str) -> ArticPatch | None:
    t = vowel_table()
    v = t.get(body_name, {}).get(ipa)
    if v is None:
        v = t.get("male", {}).get(ipa)
    if v is None:
        return None
    out = {k: v[k] for k in ARTIC_KEYS if k in _VOWEL_KEYS and k in v}
    # a rhotic's tip is raised and curled back, not resting behind the teeth: without its calibrated position
    # the tip stayed down and /ɝ/'s F3 sat 15-20 % high
    if v.get("ttCurl", 0.0) > 0.05:
        for k in ("ttX", "ttY"):
            if k in v:
                out[k] = v[k]
        out = {k: out[k] for k in ARTIC_KEYS if k in out}
    return out


def vowel_artic(ipa: str, body: BodyMix) -> ArticPatch | None:
    """The calibrated articulation of a vowel for a body; between presets the presets' articulations blend by
    the body's weights (linear in articulator space)."""
    body = BodyMix.parse(body)
    preset = body.preset()
    if preset is not None:
        return _preset_vowel_artic(ipa, preset)
    parts = []
    for name, w in zip(BODY_NAMES, body.weights()):
        if w > 0.0:
            p = _preset_vowel_artic(ipa, name)
            if p is None:
                return None
            parts.append((w, p))
    return blend(parts)


def _truthy(x: float | None) -> bool:
    """JS truthiness of an optional number (0 and NaN are false)."""
    return x is not None and x != 0.0 and not math.isnan(x)


def _g(t0: float, t1: float, ramp_in: float, ramp_out: float, strength: float, targets: dict, label: str, *,
       speed: float | None = None, release_hold: float | None = None) -> Gesture:
    return Gesture(t0, t1, ramp_in, ramp_out, strength, targets, speed=speed, release_hold=release_hold, label=label)


class _Seg:
    """A planner segment: the input segment plus what the planner attaches to it."""

    __slots__ = ("ipa", "dur", "stress", "f0", "defn", "vowel", "glide_from")

    def __init__(self, ipa, dur, stress, f0, glide_from):
        self.ipa, self.dur, self.stress, self.f0, self.glide_from = ipa, dur, stress, f0, glide_from
        self.defn: PhonemeDef | None = None
        self.vowel: ArticPatch | None = None


_PLACE_RANK = {"bilabial": 0, "labiodental": 1, "dental": 2, "alveolar": 3, "postalveolar": 4, "retroflex": 5,
               "palatal": 6, "velar": 7, "uvular": 8}
# How much later a pre-placed closure behind another stop's closure is aimed (s): the time its shorter travel
# saves, so it seals as the front closure opens.
_PLACED_CLOSURE_DELAY = 0.025


def _pre_placed_behind_closure(segs: list[_Seg], i: int) -> bool:
    """Stop ``i`` closes behind the closure of the stop before it, its articulators already set by the segment
    before that one (/t/ in /ɹpt/: the tip is up for the /ɹ/ and stays free during the lip closure; a rhotic
    vowel raises it too)."""
    d = segs[i].defn
    prev = segs[i - 1].defn if i >= 1 else None
    if d is None or prev is None:
        return False
    if d.manner is not Manner.STOP or prev.manner is not Manner.STOP or i < 2:
        return False
    f, b = _PLACE_RANK.get(prev.place), _PLACE_RANK.get(d.place)
    if f is None or b is None or not f < b:
        return False
    before = segs[i - 2]
    rhotic = before.vowel if before.vowel is not None and before.vowel.get("ttCurl", 0.0) > 0.05 else None
    tg = before.defn.targets if before.defn is not None else rhotic
    if tg is None:
        return False
    return all(k in tg for k in d.dominant)


def _pitch_ramp(dur: float) -> float:
    """Ramps of a sung note's pitch on a segment ``dur`` long: 60 ms, but at most half the segment (a ramp-in
    starts before the segment, so in a fast run a full 60 ms would pull the previous vowel off its note)."""
    return js_min(0.06, js_max(0.008, 0.5 * dur))


_CLOSING = (Manner.STOP, Manner.NASAL, Manner.AFFRICATE)
_ALVEOLAR_FRONTED = (Manner.STOP, Manner.NASAL, Manner.FRICATIVE, Manner.AFFRICATE, Manner.TAP)


def oral_gestures_into(d: PhonemeDef, tg: ArticPatch, a: float, b: float, speed: float, label: str,
                       out: list[Gesture]) -> None:
    """Oral gestures for one consonant constriction held acoustically from ``a`` to ``b`` (closure to release):
    a dominant gesture on the consonant's own articulators and weak ones on the rest. Movement latencies are
    compensated so the acoustic closure and release land on ``a`` and ``b``."""
    dom = {k: x for k, x in tg.items() if k in d.dominant}
    sec = {k: x for k, x in tg.items() if k not in d.dominant}
    # the dorsum is heavy: its releases are slower than tip and lip releases
    body_only = all(k in ("tbX", "tbY", "root") for k in d.dominant)
    sp = 1.0 + (speed - 1.0) * 0.35 if body_only else speed
    closing = d.manner in _CLOSING
    lips_only = all(k in ("lipAp", "labiodental") for k in d.dominant)
    # articulators need 10-15 ms to climb out of the overshoot (contact) target, so the gesture releases early
    # and the acoustic release lands on the planned boundary (measured closure/release lags per articulator)
    lead = (0.034 if body_only else 0.019 if lips_only else 0.02) if closing else 0.0
    early = (0.024 if body_only else 0.022 if lips_only else 0.031) if closing else 0.008
    if dom:
        # a consonant owns its constriction articulators: at strength 40 the overlapping vowel contributes
        # about 2 %, so closures seal instead of being diluted
        out.append(_g(a - early, js_max(a + 0.01, b - lead), 0.03, 0.03 if body_only else 0.018, 40.0, dom, label,
                      speed=sp, release_hold=0.045))
    # alveolars front the tongue body during the closure (the alveolar F2 locus, about 1.8 kHz)
    if d.place == "alveolar" and "tbX" not in tg and d.manner in _ALVEOLAR_FRONTED:
        out.append(_g(a, b, 0.05, 0.05, 1.2, {"tbX": 3.2, "tbY": 7.4}, label))
    if sec:
        # a velar's front-back place follows the vowel (fronted /k/ in "key", backed in "caw"): its own tbX
        # only weakly, so the F2 locus moves with the vowel as in natural speech
        place = sec.pop("tbX", None) if body_only else None
        if sec:
            out.append(_g(a, b, 0.05, 0.05, 2.2, sec, label))
        if place is not None:
            out.append(_g(a, b, 0.05, 0.05, 0.25, {"tbX": place}, label))


def plan_utterance(segs: Sequence[Segment], opts: PlanOptions) -> Plan:
    """One breath group: Klatt durations, diphthongs split, vowel and consonant gestures, phonation, breath,
    end of breath, intonation (planner.rs:181-399). Times are on the voice clock; the first segment starts at
    ``opts.t0 + 0.06``."""
    o = opts
    rate = o.rate if o.rate is not None else 1.0
    body = BodyMix.parse(o.body)
    # durations by the Klatt rules, on whole segments (diphthongs before they are split)
    durs = klatt_durations(segs, PlanOptions(body, o.t0))
    ss: list[_Seg] = []
    for si, s in enumerate(segs):
        parts = _EXPAND.get(s.ipa)
        if parts is not None:
            base = durs[si]
            for i, p in enumerate(parts):
                dur = base * (0.55 if i == 0 else 0.45) if len(parts) > 1 else base
                ss.append(_Seg(p, dur, s.stress, s.f0, i > 0))
        else:
            ss.append(_Seg(s.ipa, durs[si], s.stress, s.f0, False))
    for s in ss:
        v = vowel_artic(s.ipa, body)
        if v is not None:
            s.vowel = v
        else:
            s.defn = phoneme(s.ipa, body)

    gestures: list[Gesture] = []
    planned: list[PlannedSegment] = []
    t = o.t0 + 0.06                      # breath onset lead
    times: list[tuple[float, float]] = []
    for s in ss:
        if s.dur is not None:
            d = s.dur
        elif s.vowel is not None:
            d = 0.17 if s.stress == 1 else 0.13 if s.stress == 2 else 0.095
        else:
            d = s.defn.dur if s.defn is not None else 0.08
        d /= rate
        times.append((t, t + d))
        t += d
    end = t

    # vowel gestures overlap their neighbours, so the tongue travels vowel to vowel through the consonants
    # (Öhman): the previous vowel lets go early in the gap and the next one ramps in over most of it
    v_idx = [i for i, s in enumerate(ss) if s.vowel is not None]
    for n, i in enumerate(v_idx):
        prev = times[v_idx[n - 1]][1] if n > 0 else times[i][0] - 0.08
        nxt = times[v_idx[n + 1]][0] if n < len(v_idx) - 1 else times[i][1] + 0.08
        r_in = js_max(0.03, 0.75 * (times[i][0] - prev)) + 0.01
        r_out = js_max(0.03, 0.4 * (nxt - times[i][1])) + 0.01
        gestures.append(_g(times[i][0], times[i][1], r_in, r_out, 1.0, dict(ss[i].vowel), ss[i].ipa))

    # consonant gestures
    for i, s in enumerate(ss):
        t0, t1 = times[i]
        dur = t1 - t0
        if s.vowel is not None:
            planned.append(PlannedSegment(s.ipa, "vowel", t0, t1, s.stress if s.stress is not None else 0))
        else:
            planned.append(PlannedSegment(s.ipa, s.defn.manner.value if s.defn is not None else "?", t0, t1))
        d = s.defn
        if d is None:
            continue
        next_vowel = i < len(ss) - 1 and ss[i + 1].vowel is not None
        prev_seg = ss[i - 1] if i > 0 else None
        m = d.manner
        if m is Manner.STOP and _pre_placed_behind_closure(ss, i):
            # closing behind another stop with its articulator already placed it reaches contact sooner; the
            # delay leaves the closure 30 ms before its release, so a short sung stop keeps its closure inside
            oral_gestures_into(d, d.targets, t0 + js_min(_PLACED_CLOSURE_DELAY, js_max(0.0, dur - 0.03)), t1, 1.55,
                               d.ipa, gestures)
        elif m is Manner.STOP or m is Manner.NASAL:
            oral_gestures_into(d, d.targets, t0, t1, 1.55, d.ipa, gestures)
        elif m is Manner.TAP:
            oral_gestures_into(d, d.targets, t0, t1, 2.8, d.ipa, gestures)
        elif m is Manner.TRILL:
            # driven trill: rapid alternation of near-closure and release at about 26 Hz
            n_pulses = js_max(2.0, js_round(dur * 26.0))
            k = 0.0
            while k < n_pulses:
                a = t0 + (k / n_pulses) * dur
                oral_gestures_into(d, d.targets, a, a + 0.35 / 26.0, 3.2, d.ipa, gestures)
                k += 1.0
        elif m is Manner.AFFRICATE:
            split = t0 + dur * 0.55
            oral_gestures_into(d, d.targets, t0, split, 2.0, d.ipa, gestures)
            rel = phoneme(d.release, body) if d.release is not None else None
            if rel is not None:
                oral_gestures_into(d, rel.targets, split + 0.008, t1, 1.4, d.ipa, gestures)
        elif m in (Manner.FRICATIVE, Manner.APPROXIMANT, Manner.LATERAL):
            oral_gestures_into(d, d.targets, t0, t1, 1.3, d.ipa, gestures)
        # nasality: the velum lowers in anticipation (into the preceding vowel) and seals again by the end of
        # the nasal; 0.45 cm² is ample for the murmur
        if d.nasal is True:
            gestures.append(_g(t0 - 0.02, t1 - 0.02, 0.05, 0.03, 4.0, {"velum": 0.45}, "velum"))
        # phonation (abduction is body-scaled in the voice)
        if d.glottal is Glottal.OPEN:
            gestures.append(_g(t0 - 0.01, t1, 0.03, 0.035, 5.0, {"abduction": 0.33 * 1.0}, "h"))
        elif d.glottal is Glottal.BREATHY:
            gestures.append(_g(t0, t1, 0.03, 0.03, 5.0, {"abduction": 0.1 * 1.0}, "ɦ"))
        elif d.glottal is Glottal.STOP:
            gestures.append(_g(t0, t1, 0.012, 0.015, 8.0, {"closure": 1.0}, "ʔ"))
        elif d.voicing is Voicing.VOICELESS:
            if m is Manner.STOP or m is Manner.AFFRICATE:
                # aspirated before a vowel and not after /s/ (English-like)
                aspirated = next_vowel and not (prev_seg is not None and prev_seg.ipa == "s")
                vot = 0.05 / rate if aspirated else 0.012
                gestures.append(_g(t0 + 0.25 * dur, t1 + vot * 0.4, 0.035, 0.045 if aspirated else 0.02, 6.0,
                                   {"abduction": 0.36 if aspirated else 0.22}, "aspiration"))
            else:
                gestures.append(_g(t0 - 0.015, t1 + 0.005, 0.03, 0.03, 6.0, {"abduction": 0.3}, "voiceless"))
        elif m is Manner.FRICATIVE:
            gestures.append(_g(t0, t1, 0.03, 0.03, 3.0, {"abduction": 0.035}, "voiced frication"))

    open_end = o.open_end is True
    sung = o.sung is True
    # breath on through the utterance. A sung phrase's onset is planned (its first vowel lands on the beat), so
    # the pressure reaches full drive as the first segment starts; speech and live notes keep a 20 ms lead.
    breath_at = _BREATH_SUNG_AT if sung else 0.04
    gestures.append(_g(o.t0 + breath_at, end, 0.04, 0.07 if open_end else 0.05, 1.0, {"breath": 1.0}, "breath"))
    # end of the breath group: the folds part as the breath stops, so a final obstruent is released into
    # silence instead of into an added vowel; after a sonorant they part once the pressure has fallen
    if not open_end:
        last = ss[-1] if ss else None
        sonorant = last is not None and (last.vowel is not None or (last.defn is not None and last.defn.manner in (
            Manner.NASAL, Manner.APPROXIMANT, Manner.LATERAL, Manner.TAP, Manner.TRILL)))
        at = end + (0.045 if sonorant else 0.015)
        gestures.append(_g(at, end + 0.1, 0.03, 0.06, 6.0, {"abduction": 0.22}, EXHALE_LABEL))

    # intonation: pitch targets on vowels
    f0 = o.f0 if o.f0 is not None else 110.0
    accent = o.accent if o.accent is not None else 0.14
    total = js_max(0.2, end - o.t0)
    # the nuclear accent: the phrase's last primary-stressed vowel, a bigger excursion that falls (statements)
    # into the final low; earlier accents are smaller (a "hat" pattern)
    nuclear = -1
    for i in v_idx:
        if ss[i].stress == 1 and not ss[i].glide_from:
            nuclear = i
    intonation = o.intonation if o.intonation is not None else Intonation.STATEMENT
    for n, i in enumerate(v_idx):
        s = ss[i]
        t0, t1 = times[i]
        # pitch holds through the consonants up to the next vowel (instead of sagging to the base)
        hold = js_max(t1, times[v_idx[n + 1]][0] - 0.03) if n < len(v_idx) - 1 else t1
        has_f0 = _truthy(s.f0)
        if has_f0:
            target = s.f0
        else:
            prog = (t0 - o.t0) / total
            decl = 1.0 - 0.1 * prog
            is_nuclear = i == nuclear
            if s.stress == 1:
                acc = 1.0 + (accent * 1.8 if is_nuclear else accent)
            elif s.stress == 2:
                acc = 1.0 + accent * 0.5
            else:
                acc = 1.0
            target = f0 * decl * acc
            last = n == len(v_idx) - 1
            if last and o.intonation is Intonation.QUESTION:
                target = f0 * 1.35
            if last and intonation is Intonation.STATEMENT:
                if is_nuclear and t1 - t0 > 0.08:
                    # accent peak early in the vowel, then the fall within it
                    gestures.append(_g(t0, t0 + 0.35 * (t1 - t0), 0.05, 0.05, 1.0, {"f0": target}, "pitch"))
                    gestures.append(_g(t0 + 0.55 * (t1 - t0), t1, 0.06, 0.06, 1.0, {"f0": f0 * 0.82}, "pitch"))
                    continue
                target = f0 * 0.82
        r = _pitch_ramp(t1 - t0) if has_f0 and sung else 0.06
        gestures.append(_g(t0, t1 if has_f0 else hold, r, r, 6.0 if has_f0 else 1.0, {"f0": target}, "pitch"))
    # sung: a consonant given its note's pitch holds it, so the pitch moves during an onset consonant and has
    # arrived when the vowel starts
    if sung:
        for i, s in enumerate(ss):
            if s.vowel is None and _truthy(s.f0):
                t0, t1 = times[i]
                r = _pitch_ramp(t1 - t0)
                gestures.append(_g(t0, t1, r, r, 6.0, {"f0": s.f0}, "pitch"))
    return Plan(gestures, planned, end)


def plan_phrases(phrases: Sequence[PhraseInput], opts: PlanOptions) -> Plan:
    """Consecutive phrases, each its own breath group with its own intonation (missing: statement), the next
    starting after the previous one's pause (default 0.25 s). ``opts.intonation`` is ignored."""
    out = Plan([], [], opts.t0)
    t = opts.t0
    for p in phrases:
        intonation = p.intonation if p.intonation is not None else Intonation.STATEMENT
        plan = plan_utterance(p.segments, dataclasses.replace(opts, t0=t, intonation=intonation))
        out.gestures.extend(plan.gestures)
        out.segments.extend(plan.segments)
        t = plan.end + (p.pause if p.pause is not None else 0.25)
        out.end = plan.end
    return out
