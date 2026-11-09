"""Singing: syllables on notes through the speech planner; live notes (port of phonetics/sing.rs).

- A note's vowel lands on the beat. Its onset consonants are sung just before it (they take their time from the
  previous note, as singers do), its coda just before the next onset.
- Diphthongs sustain their first vowel and glide to the second at the end of the note.
- A melisma ("-") carries the vowel onto further notes; the coda waits for the last of them.
- Vibrato blooms a quarter of a second into a held vowel.

Live playing uses open-ended notes: a key press sings the onset and starts a vowel that lasts until the key is
released (tagged gestures, ended by releasing the tag), and the release sings the glide and the coda.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.body import BodyMix
from ..core.jsmath import js_max, js_min, midi_to_hz
from ..core.types import Gesture, Intonation, Plan, PlanOptions, Segment, SongOptions, SungNote, Syllable
from .inventory import phoneme
from .planner import diphthong_parts, plan_utterance

# a sung vowel keeps at least this long when a short note squeezes its consonants (s)
MIN_VOWEL = 0.05
# a squeezed consonant keeps at least this long (s); a flap is 16 ms
MIN_CONSONANT = 0.012
# live notes: gestures relative to "now" (the planner's 60 ms lead is mostly taken back)
LIVE_T0 = -0.045
# a held live vowel lasts at most this long (s)
HOLD = 30.0


@dataclass
class LegatoFrom:
    """The previous note of a legato pair (sing.rs:194-200)."""

    syl: Syllable
    midi: float

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "LegatoFrom":
        return cls(Syllable.from_json(d["syl"]), d["midi"])

    def to_json(self) -> dict:
        return {"syl": self.syl.to_json(), "midi": self.midi}


@dataclass
class NoteOnOptions:
    """sing.rs:201-210."""

    body: BodyMix
    vibrato: float
    tag: float
    legato_from: LegatoFrom | None = None

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "NoteOnOptions":
        lf = d.get("legatoFrom")
        return cls(BodyMix.parse(d.get("body", "male")), d["vibrato"], d["tag"],
                   LegatoFrom.from_json(lf) if lf is not None else None)

    def to_json(self) -> dict:
        out: dict[str, Any] = {"body": self.body.to_json(), "vibrato": self.vibrato, "tag": self.tag}
        if self.legato_from is not None:
            out["legatoFrom"] = self.legato_from.to_json()
        return out


@dataclass(frozen=True)
class MelismaOptions:
    """sing.rs:230-236."""

    vibrato: float
    tag: float

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "MelismaOptions":
        return cls(d["vibrato"], d["tag"])

    def to_json(self) -> dict:
        return {"vibrato": self.vibrato, "tag": self.tag}


def _cdur(ipa: str, body: BodyMix, k: float) -> float:
    """Sung consonants are a little shorter than spoken ones (inventory duration times ``k``)."""
    d = phoneme(ipa, body)
    return (d.dur if d is not None else 0.07) * k


def _consonants(segs: Sequence[Segment], body: BodyMix, k: float) -> list[Segment]:
    return [Segment(c.ipa, dur=_cdur(c.ipa, body, k)) for c in segs]


def _consonants_at(segs: Sequence[Segment], body: BodyMix, k: float, hz: float) -> list[Segment]:
    """Sung consonants at their note's pitch: an onset consonant moves the pitch to the new note before its
    vowel (as singers do), a coda holds the old one."""
    out = _consonants(segs, body, k)
    for c in out:
        c.f0 = hz
    return out


def _total(segs: Sequence[Segment]) -> float:
    t = 0.0
    for c in segs:
        t = t + (c.dur if c.dur is not None else 0.0)
    return t


def _fit(a: list[Segment], b: list[Segment], span: float) -> None:
    """Fit the consonants between a vowel and the next one into ``span`` (beat to next beat, or to the end of a
    phrase's last note): when they leave the vowel less than MIN_VOWEL they shrink in proportion, each down to
    MIN_CONSONANT, so the next vowel still lands on its beat and a fast run does not drift note by note."""
    cons = _total(a) + _total(b)
    room = span - MIN_VOWEL
    if cons <= room or cons <= 0.0:
        return
    scale = js_max(0.0, room) / cons
    for c in a + b:
        c.dur = js_max(MIN_CONSONANT, (c.dur if c.dur is not None else 0.0) * scale)


def _melisma(s: Syllable) -> bool:
    return s.melisma is True


def _vibrato_gesture(t0: float, t1: float, depth: float, tag: float | None) -> Gesture:
    return Gesture(t0 + 0.22, t1, 0.3, 0.06, 4.0, {"vibrato": depth}, label="vibrato", tag=tag)


def _last_of_chain(notes: Sequence[SungNote], k: int) -> Syllable:
    """The syllable whose coda ends a melisma chain that reaches note ``k``."""
    b = k
    while b > 0 and _melisma(notes[b].syl):
        b -= 1
    return notes[b].syl


def plan_song(notes: Sequence[SungNote], opts: SongOptions) -> Plan:
    """A whole melody, one utterance (and breath) per phrase; a phrase ends at a syllable with ``phrase_end``.
    ``opts.t0`` is the time of the first note's beat (its onset consonants come before it)."""
    o = opts
    body = BodyMix.parse(o.body)
    beat = 60.0 / o.bpm
    gestures: list[Gesture] = []
    segments = []
    times: list[float] = []
    t = o.t0
    for n in notes:
        times.append(t)
        t += n.beats * beat
    end = t
    i = 0
    while i < len(notes):
        j = i
        while j < len(notes) - 1 and not notes[j].syl.phrase_end:
            j += 1
        segs: list[Segment] = []
        vowel_spans: list[tuple[float, float]] = []

        def hz_of(k: int) -> float:
            return midi_to_hz(notes[k].midi)

        # each note's onset and coda consonants at the note's pitch (a coda at the pitch of the note whose
        # syllable it ends; a melisma has neither)
        onsets = [[] if _melisma(notes[k].syl) else _consonants_at(notes[k].syl.onset, body, 0.8, hz_of(k))
                  for k in range(i, j + 1)]
        codas = [[] if k < j and _melisma(notes[k + 1].syl)
                 else _consonants_at(_last_of_chain(notes, k).coda, body, 0.8, hz_of(k)) for k in range(i, j + 1)]
        # the phrase's last vowel leaves a short breath before the next phrase; the call's last phrase is
        # followed by a rest (callers split phrases at rests), so it holds to its end less the breath's release
        reserve = 0.05 if j == len(notes) - 1 else 0.12
        last_end = times[j] + notes[j].beats * beat - js_min(reserve, 0.3 * notes[j].beats * beat)
        for k in range(i, j + 1):
            if k < j:
                _fit(codas[k - i], onsets[k + 1 - i], times[k + 1] - times[k])
            else:
                _fit(codas[k - i], [], last_end - times[k])
        first_onset = _total(onsets[0])
        for k in range(i, j + 1):
            n = notes[k]
            hz = hz_of(k)
            next_melisma = k < j and _melisma(notes[k + 1].syl)
            coda, onset = codas[k - i], onsets[k - i]
            # vowel from this note's beat to the next note's onset (less this note's coda); only consonants
            # squeezed to their floor can leave less than MIN_VOWEL
            note_end = times[k + 1] - _total(onsets[k + 1 - i]) if k < j else last_end
            vdur = js_max(0.02, note_end - times[k] - _total(coda))
            segs.extend(onset)
            parts = diphthong_parts(n.syl.nucleus.ipa)
            if parts is not None and not next_melisma:
                segs.append(Segment(parts[0], dur=vdur * 0.8, f0=hz, stress=1))
                segs.append(Segment(parts[1], dur=vdur * 0.2, f0=hz, stress=1))
            else:
                segs.append(Segment(parts[0] if parts is not None else n.syl.nucleus.ipa, dur=vdur, f0=hz, stress=1))
            segs.extend(coda)
            vowel_spans.append((times[k], times[k] + vdur))
        if segs:
            # the planner lays segments out from t0 + 0.06: start so the first vowel lands on its beat
            plan = plan_utterance(segs, PlanOptions(body, times[i] - first_onset - 0.06, f0=hz_of(i),
                                                    intonation=Intonation.FLAT, sung=True))
            gestures.extend(plan.gestures)
            segments.extend(plan.segments)
            for a, b in vowel_spans:
                if b - a > 0.3:
                    gestures.append(_vibrato_gesture(a, b, o.vibrato, None))
            end = js_max(end, plan.end)
        i = j + 1
    return Plan(gestures, segments, end)


def _tail(syl: Syllable, body: BodyMix, hz: float) -> list[Segment]:
    """What follows a held vowel: the diphthong's glide (at the note's pitch) and the coda."""
    parts = diphthong_parts(syl.nucleus.ipa)
    out: list[Segment] = []
    if parts is not None:
        out.append(Segment(parts[1], dur=0.07, stress=1, f0=hz))
    out.extend(_consonants(syl.coda, body, 0.7))
    return out


def plan_note_on(syl: Syllable, midi: float, o: NoteOnOptions) -> list[Gesture]:
    """Key pressed: the previous syllable's glide and coda if it is still sounding (legato), then this
    syllable's onset, then its vowel held open-ended. Every gesture gets ``o.tag``; pitch and vibrato get
    ``o.tag + 1``, so a melisma can change the pitch without ending the vowel."""
    body = BodyMix.parse(o.body)
    hz = midi_to_hz(midi)
    segs: list[Segment] = []
    if o.legato_from is not None:
        segs.extend(_tail(o.legato_from.syl, body, midi_to_hz(o.legato_from.midi)))
    segs.extend(_consonants(syl.onset, body, 0.7))
    parts = diphthong_parts(syl.nucleus.ipa)
    segs.append(Segment(parts[0] if parts is not None else syl.nucleus.ipa, dur=HOLD, f0=hz, stress=1))
    plan = plan_utterance(segs, PlanOptions(body, LIVE_T0, f0=hz, intonation=Intonation.FLAT, open_end=True))
    vowel = plan.segments[-1]
    out = [dataclasses.replace(g, tag=o.tag + 1.0 if "f0" in g.targets else o.tag) for g in plan.gestures]
    out.append(_vibrato_gesture(vowel.t0, vowel.t1, o.vibrato, o.tag + 1.0))
    return out


def plan_melisma(midi: float, o: MelismaOptions) -> list[Gesture]:
    """Live melisma: the held vowel moves to a new pitch (release ``tag + 1`` first)."""
    return [Gesture(0.0, HOLD, 0.05, 0.06, 6.0, {"f0": midi_to_hz(midi)}, label="pitch", tag=o.tag + 1.0),
            _vibrato_gesture(0.0, HOLD, o.vibrato, o.tag + 1.0)]


def plan_note_off(syl: Syllable, midi: float, body: BodyMix) -> list[Gesture]:
    """Key released: the diphthong's glide and the coda (release the note's tags first); empty if none."""
    body = BodyMix.parse(body)
    segs = _tail(syl, body, midi_to_hz(midi))
    if not segs:
        return []
    return plan_utterance(segs, PlanOptions(body, LIVE_T0, f0=midi_to_hz(midi), intonation=Intonation.FLAT)).gestures
