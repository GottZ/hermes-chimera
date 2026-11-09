"""Score format and validation: precise issues with a JSON path and the fix. No numpy import
at module level (the size checks compile the score, which may use numpy for dyn envelopes)."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..core.body import BODY_NAMES, BodyMix

SCORE_SCHEMA: dict = {
    "type": "object",
    "required": ["bpm", "voices"],
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "bpm": {"type": "number", "minimum": 30, "maximum": 240},
        "lead_in": {"type": "number", "minimum": 0, "maximum": 2},
        "normalize": {"type": "boolean"},
        "stereo": {"enum": ["auto", True, False]},
        "format": {"enum": ["auto", "wav", "mp3"]},
        "voices": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["notes"], "properties": {
            "name": {"type": "string", "maxLength": 40}, "body": {}, "start": {"type": "number", "minimum": 0},
            "volume": {"type": "number", "minimum": -40, "maximum": 12},
            "pan": {"type": "number", "minimum": -1, "maximum": 1},
            "breath": {"type": "number", "minimum": 3, "maximum": 20},
            "tension": {"type": "number", "minimum": 0, "maximum": 1},
            "vibrato": {"type": "number", "minimum": 0, "maximum": 100},
            "vibrato_rate": {"type": "number", "minimum": 3.5, "maximum": 7.5},
            "breathiness": {"type": "number", "minimum": 0, "maximum": 1},
            "brightness": {"type": "number", "minimum": 0, "maximum": 1},
            "roughness": {"type": "number", "minimum": 0, "maximum": 1},
            "ring": {"type": "number", "minimum": 0, "maximum": 1},
            "seed": {"type": ["integer", "null"], "minimum": 0, "maximum": 4294967295},
            "lyrics": {"type": ["string", "null"]},
            "notes": {"type": "array", "minItems": 1, "maxItems": 2000, "items": {"type": "object", "properties": {
                "pitch": {"type": ["string", "number"]}, "beats": {"type": "number", "exclusiveMinimum": 0},
                "lyric": {"type": "string"}, "dyn": {"enum": ["pp", "p", "mp", "mf", "f", "ff"]},
                "rest": {"type": "number", "exclusiveMinimum": 0}, "say": {"type": "string"},
                "rate": {"type": "number", "minimum": 0.5, "maximum": 1.8},
                "hz": {"type": "number", "minimum": 50, "maximum": 600}}}}}}},
    },
}

NATURAL_RANGE_MIDI: Mapping[str, tuple[int, int]] = {"male": (38, 69), "female": (53, 84), "child": (57, 81)}
    # D2–A4, F3–C6, A3–A5: convention, warnings only; blends interpolate the bounds by BodyMix weights

MIDI_MIN, MIDI_MAX = 28, 100     # every body hits these pitches within a few cents (measured on the template)
DYN_LEVELS: tuple[str, ...] = ("pp", "p", "mp", "mf", "f", "ff")
LEAD_IN_MARGIN = 0.005
TOP_KEYS = ("title", "bpm", "lead_in", "normalize", "stereo", "format", "voices")
VOICE_KEYS = ("name", "body", "start", "volume", "pan", "breath", "tension", "vibrato", "vibrato_rate",
              "breathiness", "brightness", "roughness", "ring", "seed", "lyrics", "notes")
NOTE_KEYS = ("pitch", "beats", "lyric", "dyn", "rest", "say", "rate", "hz")
VOICE_DEFAULTS = {"start": 0, "volume": 0, "pan": 0, "breath": 9, "tension": 0.55, "vibrato": 35,
                  "vibrato_rate": 6.0, "breathiness": 0.15, "brightness": 0.25, "roughness": 0.1, "ring": 0,
                  "seed": None, "lyrics": None}
VOICE_RANGES = {"start": (0, 10000, "beats"), "volume": (-40, 12, "dB"), "pan": (-1, 1, ""),
                "breath": (3, 20, "cmH2O"), "tension": (0, 1, ""), "vibrato": (0, 100, "cents"),
                "vibrato_rate": (3.5, 7.5, "Hz"), "breathiness": (0, 1, ""), "brightness": (0, 1, ""),
                "roughness": (0, 1, ""), "ring": (0, 1, "")}


@dataclass(frozen=True)
class ScoreIssue:
    path: str                      # JSON path, e.g. "voices[1].notes[7].pitch"
    severity: Literal["error", "warning"]
    message: str                   # one sentence, says what is wrong AND what to write instead


@dataclass(frozen=True)
class VoiceSummary:
    name: str
    body: str
    items: int
    sounding: int
    spoken: int
    first_beat: float
    end_beat: float
    seconds: float
    lowest: str
    highest: str
    syllables: list[str]           # per sounding note as the engine will sing it ("~" = melisma)


@dataclass(frozen=True)
class ScoreReport:
    ok: bool
    issues: list[ScoreIssue]
    voices: list[VoiceSummary]
    seconds: float                 # mix length incl. lead_in and release tail
    voice_seconds: float           # Σ RENDERED voice lengths (start_block … end + 0.4 s)
    lead_in: float                 # effective (possibly raised) lead_in
    memory_mb: float
    disk_mb: float
    estimate_s: dict[str, float]
    normalized: dict               # score with all defaults filled, pitches as MIDI floats


# --- pitch, body ---------------------------------------------------------------------------------------------

_PITCH = re.compile(r"^([A-Ga-g])(##|bb|#|b)?(-?\d)([+-]\d+(?:\.\d+)?)?$")
_SEMI = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_ACC = {None: 0, "#": 1, "##": 2, "b": -1, "bb": -2}
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_pitch(value: Any) -> float:
    """Note name ``<letter>[#|b|##|bb]<octave>[±cents]`` (C4 = 60) or a MIDI number -> MIDI float; ValueError if invalid."""
    if _is_num(value):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"{value!r} is not a note name or MIDI number")
    m = _PITCH.match(value)
    if not m:
        raise ValueError(f"{value!r} is not a note name")
    letter, acc, octave, cents = m.groups()
    midi = (int(octave) + 1) * 12 + _SEMI[letter.lower()] + _ACC[acc]
    return float(midi) + (float(cents) / 100.0 if cents else 0.0)


def midi_name(midi: float) -> str:
    """MIDI float -> note name with sharps and cents if fractional, e.g. 61 -> "C#4", 69.15 -> "A4+15"."""
    base = int(math.floor(midi + 0.5))
    cents = int(round((midi - base) * 100))
    name = f"{_NAMES[base % 12]}{base // 12 - 1}"
    return name + (f"{cents:+d}" if cents else "")


def parse_body(spec: Any) -> BodyMix:
    """Score body: "male"|"female"|"child"|{"female","child"}|"female=0.5,child=0.2" -> BodyMix; ValueError if invalid.
    ValueError.args[1] (when present) names the offending key."""
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in BODY_NAMES:
            return BodyMix.parse(s)
        if "=" in s:
            vals: dict[str, float] = {}
            for part in s.split(","):
                k, _, v = part.partition("=")
                k = k.strip()
                if k not in ("female", "child") or k in vals:
                    raise ValueError(f"unknown body part {k!r}", None)
                try:
                    x = float(v)
                except ValueError:
                    raise ValueError(f"{v!r} is not a number", None) from None
                if not 0.0 <= x <= 1.0:
                    raise ValueError(f"{k}={x:g} is outside 0..1", None)
                vals[k] = x
            return BodyMix(vals.get("female", 0.0), vals.get("child", 0.0))
        raise ValueError(f"{spec!r} is not a body", None)
    if isinstance(spec, Mapping):
        for k, v in spec.items():
            if k not in ("female", "child"):
                raise ValueError(f"unknown body part {k!r}", k)
            if not _is_num(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{k} must be a number 0..1", k)
        return BodyMix(float(spec.get("female", 0.0)), float(spec.get("child", 0.0)))
    raise ValueError(f"{spec!r} is not a body", None)


def body_text(mix: BodyMix) -> str:
    p = mix.preset()
    return p if p is not None else f"female={mix.female:g},child={mix.child:g}"


def natural_range(mix: BodyMix) -> tuple[float, float]:
    w = mix.weights()
    lo = sum(wi * NATURAL_RANGE_MIDI[b][0] for wi, b in zip(w, BODY_NAMES))
    hi = sum(wi * NATURAL_RANGE_MIDI[b][1] for wi, b in zip(w, BODY_NAMES))
    return lo, hi


# --- validation --------------------------------------------------------------------------------------------------

@dataclass
class _Ctx:
    issues: list[ScoreIssue] = field(default_factory=list)

    def err(self, path: str, msg: str) -> None:
        self.issues.append(ScoreIssue(path, "error", msg))

    def warn(self, path: str, msg: str) -> None:
        self.issues.append(ScoreIssue(path, "warning", msg))

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")


def _unknown(ctx: _Ctx, path: str, key: str, allowed: tuple[str, ...]) -> None:
    ctx.err(path, f'unknown field "{key}" - remove it; allowed here: {", ".join(allowed)}.')


def _number(ctx: _Ctx, path: str, v: Any, lo: float, hi: float, unit: str = "", *, low_open: bool = False) -> bool:
    u = f" {unit}" if unit else ""
    if not _is_num(v):
        ctx.err(path, f"{v!r} is not a number - write a number from {lo:g} to {hi:g}{u}.")
        return False
    if v < lo or v > hi or (low_open and v <= lo):
        rng = f"above {lo:g} and at most {hi:g}" if low_open else f"from {lo:g} to {hi:g}"
        ctx.err(path, f"{v:g}{u} is out of range - use a value {rng}{u}.")
        return False
    return True


def validate_score(score: "dict | str", *, g2p: Any, max_voices: int = 8,
                   max_seconds: float = 300.0, max_voice_seconds: float = 1200.0) -> ScoreReport:
    """Check types, ranges, syllable counts, spoken slots and size; raise lead_in when an onset would start before 0.005 s."""
    ctx = _Ctx()

    def fail() -> ScoreReport:
        return ScoreReport(False, ctx.issues, [], 0.0, 0.0, 0.0, 0.0, 0.0, {}, {})

    if isinstance(score, str):
        try:
            score = json.loads(score)
        except ValueError as e:
            ctx.err("score", f"the score is not valid JSON ({e.msg} at line {e.lineno}) - write one JSON object with "
                             f'"bpm" and "voices".')
            return fail()
    if not isinstance(score, Mapping):
        ctx.err("score", f'the score must be a JSON object, not {type(score).__name__} - write {{"bpm": …, '
                         f'"voices": […]}}.')
        return fail()
    for k in score:
        if k not in TOP_KEYS:
            _unknown(ctx, str(k), str(k), TOP_KEYS)
    norm: dict[str, Any] = {}
    if "title" in score:
        t = score["title"]
        if not isinstance(t, str) or len(t) > 120:
            ctx.err("title", "the title must be a text of at most 120 characters - shorten it or remove it.")
        else:
            norm["title"] = t
    if "bpm" not in score:
        ctx.err("bpm", 'the tempo is missing - add "bpm" (beats per minute, 30 to 240).')
    elif _number(ctx, "bpm", score["bpm"], 30, 240, "bpm"):
        norm["bpm"] = score["bpm"]
    lead_in = score.get("lead_in", 0.3)
    if _number(ctx, "lead_in", lead_in, 0, 2, "s"):
        norm["lead_in"] = float(lead_in)
    for key, allowed, default in (("normalize", (True, False), True), ("stereo", ("auto", True, False), "auto"),
                                  ("format", ("auto", "wav", "mp3"), "auto")):
        v = score.get(key, default)
        if not any(v is a or (isinstance(a, str) and v == a) for a in allowed):
            ctx.err(key, f'{json.dumps(v)} is not allowed for "{key}" - use one of '
                         f'{", ".join(json.dumps(a) for a in allowed)}.')
        else:
            norm[key] = v
    voices = score.get("voices")
    if voices is None:
        ctx.err("voices", 'the score has no voices - add "voices": a list with at least one voice.')
        return fail()
    if not isinstance(voices, list):
        ctx.err("voices", '"voices" must be a list of voice objects - write "voices": [{"notes": […]}].')
        return fail()
    if not voices:
        ctx.err("voices", '"voices" is empty - add at least one voice with "notes".')
        return fail()
    if len(voices) > max_voices:
        ctx.err("voices", f"{len(voices)} voices are more than the limit of {max_voices} - use at most {max_voices} "
                          f"voices.")
        return fail()
    norm_voices = []
    for i, v in enumerate(voices):
        nv = _check_voice(ctx, i, v)
        norm_voices.append(nv)
    if ctx.n_errors:
        return fail()
    norm["voices"] = norm_voices
    return _check_music(ctx, norm, g2p, max_seconds, max_voice_seconds)


def _check_voice(ctx: _Ctx, i: int, v: Any) -> dict | None:
    p = f"voices[{i}]"
    if not isinstance(v, Mapping):
        ctx.err(p, f'a voice must be an object like {{"body": "female", "notes": […]}} - replace {json.dumps(v)}.')
        return None
    before = ctx.n_errors
    for k in v:
        if k not in VOICE_KEYS:
            _unknown(ctx, f"{p}.{k}", str(k), VOICE_KEYS)
    nv: dict[str, Any] = {}
    name = v.get("name", f"voice{i}")
    if not isinstance(name, str) or len(name) > 40:
        ctx.err(f"{p}.name", "the name must be a text of at most 40 characters - shorten it.")
    nv["name"] = name
    try:
        mix = parse_body(v.get("body", "male"))
        nv["body"] = mix.preset() or {"female": mix.female, "child": mix.child}
    except ValueError as e:
        sub = e.args[1] if len(e.args) > 1 and e.args[1] is not None else None
        ctx.err(f"{p}.body.{sub}" if sub else f"{p}.body",
                f"{e.args[0]} - use \"male\", \"female\", \"child\" or a mix like {{\"female\": 0.5, \"child\": 0.2}} "
                f"(values 0 to 1).")
    for k, (lo, hi, unit) in VOICE_RANGES.items():
        val = v.get(k, VOICE_DEFAULTS[k])
        if _number(ctx, f"{p}.{k}", val, lo, hi, unit):
            nv[k] = val
    seed = v.get("seed")
    if seed is not None and not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= 0xFFFFFFFF):
        ctx.err(f"{p}.seed", f"{json.dumps(seed)} is not a seed - use a whole number from 0 to 4294967295 or null.")
    nv["seed"] = seed
    lyrics = v.get("lyrics")
    if lyrics is not None and not isinstance(lyrics, str):
        ctx.err(f"{p}.lyrics", '"lyrics" must be a text (one syllable per sounding note) - write a string or remove it.')
    nv["lyrics"] = lyrics
    notes = v.get("notes")
    if notes is None:
        ctx.err(f"{p}.notes", 'the voice has no notes - add "notes": a list of {"pitch", "beats", "lyric"} items.')
        return None
    if not isinstance(notes, list) or not notes:
        ctx.err(f"{p}.notes", '"notes" must be a non-empty list - add at least one {"pitch": "C4", "beats": 1} item.')
        return None
    if len(notes) > 2000:
        ctx.err(f"{p}.notes", f"{len(notes)} items are more than the limit of 2000 - split the voice into shorter "
                              f"parts.")
        return None
    out_notes = []
    for k, n in enumerate(notes):
        out_notes.append(_check_note(ctx, f"{p}.notes[{k}]", n))
    nv["notes"] = out_notes
    if ctx.n_errors > before:
        return None
    sounding = [(k, n) for k, n in enumerate(out_notes) if "pitch" in n]
    with_lyric = [k for k, n in sounding if n.get("lyric")]
    if lyrics is not None and with_lyric:
        ctx.err(f"{p}.lyrics", 'give either "lyrics" or a "lyric" per note, not both - remove "lyrics" or the '
                               'per-note lyrics.')
        return None
    if with_lyric and len(with_lyric) != len(sounding):
        k = next(k for k, n in sounding if not n.get("lyric"))
        ctx.err(f"{p}.notes[{k}].lyric", 'this note has no lyric while others have one - give every sounding note a '
                                         'syllable (use "-" to hold the previous vowel).')
        return None
    if sounding and (sounding[0][1].get("lyric") or "").strip() in ("-", "~"):
        ctx.err(f"{p}.notes[{sounding[0][0]}].lyric", '"-" holds the previous vowel but there is none yet - start the '
                                                      'voice with a syllable.')
        return None
    return nv


def _check_note(ctx: _Ctx, p: str, n: Any) -> dict:
    if not isinstance(n, Mapping):
        ctx.err(p, f'an item must be an object like {{"pitch": "C4", "beats": 1, "lyric": "la"}} - replace '
                   f'{json.dumps(n)}.')
        return {}
    for k in n:
        if k not in NOTE_KEYS:
            _unknown(ctx, f"{p}.{k}", str(k), NOTE_KEYS)
    kinds = [k for k in ("pitch", "rest", "say") if k in n]
    if len(kinds) != 1:
        what = " and ".join(f'"{k}"' for k in kinds) if kinds else "none of pitch, rest, say"
        ctx.err(p, f'an item is exactly one of a note ("pitch"), a rest ("rest") or spoken words ("say"); this one has '
                   f'{what} - split it into separate items.')
        return {}
    kind = kinds[0]
    out: dict[str, Any] = {}
    if kind == "pitch":
        try:
            midi = parse_pitch(n["pitch"])
        except ValueError:
            ctx.err(f"{p}.pitch", f"{json.dumps(n['pitch'])} is not a note name - use C, C#, Db, D, … B plus an octave "
                                  f"(C4 = middle C) or a MIDI number {MIDI_MIN}-{MIDI_MAX}.")
            midi = None
        if midi is not None:
            if not MIDI_MIN <= midi <= MIDI_MAX:
                ctx.err(f"{p}.pitch", f"{json.dumps(n['pitch'])} (MIDI {midi:g}) is outside the singable range - use "
                                      f"MIDI {MIDI_MIN}-{MIDI_MAX} (E1 to E7).")
            else:
                out["pitch"] = midi
        _beats(ctx, p, n, out)
        if "lyric" in n:
            if not isinstance(n["lyric"], str):
                ctx.err(f"{p}.lyric", f'{json.dumps(n["lyric"])} is not a syllable - write the syllable as text, e.g. '
                                      f'"la".')
            elif n["lyric"].strip():
                out["lyric"] = n["lyric"]
        if "dyn" in n:
            if n["dyn"] not in DYN_LEVELS:
                ctx.err(f"{p}.dyn", f'{json.dumps(n["dyn"])} is not a dynamic - use one of pp, p, mp, mf, f, ff.')
            else:
                out["dyn"] = n["dyn"]
        for k in ("rate", "hz"):
            if k in n:
                ctx.err(f"{p}.{k}", f'"{k}" belongs to spoken items ("say") - remove it from this note.')
    elif kind == "rest":
        if _number(ctx, f"{p}.rest", n["rest"], 0, 64, "beats", low_open=True):
            out["rest"] = n["rest"]
        for k in ("lyric", "dyn", "beats", "rate", "hz"):
            if k in n:
                ctx.err(f"{p}.{k}", f'a rest has no "{k}" (its length is "rest") - remove "{k}".')
    else:
        s = n["say"]
        if not isinstance(s, str) or not s.strip():
            ctx.err(f"{p}.say", 'the spoken text is empty - write the words to say, e.g. "and now, all together!".')
        else:
            out["say"] = s
        _beats(ctx, p, n, out)
        if "rate" in n and _number(ctx, f"{p}.rate", n["rate"], 0.5, 1.8):
            out["rate"] = n["rate"]
        if "hz" in n and _number(ctx, f"{p}.hz", n["hz"], 50, 600, "Hz"):
            out["hz"] = n["hz"]
        for k in ("lyric", "dyn"):
            if k in n:
                ctx.err(f"{p}.{k}", f'spoken words have no "{k}" - remove "{k}" from this item.')
    return out


def _beats(ctx: _Ctx, p: str, n: Mapping, out: dict) -> None:
    if "beats" not in n:
        ctx.err(f"{p}.beats", 'the length is missing - add "beats", e.g. 1 for a quarter note at the score tempo.')
    elif _number(ctx, f"{p}.beats", n["beats"], 0, 64, "beats", low_open=True):
        out["beats"] = n["beats"]


def _check_music(ctx: _Ctx, norm: dict, g2p: Any, max_seconds: float, max_voice_seconds: float) -> ScoreReport:
    from .compile import compile_normalized, voice_syllables
    from .estimate import estimate_kernels
    from .mixer import estimate_memory
    from .planning import planner_api

    def fail() -> ScoreReport:
        return ScoreReport(False, ctx.issues, [], 0.0, 0.0, 0.0, 0.0, 0.0, {}, norm)

    api = planner_api()
    bpm = float(norm["bpm"])
    for i, v in enumerate(norm["voices"]):
        p = f"voices[{i}]"
        sounding = [n for n in v["notes"] if "pitch" in n]
        syl, _text = voice_syllables(v, api, g2p)
        if len(syl) != len(sounding):
            listing = "|".join(s.text for s in syl)
            ctx.err(p, f"{len(syl)} syllables for {len(sounding)} sounding notes ({listing}) - give every sounding note "
                       f'one syllable; use "-" to hold a vowel over the next note.')
            continue
        for k, n in enumerate(v["notes"]):
            if "say" in n and not g2p.to_phrases(api.prepare_text(n["say"])):
                ctx.err(f"{p}.notes[{k}].say", f'nothing speakable in {json.dumps(n["say"])} - write English words '
                                               f'(chimera reads English text).')
    if ctx.n_errors:
        return fail()
    plans = compile_normalized(norm, g2p=g2p)
    for i, vp in enumerate(plans):
        for k, end in vp.say_ends.items():
            slot_end = vp.say_slots[k]
            if end > slot_end + 1e-9:
                n = norm["voices"][i]["notes"][k]
                need = float(n["beats"]) + (end - slot_end) * bpm / 60.0
                ctx.err(f"voices[{i}].notes[{k}]", f"spoken text needs {need:.1f} beats at {bpm:g} bpm but the slot has "
                                                   f'{float(n["beats"]):g} - raise "beats" or shorten the text.')
    if ctx.n_errors:
        return fail()
    lead_in = float(norm["lead_in"])
    earliest = min((g.t0 - g.ramp_in for vp in plans for g in vp.job.gestures), default=lead_in)
    if earliest < LEAD_IN_MARGIN:
        raise_s = math.ceil((LEAD_IN_MARGIN - earliest) * 100.0 - 1e-9) / 100.0
        new = round(lead_in + raise_s, 9)
        ctx.warn("lead_in", f"lead_in raised from {lead_in:g} s to {new:g} s: the first onset needs "
                            f"{lead_in - earliest:.3f} s before beat 0 - set \"lead_in\" to {new:g} to keep it.")
        lead_in = new
        norm["lead_in"] = new
        plans = compile_normalized(norm, g2p=g2p)
    n_mix = max(vp.mix.offset + vp.job.n_out for vp in plans)
    seconds = n_mix / 48000.0
    voice_seconds = sum(vp.job.n_out for vp in plans) / 48000.0
    if seconds > max_seconds:
        ctx.err("score", f"the mix is {seconds:.0f} s long, the limit is {max_seconds:.0f} s - shorten the score or "
                         f"split it into parts.")
    if voice_seconds > max_voice_seconds:
        ctx.err("score", f"{len(plans)} voices render {voice_seconds:.0f} voice-seconds in total, the limit is "
                         f"{max_voice_seconds:.0f} - use fewer or shorter voices.")
    if ctx.n_errors:
        return fail()
    summaries = []
    for i, (v, vp) in enumerate(zip(norm["voices"], plans)):
        mix = BodyMix.parse(v["body"])
        lo, hi = natural_range(mix)
        sounding = [n for n in v["notes"] if "pitch" in n]
        for k, n in enumerate(v["notes"]):
            if "pitch" in n and not lo - 1e-9 <= n["pitch"] <= hi + 1e-9:
                side = "above" if n["pitch"] > hi else "below"
                bound = midi_name(hi if side == "above" else lo)
                ctx.warn(f"voices[{i}].notes[{k}].pitch",
                         f"{midi_name(n['pitch'])} ({_hz(n['pitch']):.0f} Hz) is {side} the natural range of body "
                         f"\"{body_text(mix)}\" ({'up to' if side == 'above' else 'down to'} {bound}) - it will sound "
                         f"strained; choose another body or octave if that is not wanted.")
        items_beats = sum(float(n.get("beats", n.get("rest", 0))) for n in v["notes"])
        midis = [n["pitch"] for n in sounding]
        summaries.append(VoiceSummary(
            name=v["name"], body=body_text(mix), items=len(v["notes"]), sounding=len(sounding),
            spoken=sum(1 for n in v["notes"] if "say" in n), first_beat=v["start"],
            end_beat=v["start"] + items_beats, seconds=vp.job.n_out / 48000.0,
            lowest=midi_name(min(midis)) if midis else "", highest=midi_name(max(midis)) if midis else "",
            syllables=[s.text for s in vp.syllables]))
    stereo = norm["stereo"] if norm["stereo"] != "auto" else any(v["pan"] != 0 for v in norm["voices"])
    from .estimate import music_slots
    workers = _default_workers()
    mem, disk = estimate_memory(n_mix, bool(stereo), [vp.job.n_out for vp in plans], music_slots(workers))
    est = estimate_kernels([vp.job.n_out / 48000.0 for vp in plans], workers=workers)
    return ScoreReport(True, ctx.issues, summaries, seconds, voice_seconds, lead_in, mem, disk, est, norm)


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _default_workers() -> int:
    from ..scheduler import default_workers
    return default_workers(os.cpu_count())
