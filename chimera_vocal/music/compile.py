"""Score -> one RenderJob per voice, faithful to ``chimera render notes`` (render.rs:481-575 of the template).

Also ``render_job``: one job through a HumanVoice (worker or in-process). The voice is built by
``voice_factory()`` - the real ``dsp.voice.HumanVoice`` unless a factory was installed with ``set_voice_factory``
(tests: FakeVoice; the worker installs one from ``CHIMERA_WORKER_VOICE_FACTORY``).
"""

from __future__ import annotations

import dataclasses
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

from ..core import jsmath
from ..core.body import BodyMix, default_f0
from ..core.jsmath import js_max, js_round
from ..core.types import Gesture, PhraseInput, PlanOptions, Seeds, SongOptions, SourceParams, SungNote, Syllable

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .schema import ScoreReport

VoiceFactory = Callable[["RenderJob", Any, Any], Any]    # (job, kernel, geometry) -> HumanVoice-like object

PRE_ROLL_S = 0.30        # silent pre-roll of a late voice: with a settled glide it renders like one started at 0
TAIL_S = 0.4             # render notes tail (render.rs:561)
SAY_LEAD_S = 0.06        # a spoken slot starts its plan 60 ms before the beat
DYN_PRESSURE = {"pp": 0.55, "p": 0.7, "mp": 0.85, "mf": 1.0, "f": 1.2, "ff": 1.4}
DYN_GAIN_DB = {"pp": -12.0, "p": -8.0, "mp": -4.0, "mf": 0.0, "f": 2.0, "ff": 4.0}
DYN_RAMP_S = 0.08
DYN_STEPS = 8
_SOURCE_FIELDS = tuple(f.name for f in dataclasses.fields(SourceParams))


@dataclass(frozen=True)
class Command:
    at: int                          # relative to the job's first output sample
    op: Literal["source", "gate", "artic", "schedule", "release", "cancel"]
    data: dict


@dataclass(frozen=True)
class RenderJob:
    id: str
    body: BodyMix
    seeds: Seeds
    fs_out: int = 48000
    oversample: int = 2
    source: SourceParams = SourceParams()
    artic: dict | None = None
    gestures: list[Gesture] = ()
    commands: list[Command] = ()
    n_out: int = 0
    start_block: int = 0
    settle_f0: bool = False

    def __post_init__(self) -> None:
        # lists in every instance (the defaults are tuples): equality after a JSON round trip
        object.__setattr__(self, "gestures", list(self.gestures))
        object.__setattr__(self, "commands", list(self.commands))

    def to_json(self) -> dict:
        cmds = []
        for c in self.commands:
            data = dict(c.data)
            if c.op == "schedule":
                data["gestures"] = [g.to_json() if isinstance(g, Gesture) else g for g in data.get("gestures", [])]
            cmds.append({"at": c.at, "op": c.op, "data": data})
        return {"id": self.id, "body": self.body.to_json(),
                "seeds": {"net": self.seeds.net, "glottis": self.seeds.glottis, "drift": self.seeds.drift},
                "fs_out": self.fs_out, "oversample": self.oversample, "source": self.source.to_json(),
                "artic": None if self.artic is None else dict(self.artic),
                "gestures": [g.to_json() for g in self.gestures], "commands": cmds, "n_out": self.n_out,
                "start_block": self.start_block, "settle_f0": self.settle_f0}

    @classmethod
    def from_json(cls, d: dict) -> "RenderJob":
        cmds = []
        for c in d.get("commands", []):
            data = dict(c["data"])
            if c["op"] == "schedule":
                data["gestures"] = [Gesture.from_json(g) for g in data.get("gestures", [])]
            cmds.append(Command(int(c["at"]), c["op"], data))
        b, s = d["body"], d["seeds"]
        return cls(id=d["id"], body=BodyMix(b["female"], b["child"]), seeds=Seeds(s["net"], s["glottis"], s["drift"]),
                   fs_out=int(d["fs_out"]), oversample=int(d["oversample"]), source=SourceParams.from_json(d["source"]),
                   artic=None if d.get("artic") is None else dict(d["artic"]),
                   gestures=[Gesture.from_json(g) for g in d.get("gestures", [])], commands=cmds,
                   n_out=int(d["n_out"]), start_block=int(d["start_block"]), settle_f0=bool(d["settle_f0"]))


@dataclass(frozen=True)
class VoiceMix:
    offset: int                      # = 64·job.start_block
    gain_db: float
    pan: float
    gain_env: "tuple[tuple[int, float], ...] | None"   # ((sample, dB), ...) relative to the job, piecewise linear
                                                        # in dB, constant outside; None = 0 dB


_factory_lock = threading.Lock()
_voice_factory: VoiceFactory | None = None


def set_voice_factory(factory: VoiceFactory | None) -> VoiceFactory | None:
    """Install a voice factory (None restores the real HumanVoice); returns the previous one."""
    global _voice_factory
    with _factory_lock:
        prev, _voice_factory = _voice_factory, factory
    return prev


def voice_factory() -> VoiceFactory:
    """The installed factory, else one that builds ``dsp.voice.HumanVoice`` from the job."""
    f = _voice_factory
    if f is not None:
        return f
    return _human_voice


def _human_voice(job: RenderJob, kernel: Any, geometry: Any) -> Any:
    from ..dsp.voice import HumanVoice
    kw: dict[str, Any] = {}
    if kernel is not None and not isinstance(kernel, str):
        kw["kernel"] = kernel
    if geometry is not None:
        kw["geometry"] = geometry
    return HumanVoice(float(job.fs_out), job.body, job.oversample, seeds=job.seeds, start_block=job.start_block, **kw)


# --- lyrics ---------------------------------------------------------------------------------------------------

def _letters(word: str) -> str:
    return "".join(c for c in word if c.isalpha() or c == "'")


def lyric_text(lyrics: Sequence[str], api: Any, g2p: Any) -> str:
    """Per-note lyrics -> template lyrics text: "twin-"+"kle" -> "twinkle" when the dictionary
    gives the word that many syllables, else the typed form "twin-kle"; "-"/"~" = melisma."""
    tokens: list[str] = []
    group: list[str] = []
    for raw in list(lyrics) + [None]:
        if raw is not None and raw.strip() not in ("-", "~") and raw.strip().endswith("-") and len(raw.strip()) > 1:
            group.append(raw.strip()[:-1])
            continue
        if group:
            last = raw.strip() if raw is not None else ""
            frags = group + [last]
            joined = "".join(frags)
            n = len(frags)
            word = _letters(joined)
            count = len(api.lyrics_to_syllables(word, g2p)) if word else 0
            tokens.append(joined if count == n else "-".join(frags))
            group = []
            continue
        if raw is None:
            break
        tokens.append(raw.strip())
    return " ".join(t for t in tokens if t)


def voice_syllables(voice: Mapping[str, Any], api: Any, g2p: Any) -> tuple[list[Syllable], str]:
    """(syllables, text) of a normalized voice: ``lyrics`` string, per-note ``lyric``s, or "ah" per sounding note."""
    sounding = [n for n in voice["notes"] if "pitch" in n]
    if voice.get("lyrics") is not None:
        text = voice["lyrics"]
    elif any(n.get("lyric") for n in sounding):
        text = lyric_text([n.get("lyric") or "" for n in sounding], api, g2p)
    else:
        text = " ".join(["ah"] * len(sounding))
    return list(api.lyrics_to_syllables(text, g2p)), text


# --- compile ---------------------------------------------------------------------------------------------------

@dataclass
class VoicePlan:
    """Everything compile learns about one voice (used by validate_score as well)."""
    job: RenderJob
    mix: VoiceMix
    syllables: list[Syllable]
    say_ends: dict[int, float] = field(default_factory=dict)     # item index -> plan end (s)
    say_slots: dict[int, float] = field(default_factory=dict)    # item index -> slot end t(b + beats) (s)
    end: float = 0.0


def _dyn_env(events: list[tuple[float, float, float]], offset: int, fs: int) -> tuple[tuple[int, float], ...]:
    """Mix gain breakpoints ``((sample, dB), ...)`` relative to the job: the first level from sample 0, then per level
    change a ramp of DYN_RAMP_S that ends on the beat (two points: start at the old level, beat at the new one)."""
    pts: list[tuple[int, float]] = [(0, float(events[0][2]))]
    for t_beat, prev_db, new_db in events[1:]:
        a = max(pts[-1][0], int(js_round(fs * (t_beat - DYN_RAMP_S))) - offset)
        b = max(a, int(js_round(fs * t_beat)) - offset)
        pts.append((a, float(prev_db)))
        pts.append((b, float(new_db)))
    return tuple(pts)


def compile_voice(i: int, voice: Mapping[str, Any], *, bpm: float, lead_in: float, api: Any, g2p: Any,
                  fs_out: int = 48000) -> VoicePlan:
    """render.rs:481-575 for one normalized voice, plus the extensions start/dyn/say/ring/seed."""
    body = BodyMix.parse(voice["body"]) if not isinstance(voice["body"], BodyMix) else voice["body"]
    syl, _ = voice_syllables(voice, api, g2p)
    beat = 60.0 / bpm
    breath = float(voice["breath"])
    sounding = [n for n in voice["notes"] if "pitch" in n]
    first_f0 = jsmath.midi_to_hz(sounding[0]["pitch"]) if sounding else None
    if first_f0 is None:
        says = [n for n in voice["notes"] if "say" in n]
        first_f0 = float(says[0]["hz"]) if says and says[0].get("hz") is not None else default_f0(body)
    level0 = next((n["dyn"] for n in sounding[:1] if n.get("dyn")), "mf")
    source = SourceParams().patched(f0=first_f0, voiced=True, pressure=breath * DYN_PRESSURE[level0],
                                    tension=float(voice["tension"]), breathiness=float(voice["breathiness"]),
                                    brightness=float(voice["brightness"]), roughness=float(voice["roughness"]),
                                    vibrato_rate=float(voice["vibrato_rate"]))
    gestures: list[Gesture] = []
    phrase: list[SungNote] = []
    state = {"end": 0.0, "start": 0.0}
    vp_say_ends: dict[int, float] = {}
    vp_say_slots: dict[int, float] = {}
    dyn_events: list[tuple[float, float, float, float, float]] = []   # (t_beat, p_prev, p_new, db_prev, db_new)
    level = level0

    def flush() -> None:
        if not phrase:
            return
        plan = api.plan_song(list(phrase), SongOptions(body, bpm, state["start"], float(voice["vibrato"])))
        gestures.extend(plan.gestures)
        state["end"] = js_max(state["end"], plan.end)
        phrase.clear()

    t = lead_in + float(voice["start"]) * beat
    k = 0
    for idx, n in enumerate(voice["notes"]):
        if "pitch" in n:
            if not phrase:
                state["start"] = t
            if n.get("dyn") and n["dyn"] != level:
                dyn_events.append((t, breath * DYN_PRESSURE[level], breath * DYN_PRESSURE[n["dyn"]],
                                   DYN_GAIN_DB[level], DYN_GAIN_DB[n["dyn"]]))
                level = n["dyn"]
            phrase.append(SungNote(float(n["pitch"]), float(n["beats"]), syl[k]))
            k += 1
            t += float(n["beats"]) * beat
        elif "rest" in n:
            flush()
            t += float(n["rest"]) * beat
        else:
            flush()
            text = api.prepare_text(n["say"])
            phrases = g2p.to_phrases(text)
            hz = float(n["hz"]) if n.get("hz") is not None else default_f0(body)
            rate = float(n["rate"]) if n.get("rate") is not None else None
            plan = api.plan_phrases([PhraseInput(p.segments, p.intonation, p.pause) for p in phrases],
                                    PlanOptions(body, t - SAY_LEAD_S, rate, hz))
            gestures.extend(plan.gestures)
            state["end"] = js_max(state["end"], plan.end)
            t += float(n["beats"]) * beat
            vp_say_ends[idx] = plan.end
            vp_say_slots[idx] = t
    flush()
    end = state["end"]
    t_first = min((g.t0 - g.ramp_in for g in gestures), default=lead_in)
    start_block = max(0, math.floor((t_first - PRE_ROLL_S) * fs_out / 64))
    offset = 64 * start_block
    n_out = int(js_round(fs_out * (end + TAIL_S))) - offset
    commands: list[Command] = []
    for t_beat, p_prev, p_new, _, _ in dyn_events:
        for j in range(DYN_STEPS):
            tc = t_beat - DYN_RAMP_S + (j + 1) * (DYN_RAMP_S / DYN_STEPS)
            at = max(0, int(js_round(fs_out * tc)) - offset)
            commands.append(Command(at, "source", {"pressure": p_prev + (p_new - p_prev) * (j + 1) / DYN_STEPS}))
    commands.sort(key=lambda c: c.at)
    env_events = [(0.0, DYN_GAIN_DB[level0], DYN_GAIN_DB[level0])] + [(e[0], e[3], e[4]) for e in dyn_events]
    gain_env = _dyn_env(env_events, offset, fs_out) if dyn_events or level0 != "mf" else None
    ring = float(voice["ring"])
    seed = voice.get("seed")
    job = RenderJob(id=f"v{i}", body=body, seeds=Seeds.for_voice(i, seed), fs_out=fs_out, oversample=2,
                    source=source, artic={"epilarynx": ring} if ring > 0 else None, gestures=gestures,
                    commands=commands, n_out=n_out, start_block=start_block, settle_f0=start_block > 0)
    mix = VoiceMix(offset, float(voice["volume"]), float(voice["pan"]), gain_env)
    return VoicePlan(job, mix, syl, vp_say_ends, vp_say_slots, end)


def compile_normalized(normalized: Mapping[str, Any], *, g2p: Any, fs_out: int = 48000,
                       lead_in: float | None = None) -> list[VoicePlan]:
    from .planning import planner_api
    api = planner_api()
    li = float(normalized["lead_in"] if lead_in is None else lead_in)
    return [compile_voice(i, v, bpm=float(normalized["bpm"]), lead_in=li, api=api, g2p=g2p, fs_out=fs_out)
            for i, v in enumerate(normalized["voices"])]


def compile_score(report: "ScoreReport", *, g2p: Any, fs_out: int = 48000) -> "tuple[list[RenderJob], list[VoiceMix], int]":
    """-> (one job per voice, mixing data, n_mix)."""
    if not report.ok:
        raise ValueError("compile_score needs a valid report (report.ok)")
    plans = compile_normalized(report.normalized, g2p=g2p, fs_out=fs_out)
    jobs = [p.job for p in plans]
    mixes = [p.mix for p in plans]
    n_mix = max((p.mix.offset + p.job.n_out for p in plans), default=0)
    return jobs, mixes, n_mix


# --- render ----------------------------------------------------------------------------------------------------

def apply_command(v: Any, cmd: Command) -> None:
    d = cmd.data
    if cmd.op == "source":
        v.set_source(d)
    elif cmd.op == "gate":
        v.set_gate(float(d["level"]))
    elif cmd.op == "artic":
        v.set_articulation(d["patch"], d.get("tip_free"))
    elif cmd.op == "schedule":
        v.schedule([g if isinstance(g, Gesture) else Gesture.from_json(g) for g in d["gestures"]])
    elif cmd.op == "release":
        v.release(float(d["tag"]))
    elif cmd.op == "cancel":
        v.cancel_score()
    else:
        raise ValueError(f"unknown command op {cmd.op!r}")


def render_job(job: RenderJob, kernel: Any, geometry: Any, *, out: "np.ndarray | None" = None,
               chunk: int = 48000, progress: Callable[[int], None] | None = None,
               cancel: Callable[[], bool] | None = None) -> "np.ndarray":
    """float32[n_out]; chunked (progress after each chunk, cancel checked between chunks); deterministic."""
    import numpy as np
    from .. import RenderCancelled
    v = voice_factory()(job, kernel, geometry)
    v.set_source({k: getattr(job.source, k) for k in _SOURCE_FIELDS})
    if job.artic:
        v.set_articulation(job.artic)
    if job.gestures:
        v.schedule(list(job.gestures))
    if job.settle_f0:
        v.settle_pitch()
    n = int(job.n_out)
    buf = np.zeros(n, np.float32) if out is None else out
    cmds = sorted(job.commands, key=lambda c: c.at)
    chunk = max(1, int(chunk))
    pos, i = 0, 0
    while pos < n:
        while i < len(cmds) and cmds[i].at <= pos:
            apply_command(v, cmds[i])
            i += 1
        if cancel is not None and cancel():
            raise RenderCancelled(f"job {job.id} cancelled")
        stop = min(n, (pos // chunk + 1) * chunk)
        if i < len(cmds):
            stop = min(stop, cmds[i].at)
        v.render(stop - pos, buf[pos:stop])
        pos = stop
        if progress is not None and (pos % chunk == 0 or pos == n):
            progress(pos)
    return buf
