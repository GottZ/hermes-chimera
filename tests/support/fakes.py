"""C4 test fakes (BUILD-ORDER §1): FakeVoice, StubG2P, StubPlanner, a worker voice factory, process helpers.

* ``FakeVoice`` - deterministic synth with the HumanVoice API (INTERFACES §6.5). Signal model (global output
  sample g = 64·start_block + local index, t = g/fs):
    - f0(t): target of the active pitch gesture (``targets["f0"]``, active from t0 − ramp_in to t1 + ramp_out,
      the latest t0 wins), else the source f0; without ``settle_pitch()`` a fresh voice starts at 110 Hz and its
      log-f0 approaches the target as exp(-t/0.06 s) (the private glide state of human_voice.rs:49,225,499-500),
      with it there is no glide;
    - carrier: harmonics H1..H4 = 0.15/1.0/0.1/0.3 (H2 16.5 dB above H1 like a sung /ɑ/, E1 pitch-range report:
      naked YIN reports the octave above, sung_f0 must not), phase
      accumulated sample by sample (chunk-size independent, bit-exact);
    - level: gate(t) = max(set_gate level, breath gestures with linear ramps); voicing = gate × vowel window (vowel
      gestures = targets with "jaw", hard start at t0) × 0.6·pressure/9, plus seeded noise 0.02·gate; stop bursts
      (labels t d k p b g) at round(t1·fs): 48 samples of seeded noise;
    - kernel "numba" scales the output by (1 + 2⁻¹⁰) - mixing kernels inside one request is audible in the bits;
    - hold limiter = hard clamp to ±0.75 (human_voice.rs:374-392 peak), float32 output.
* ``StubG2P`` / ``StubPlanner`` - record their arguments; StubPlanner implements the C1 calls the music and speech
  paths make (``planner_api()`` seam, chimera_vocal/music/planning.py) with deterministic, documented timing.
* ``fake_voice_factory`` - installed in render workers through ``CHIMERA_WORKER_VOICE_FACTORY`` (effective only
  with ``CHIMERA_TESTING=1``); directives in the
  job id (``|``-separated) or in env ``CHIMERA_FAKE`` steer crashes, errors, slowness, stray output and probes.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core import jsmath
from chimera_vocal.core.types import (Gesture, Intonation, Phrase, PhraseInput, Plan, PlannedSegment, PlanOptions,
                                      Seeds, Segment, SongOptions, SourceParams, SungNote, Syllable)

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORY_SPEC = "support.fakes:fake_voice_factory"
HARMONICS = (0.15, 1.0, 0.1, 0.3)   # H1 16.5 dB below H2 (sung /ɑ/: 11-18 dB, E1 pitch-range)
LIMIT = 0.75
STOPS = frozenset({"t", "d", "k", "p", "b", "g"})
BURST_LEN = 48
GLIDE_S = 0.06
NUMBA_SCALE = 1.0 + 2.0 ** -10
_SOURCE_CAMEL = {"vibratoDepth": "vibrato_depth", "vibratoRate": "vibrato_rate"}


# --- deterministic noise ------------------------------------------------------------------------------------

def hash_noise(g: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 of (g, seed) -> float64 in [-1, 1); a pure function of the global sample index."""
    with np.errstate(over="ignore"):
        x = g.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64((seed * 0xBF58476D1CE4E5B9) & (2**64 - 1))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (2.0 / 9007199254740992.0) - 1.0


def _ramp_shape(t: np.ndarray, g: Gesture) -> np.ndarray:
    """0 before t0 − ramp_in, linear to 1 at t0, 1 until t1, linear to 0 at t1 + ramp_out."""
    a, b = g.t0 - g.ramp_in, g.t1 + g.ramp_out
    y = np.zeros_like(t)
    if g.ramp_in > 0:
        m = (t >= a) & (t < g.t0)
        y[m] = (t[m] - a) / g.ramp_in
    y[(t >= g.t0) & (t <= g.t1)] = 1.0
    if g.ramp_out > 0:
        m = (t > g.t1) & (t < b)
        y[m] = 1.0 - (t[m] - g.t1) / g.ramp_out
    return y


# --- FakeVoice ----------------------------------------------------------------------------------------------

class FakeVoice:
    """Deterministic stand-in for dsp.voice.HumanVoice (same constructor and methods)."""

    def __init__(self, fs_out: float = 48000.0, body: "BodyMix | Any" = BodyMix(), oversample: int = 2, *,
                 kernel: Any = None, geometry: Any = None, seeds: Seeds = Seeds(), start_block: int = 0) -> None:
        if (float(fs_out), int(oversample)) not in ((48000.0, 2), (96000.0, 1)):
            raise ValueError(f"unsupported rate {fs_out}/{oversample}")
        self.fs = float(fs_out)
        self.body = body
        self.kernel_name = kernel if isinstance(kernel, str) else getattr(kernel, "name", None) or "numpy"
        self.seeds = seeds
        self.start_block = int(start_block)
        self.g0 = 64 * self.start_block
        self.pos = self.g0
        self.carry = 0.0
        self.source = SourceParams()
        self.gate = 0.0
        self.artic: dict = {}
        self.gestures: list[Gesture] = []
        self.settled = False
        self.released: list[tuple[float, float]] = []
        self.calls: list[tuple[str, Any]] = []
        self.on_render: Callable[[int], None] | None = None

    # HumanVoice API ------------------------------------------------------------------------------------------
    def settle_pitch(self) -> None:
        self.calls.append(("settle_pitch", None))
        self.settled = True

    def set_source(self, patch: Mapping[str, Any]) -> None:
        self.calls.append(("set_source", dict(patch)))
        changes = {_SOURCE_CAMEL.get(k, k): v for k, v in patch.items()}
        self.source = self.source.patched(**changes)

    def set_gate(self, level: float) -> None:
        self.calls.append(("set_gate", level))
        self.gate = min(max(float(level), 0.0), 1.5)

    def set_articulation(self, patch: Mapping[str, float], tip_free: bool | None = None) -> None:
        self.calls.append(("set_articulation", (dict(patch), tip_free)))
        self.artic.update(patch)

    def schedule(self, gestures: Sequence[Gesture]) -> None:
        self.calls.append(("schedule", len(gestures)))
        self.gestures.extend(gestures)

    def release(self, tag: float) -> None:
        self.calls.append(("release", tag))
        self.released.append((tag, self.clock))

    def cancel_score(self) -> None:
        self.calls.append(("cancel_score", None))
        self.gestures.clear()

    @property
    def clock(self) -> float:
        return self.pos / self.fs

    @property
    def state(self) -> None:
        return None

    @property
    def index(self) -> None:
        return None

    def render(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        n = int(n)
        if self.on_render is not None:
            self.on_render(n)
        y = self._signal(self.pos, n)
        self.pos += n
        if out is not None:
            out[:n] = y
            return out[:n]
        return y

    # signal model ---------------------------------------------------------------------------------------------
    def _f0(self, g: np.ndarray, t: np.ndarray) -> np.ndarray:
        f = np.full(g.shape, float(self.source.f0))
        pitch = sorted((x for x in self.gestures if "f0" in x.targets), key=lambda x: x.t0)
        for x in pitch:
            m = (t >= x.t0 - x.ramp_in) & (t <= x.t1 + x.ramp_out)
            if m.any():
                f[m] = float(x.targets["f0"])
        if not self.settled:
            # private glide state of a fresh voice starts at 110 Hz and follows the target with the source glide
            # time (0.06 s, human_voice.rs:49,225,499-500): log-f decays as exp(-t/τ)
            local = (g - self.g0) / self.fs
            m = local < 12 * GLIDE_S
            if m.any():
                f[m] = f[m] * (110.0 / f[m]) ** np.exp(-local[m] / GLIDE_S)
        return f

    def _vowel(self, t: np.ndarray) -> np.ndarray:
        """Voicing window: 1 inside [t0, t1] of vowel gestures (targets with "jaw", no ramp-in, linear ramp-out);
        1 everywhere when no vowel gesture is scheduled (plain test jobs)."""
        vowels = [x for x in self.gestures if "jaw" in x.targets]
        if not vowels:
            return np.ones_like(t)
        v = np.zeros_like(t)
        for x in vowels:
            if x.t0 <= t[-1] and x.t1 + x.ramp_out >= t[0]:
                y = np.where((t >= x.t0) & (t <= x.t1), 1.0, 0.0)
                if x.ramp_out > 0:
                    m = (t > x.t1) & (t < x.t1 + x.ramp_out)
                    y[m] = 1.0 - (t[m] - x.t1) / x.ramp_out
                v = np.maximum(v, y)
        return v

    def _gate(self, t: np.ndarray) -> np.ndarray:
        level = np.full(t.shape, self.gate)
        for x in self.gestures:
            if "breath" in x.targets and x.t0 - x.ramp_in <= t[-1] and x.t1 + x.ramp_out >= t[0]:
                level = np.maximum(level, _ramp_shape(t, x) * float(x.targets["breath"]))
        return level

    def _signal(self, start: int, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, np.float32)
        g = np.arange(start, start + n, dtype=np.int64)
        t = g / self.fs
        f0 = self._f0(g, t)
        inc = (2.0 * math.pi / self.fs) * f0
        acc = np.add.accumulate(np.concatenate(([self.carry], inc)))
        phase, self.carry = acc[:n], float(acc[n])
        gate = self._gate(t)
        y = 0.02 * gate * hash_noise(g, self.seeds.net)
        if self.source.voiced:
            car = np.zeros(n)
            for k, h in enumerate(HARMONICS, start=1):
                car += h * np.sin(k * phase)
            y += gate * self._vowel(t) * (0.6 * float(self.source.pressure) / 9.0) * car / sum(HARMONICS)
        for x in self.gestures:
            if x.label in STOPS:
                gb = int(math.floor(x.t1 * self.fs + 0.5))
                lo, hi = max(gb, start), min(gb + BURST_LEN, start + n)
                if lo < hi:
                    k = np.arange(lo, hi, dtype=np.int64)
                    y[lo - start:hi - start] += 0.5 * np.exp(-(k - gb) / 8.0) * hash_noise(k, self.seeds.glottis)
        if self.kernel_name == "numba":
            y = y * NUMBA_SCALE
        np.clip(y, -LIMIT, LIMIT, out=y)
        return y.astype(np.float32)


def reference_render(job: Any, kernel: str = "numpy", chunk: int = 4096) -> np.ndarray:
    """Independent reading of the render_job contract (INTERFACES §8.3) with FakeVoice: source before sample 0,
    base posture, gestures, settle_pitch, then commands at their output index, rendering in chunks."""
    v = FakeVoice(job.fs_out, job.body, job.oversample, kernel=kernel, seeds=job.seeds, start_block=job.start_block)
    v.set_source({k: getattr(job.source, k) for k in SourceParams.__dataclass_fields__})
    if job.artic:
        v.set_articulation(job.artic)
    v.schedule(list(job.gestures))
    if job.settle_f0:
        v.settle_pitch()
    out = np.zeros(job.n_out, np.float32)
    pos = 0
    cmds = sorted(job.commands, key=lambda c: c.at)
    i = 0
    while pos < job.n_out:
        while i < len(cmds) and cmds[i].at <= pos:
            apply_command(v, cmds[i])
            i += 1
        stop = job.n_out if i >= len(cmds) else min(job.n_out, cmds[i].at)
        stop = min(stop, pos + chunk)
        out[pos:stop] = v.render(stop - pos)
        pos = stop
    return out


def apply_command(v: Any, cmd: Any) -> None:
    d = cmd.data
    if cmd.op == "source":
        v.set_source(d)
    elif cmd.op == "gate":
        v.set_gate(d["level"])
    elif cmd.op == "artic":
        v.set_articulation(d["patch"], d.get("tip_free"))
    elif cmd.op == "schedule":
        v.schedule([g if isinstance(g, Gesture) else Gesture.from_json(g) for g in d["gestures"]])
    elif cmd.op == "release":
        v.release(d["tag"])
    elif cmd.op == "cancel":
        v.cancel_score()
    else:
        raise ValueError(cmd.op)


# --- worker voice factory and directives --------------------------------------------------------------------

def directives(job_id: str) -> dict[str, str]:
    """``"name|slow=0.5|crash-once=/p"`` + env CHIMERA_FAKE -> {"slow": "0.5", "crash-once": "/p", ...}."""
    out: dict[str, str] = {}
    for src in (os.environ.get("CHIMERA_FAKE", ""), job_id or ""):
        for part in src.split("|"):
            if not part:
                continue
            k, _, v = part.partition("=")
            out[k.strip()] = v
    return out


def die_now() -> None:
    """Abrupt worker death without a reply and without a core file."""
    sys.stdout.flush()
    if hasattr(signal, "SIGKILL"):
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(3)


def _log(dirpath: str, line: str) -> None:
    p = Path(dirpath)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{os.getpid()}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fake_voice_factory(job: Any, kernel: Any, geometry: Any) -> FakeVoice:
    """Voice factory for ``compile.set_voice_factory`` / ``CHIMERA_WORKER_VOICE_FACTORY``."""
    d = directives(job.id)
    kname = kernel if isinstance(kernel, str) else getattr(kernel, "name", None) or "numpy"
    if "probe" in d:
        import site  # noqa: F401  (only to report whether it is loaded)
        info = {"pid": os.getpid(), "sys_path": list(sys.path), "cwd": os.getcwd(),
                "flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site,
                          "ignore_environment": sys.flags.ignore_environment},
                "executable": sys.executable, "numba_loaded": "numba" in sys.modules,
                "sitecustomize_loaded": "sitecustomize" in sys.modules,
                "env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "kernel": kname}
        Path(d["probe"]).write_text(json.dumps(info), encoding="utf-8")
    if "log" in d:
        _log(d["log"], f"factory {os.getpid()} {job.id} {kname} {time.time():.6f}")
    v = FakeVoice(job.fs_out, job.body, job.oversample, kernel=kname, seeds=job.seeds, start_block=job.start_block)
    slow = float(d["slow"]) if d.get("slow") else 0.0
    state = {"first": True}

    def hook(n: int) -> None:
        if "log" in d:
            _log(d["log"], f"render {os.getpid()} {job.id} {kname} {time.time():.6f} {n}")
        if state["first"]:
            state["first"] = False
            if "crash" in d or ("crash-numba" in d and kname == "numba"):
                die_now()
            if "crash-once" in d:
                marker = Path(d["crash-once"])
                if not marker.exists():
                    marker.write_text(str(os.getpid()), encoding="utf-8")
                    die_now()
        if "error" in d:
            raise RuntimeError("fake failure: " + (d["error"] or "error directive"))
        if "stray" in d:
            print("stray print from the voice")
            sys.stdout.flush()
            try:
                os.write(1, b"CHW2\xff\xff\xff\x7fjunk on fd 1\n")
            except OSError:
                pass
        if slow:
            time.sleep(slow)

    v.on_render = hook
    return v


def worker_env(monkeypatch: Any, **directive_env: str) -> None:
    """Make every render worker started from now on use FakeVoice (and optional CHIMERA_FAKE directives)."""
    monkeypatch.setenv("CHIMERA_WORKER_VOICE_FACTORY", FACTORY_SPEC)
    monkeypatch.setenv("CHIMERA_TESTING", "1")
    if directive_env.get("fake"):
        monkeypatch.setenv("CHIMERA_FAKE", directive_env["fake"])


def read_logs(dirpath: str | os.PathLike) -> list[list[str]]:
    rows: list[list[str]] = []
    p = Path(dirpath)
    if not p.is_dir():
        return rows
    for f in sorted(p.iterdir()):
        for line in f.read_text(encoding="utf-8").splitlines():
            rows.append(line.split(" "))
    return rows


# --- stub text front end ------------------------------------------------------------------------------------

VOWEL_LETTERS = frozenset("aeiouy")
_PAUSE = {",": 0.18, ";": 0.18, ":": 0.18, ".": 0.35, "!": 0.35, "?": 0.35, "\n": 0.35}


IPA_VOWELS = frozenset("aeiouyɪəɛæɑɔʊʌɝɚ")


class StubG2P:
    """Stub G2P. ``dictionary`` maps a word to its syllable pronunciations (one string of IPA characters per
    syllable, e.g. "twinkle" -> ["twɪŋ", "kəl"]); every character is one Segment. Words without an entry are
    spelled: one Segment per letter, one syllable per vowel-letter group. ``calls`` records (method, args)."""

    def __init__(self, dictionary: Mapping[str, Sequence[str]] | None = None) -> None:
        self.dictionary = {k.lower(): list(v) for k, v in (dictionary or {}).items()}
        self.calls: list[tuple[str, Any]] = []
        self.lock = threading.Lock()

    def _record(self, name: str, args: Any) -> None:
        with self.lock:
            self.calls.append((name, args))

    def word(self, w: str, reduce: bool) -> list[Segment]:
        self._record("word", (w, reduce))
        w = w.lower()
        chars = "".join(self.dictionary[w]) if w in self.dictionary else "".join(c for c in w if c.isalpha())
        out, first = [], True
        for c in chars:
            vowel = c in IPA_VOWELS
            out.append(Segment(c, stress=1 if vowel and first else 0))
            first = first and not vowel
        return out

    def letter_to_sound(self, w: str) -> list[Segment]:
        self._record("letter_to_sound", w)
        return [Segment(c, stress=0) for c in w.lower() if c.isalpha()]

    def syllables_of(self, w: str) -> list[tuple[str, str]]:
        """[(spelling, pronunciation)] per syllable: the dictionary's syllables with the spelling split to match
        (split_spelling), else one syllable per vowel-letter group spelled as written."""
        lw = w.lower()
        if lw in self.dictionary:
            prons = self.dictionary[lw]
            return list(zip(split_spelling(w, len(prons)), prons))
        return [(p, p.lower()) for p in _vowel_group_split(w)]

    def to_phrases(self, text: str) -> list[Phrase]:
        self._record("to_phrases", text)
        phrases: list[Phrase] = []
        words: list[str] = []
        for tok in _tokens(text):
            if tok in _PAUSE:
                if words:
                    phrases.append(self._phrase(words, tok))
                    words = []
            else:
                words.append(tok)
        if words:
            phrases.append(self._phrase(words, "."))
        return phrases

    def _phrase(self, words: list[str], punct: str) -> Phrase:
        segs: list[Segment] = []
        for w in words:
            ws = self.word(w, True)
            if ws:
                ws[-1] = Segment(ws[-1].ipa, ws[-1].dur, ws[-1].f0, ws[-1].stress, True)
            segs.extend(ws)
        inton = Intonation.QUESTION if punct == "?" else Intonation.STATEMENT
        return Phrase(segs, inton, _PAUSE[punct], [w.lower() for w in words])


def split_spelling(word: str, n: int) -> list[str]:
    """syllables.rs split_spelling, simplified: vowel-letter groups merged/padded to n pieces."""
    if n <= 1:
        return [word]
    pieces = _vowel_group_split(word)
    if len(pieces) < n:
        return [word] + ["~"] * (n - 1)
    while len(pieces) > n:
        last = pieces.pop()
        pieces[-1] += last
    return pieces


def _tokens(text: str) -> list[str]:
    out, cur = [], []
    for c in text:
        if c.isalpha() or c == "'":
            cur.append(c)
            continue
        if cur:
            out.append("".join(cur))
            cur = []
        if c in _PAUSE:
            out.append(c)
    if cur:
        out.append("".join(cur))
    return out


def _vowel_group_split(w: str) -> list[str]:
    """Split a word before each vowel group after the first ("hello" -> ["hel", "lo"]); no vowel -> []."""
    groups = []
    lw = w.lower()
    i = 0
    while i < len(w):
        if lw[i] in VOWEL_LETTERS:
            s = i
            while i < len(w) and lw[i] in VOWEL_LETTERS:
                i += 1
            groups.append((s, i))
        else:
            i += 1
    if not groups:
        return []
    cuts = [0]
    for (a0, a1), (b0, _b1) in zip(groups, groups[1:]):
        gap = b0 - a1
        cuts.append(a1 + (0 if gap <= 1 else gap // 2))
    cuts.append(len(w))
    return [w[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


# --- stub planner -------------------------------------------------------------------------------------------

LYRIC_PUNCT = frozenset(".,;:!?\n")
# earliest gesture start before the first beat, as measured on the template's plan_song (native, HEAD 092b16ce12b9):
SONG_LEAD = {"ah": 0.080, "strength": 0.258, "splash": 0.266, "spring": 0.266}


def _lyric_tokens(text: str) -> list[str]:
    """syllables.rs lyric_tokens: words with inner hyphens, lone - or ~ between spaces, phrase punctuation."""
    s = text
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isascii() and (c.isalpha() or c == "'"):
            j = i + 1
            while j < len(s) and ((s[j].isascii() and (s[j].isalpha() or s[j] == "'")) or s[j] == "-"):
                j += 1
            out.append(s[i:j])
            i = j
            continue
        mel = c in "~-" and (i == 0 or s[i - 1].isspace()) and (i + 1 == len(s) or s[i + 1].isspace())
        if mel or c in LYRIC_PUNCT:
            out.append(c)
        i += 1
    return out


@dataclass
class StubPlanner:
    """Implements the PlannerAPI calls with fixed, documented timing and records every call.

    plan_phrases: phrase k starts at t (t0 = opts.t0, then end + pause), lasts ``phrase_dur(segments)``
      (table ``durations`` keyed by the joined segment ipa, default 0.1 + 0.07·len); gestures: breath
      [t − 0.01 … end] (the template's first gesture starts ≈ t0 − 0.01, E1-A §3.5), one gesture per segment (label = ipa, consonants are stops when listed in STOPS),
      one pitch gesture at opts.f0 (or 110). Labels of all gestures of phrase k carry the suffix "#k".
    plan_song: times[k] = t0 + Σ beats·60/bpm; per sounding note a vowel gesture [times[k], times[k+1] − 0.02],
      a pitch gesture at midi_to_hz(midi) and onset stops ending at the beat; earliest start of a phrase =
      first beat − lead, lead from SONG_LEAD by syllable text else 0.06 + 0.05·len(onset); end = last beat end + 0.1.
    """

    g2p: StubG2P = field(default_factory=StubG2P)
    durations: dict[str, float] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _record(self, name: str, args: Any) -> None:
        with self.lock:
            self.calls.append((name, args))

    def calls_of(self, name: str) -> list[Any]:
        return [a for n, a in self.calls if n == name]

    def api(self):
        from chimera_vocal.music.planning import PlannerAPI
        return PlannerAPI(plan_phrases=self.plan_phrases, plan_song=self.plan_song,
                          lyrics_to_syllables=self.lyrics_to_syllables, prepare_text=self.prepare_text,
                          default_g2p=lambda: self.g2p)

    # speech ---------------------------------------------------------------------------------------------------
    def phrase_dur(self, segments: Sequence[Segment]) -> float:
        key = "".join(s.ipa for s in segments)
        return self.durations.get(key, 0.1 + 0.07 * len(segments))

    def plan_phrases(self, phrases: Sequence[PhraseInput], opts: PlanOptions) -> Plan:
        self._record("plan_phrases", (list(phrases), opts))
        gestures: list[Gesture] = []
        segments: list[PlannedSegment] = []
        t = opts.t0
        end = opts.t0
        f0 = opts.f0 if opts.f0 is not None else 110.0
        for k, p in enumerate(phrases):
            dur = self.phrase_dur(p.segments)
            end = t + dur
            gestures.append(Gesture(t + 0.03, end, 0.04, 0.06, 1.0, {"breath": 1.0}, label="breath"))
            gestures.append(Gesture(t, end, 0.03, 0.03, 6.0, {"f0": f0}, label="pitch"))
            step = dur / max(1, len(p.segments))
            for i, s in enumerate(p.segments):
                a = t + i * step
                gestures.append(Gesture(a, a + step, 0.02, 0.02, 1.0, {"jaw": 0.5}, label=s.ipa))
                segments.append(PlannedSegment(s.ipa, "vowel" if s.ipa in VOWEL_LETTERS else "consonant",
                                               a, a + step, s.stress))
            t = end + (p.pause if p.pause is not None else 0.25)
        return Plan(gestures, segments, end)

    # song -----------------------------------------------------------------------------------------------------
    def song_lead(self, syl: Syllable) -> float:
        key = syl.text.strip().lower()
        if key in SONG_LEAD:
            return SONG_LEAD[key]
        return 0.06 + 0.05 * len(syl.onset)

    def plan_song(self, notes: Sequence[SungNote], opts: SongOptions) -> Plan:
        self._record("plan_song", (list(notes), opts))
        beat = 60.0 / opts.bpm
        times = [opts.t0]
        for n in notes:
            times.append(times[-1] + n.beats * beat)
        gestures: list[Gesture] = []
        segments: list[PlannedSegment] = []
        lead = self.song_lead(notes[0].syl) if notes else 0.0
        if notes:
            gestures.append(Gesture(times[0] - lead + 0.04, times[-1], 0.04, 0.05, 1.0, {"breath": 1.0},
                                    label="breath"))
        for k, n in enumerate(notes):
            a, b = times[k], times[k + 1]
            hz = jsmath.midi_to_hz(n.midi)   # module attribute: the K-4.1 red proof patches it
            gestures.append(Gesture(a, b - 0.02, 0.06, 0.06, 6.0, {"f0": hz}, label="pitch"))
            if not n.syl.melisma:
                gestures.append(Gesture(a, b - 0.02, 0.07, 0.05, 1.0, {"jaw": 0.6}, label=n.syl.nucleus.ipa))
                for j, c in enumerate(n.syl.onset):
                    rel = a - (len(n.syl.onset) - j) * 0.03
                    gestures.append(Gesture(rel, rel + 0.03, 0.03, 0.018, 40.0, {"ttY": 9.68}, label=c.ipa))
                segments.append(PlannedSegment(n.syl.nucleus.ipa, "vowel", a, b, 1))
        end = times[-1] + 0.1 if notes else opts.t0
        return Plan(gestures, segments, end)

    # lyrics ---------------------------------------------------------------------------------------------------
    def lyrics_to_syllables(self, text: str, g2p: Any) -> list[Syllable]:
        """syllables.rs:180-220 over the stub dictionary (typed hyphens split, otherwise the dictionary)."""
        self._record("lyrics_to_syllables", text)
        g = g2p if isinstance(g2p, StubG2P) else self.g2p
        out: list[Syllable] = []
        for tok in _lyric_tokens(text.replace("’", "'")):
            if len(tok) == 1 and tok in LYRIC_PUNCT:
                if out:
                    last = out[-1]
                    out[-1] = Syllable(last.text, last.onset, last.nucleus, last.coda, last.melisma, last.word_end,
                                       True)
                continue
            if tok in ("-", "~"):
                if out:
                    prev = out[-1]
                    out.append(Syllable("~", [], prev.nucleus, [], True, prev.word_end, False))
                continue
            parts = [p for p in tok.split("-") if p]
            syls: list[Syllable] = []
            for part in parts:
                sy = g.syllables_of(part)
                for i, (spell, pron) in enumerate(sy):
                    text_i = spell if len(parts) == 1 else (part if i == 0 else "~")
                    syls.append(_syllable(text_i, pron))
            if syls:
                last = syls[-1]
                syls[-1] = Syllable(last.text, last.onset, last.nucleus, last.coda, None, True, False)
            out.extend(syls)
        return out

    def prepare_text(self, text: str) -> str:
        self._record("prepare_text", text)
        return text


def _syllable(text: str, pron: str) -> Syllable:
    """Onset = characters before the first vowel character, nucleus = that vowel, coda = the rest."""
    v = next((i for i, c in enumerate(pron) if c in IPA_VOWELS), None)
    if v is None:
        return Syllable(text, [Segment(c, stress=0) for c in pron], Segment("ə", stress=0), [], None, False, False)
    onset = [Segment(c, stress=0) for c in pron[:v]]
    coda = [Segment(c, stress=0) for c in pron[v + 1:]]
    return Syllable(text, onset, Segment(pron[v], stress=1), coda, None, False, False)


def install_stub_planner(planner: StubPlanner | None = None) -> StubPlanner:
    """Install a StubPlanner through the planner seam (caller restores with set_planner_api(None))."""
    from chimera_vocal.music.planning import set_planner_api
    p = planner or StubPlanner()
    set_planner_api(p.api())
    return p


# --- process helpers ----------------------------------------------------------------------------------------

def pid_state(pid: int) -> str:
    """"gone", "zombie" or "alive" (Linux: /proc; elsewhere os.kill(pid, 0) / OpenProcess)."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return "gone"
        state = stat.rsplit(")", 1)[1].split()[0]
        return "zombie" if state in ("Z", "X") else "alive"
    if os.name == "nt":  # pragma: no cover - Windows CI
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not h:
            return "gone"
        try:
            r = ctypes.windll.kernel32.WaitForSingleObject(h, 0)
            return "gone" if r == 0 else "alive"
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:  # pragma: no cover - macOS
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "alive"
    return "alive"


def wait_until(pred: Callable[[], bool], timeout: float, step: float = 0.02) -> float | None:
    """Seconds until ``pred()`` became true, or None after ``timeout``."""
    t0 = time.monotonic()
    while True:
        if pred():
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(step)


def real_engine_available() -> bool:
    """True once C1–C3 are merged (S3): real planner, control model and HumanVoice importable."""
    import importlib.util
    need = ("chimera_vocal.text.english", "chimera_vocal.plan.sing", "chimera_vocal.plan.planner",
            "chimera_vocal.control.track", "chimera_vocal.dsp.voice", "chimera_vocal.dsp.backend")
    try:
        return all(importlib.util.find_spec(m) is not None for m in need)
    except (ImportError, ValueError):
        return False


# --- pytest fixtures (imported by tests/{music,engine,measure}/conftest.py) ---------------------------------

import pytest  # noqa: E402

STUB_DICTIONARY = {   # word -> syllable pronunciations (IPA characters); spelled words fall back to letters
    "twinkle": ["twɪŋ", "kəl"], "tiger": ["taɪ", "gɚ"], "hello": ["hə", "loʊ"], "every": ["ɛv", "ɚ", "i"],
    "little": ["lɪ", "təl"], "merrily": ["mɛ", "ɹə", "li"], "gently": ["dʒɛnt", "li"], "amen": ["ɑ", "mɛn"],
    "fire": ["faɪ", "ɚ"], "star": ["stɑɹ"], "row": ["ɹoʊ"], "boat": ["boʊt"], "strength": ["stɹɛŋθ"],
}


@pytest.fixture
def stub_planner():
    """StubPlanner (with STUB_DICTIONARY) installed through the planner seam for the duration of the test."""
    from chimera_vocal.music.planning import set_planner_api
    p = StubPlanner(StubG2P(STUB_DICTIONARY))
    prev = set_planner_api(p.api())
    try:
        yield p
    finally:
        set_planner_api(prev)


@pytest.fixture
def fake_voice():
    """In-process renders (render_job, in-process fallback) use FakeVoice."""
    from chimera_vocal.music.compile import set_voice_factory
    prev = set_voice_factory(fake_voice_factory)
    try:
        yield fake_voice_factory
    finally:
        set_voice_factory(prev)


@pytest.fixture
def fake_workers(monkeypatch):
    """Render workers started from now on use FakeVoice; numba counts as not installed (no warm-up op)."""
    import chimera_vocal.scheduler as sched
    worker_env(monkeypatch)
    monkeypatch.setattr(sched, "_numba_installed", lambda: False)
    monkeypatch.setattr(sched, "_marker_ok", lambda base: None)
    return monkeypatch


@pytest.fixture
def make_scheduler(tmp_path):
    """Factory for RenderSchedulers rooted at the repository; all are shut down after the test."""
    from chimera_vocal.scheduler import RenderScheduler
    made = []

    def make(workers: int = 2, **kw):
        kw.setdefault("cache_base", str(tmp_path / "numba-cache"))
        kw.setdefault("root", str(REPO_ROOT))
        s = RenderScheduler(workers=workers, **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.shutdown()


@pytest.fixture
def make_engine(tmp_path):
    """Factory for Engines (EngineConfig keyword arguments); all are shut down after the test."""
    from chimera_vocal import Engine, EngineConfig
    made = []

    def make(**kw):
        kw.setdefault("cache_dir", str(tmp_path / "numba-cache"))
        e = Engine(EngineConfig(**kw))
        made.append(e)
        return e

    yield make
    for e in made:
        e.shutdown()


# --- direct worker access (protocol tests, INTERFACES §9.3) --------------------------------------------------

def fake_job(job_id: str, seconds: float = 0.5, *, f0: float = 220.0, start_block: int = 0, n_out: int | None = None,
             seeds: Seeds = Seeds()):
    """A RenderJob for FakeVoice: one breath gesture over the job, settled pitch."""
    from chimera_vocal.music.compile import RenderJob
    t_start = 64 * start_block / 48000.0
    n = int(round(seconds * 48000)) if n_out is None else n_out
    breath = Gesture(t_start + 0.01, t_start + n / 48000.0, 0.01, 0.01, 1.0, {"breath": 1.0}, label="breath")
    return RenderJob(id=job_id, body=BodyMix(), seeds=seeds, source=SourceParams(f0=f0), gestures=[breath],
                     n_out=n, start_block=start_block, settle_f0=True)


def handshake(root: Path = REPO_ROOT, cache_base: str | None = None) -> dict:
    """The first stdin line the scheduler sends (INTERFACES §9.3), computed from this process."""
    import site
    cwd = os.getcwd()
    path = [p for p in sys.path if p not in ("", cwd)]
    site_dirs = [d for d in site.getsitepackages() if d in sys.path]
    return {"proto": 2, "sys_path": path, "site_dirs": site_dirs, "root": str(root), "cache_base": cache_base}


def spawn_worker(stderr_path: str | os.PathLike, *, root: Path = REPO_ROOT, cache_base: str | None = None,
                 env: Mapping[str, str] | None = None, flags: Sequence[str] = ("-I", "-S", "-u")):
    """Start ``python -I -S -u -c BOOTSTRAP`` like the scheduler and send the handshake; returns the Popen."""
    import subprocess
    from chimera_vocal import worker
    e = dict(os.environ)
    e.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CHIMERA_TESTING": "1"})
    e.update(env or {})
    err = open(stderr_path, "wb")
    p = subprocess.Popen([sys.executable, *flags, "-c", worker.BOOTSTRAP], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=err, cwd=str(root), env=e, close_fds=True)
    err.close()
    p.stdin.write((json.dumps(handshake(root, cache_base)) + "\n").encode("utf-8"))
    p.stdin.flush()
    return p
