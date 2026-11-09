"""Control models for C3's tests: replay of the template's recorded control track (V3) and a synthetic steady
posture. They stand in for C2's ControlModel (INTERFACES §6.3) through ``chimera_vocal.dsp.voice.CONTROL_FACTORY``.

Track files: in-repo ``tests/vectors/track-<scenario>.json`` (integer fields of every block, float fields of
``float_blocks``) or the lab's full ``$CHIMERA_LAB_REFS/tracks/track-<scenario>.full.json`` (every field of every
block); schema in tests/vectors/README.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

import support
from chimera_vocal.core import glottis_params as gp
from chimera_vocal.core import jsmath
from chimera_vocal.core.physics import CM_H2O
from chimera_vocal.core.types import DEFAULT_SOURCE, SourceParams

SCENARIOS = ("sustain-aa-male", "hello-male", "stop-male", "song-female")
LAB_AUDIO = {s: f"audio/track_{s}.f32" for s in SCENARIOS}
NOISE_PAD = 3


class TrackChunk(NamedTuple):
    """Same fields and dtypes as C2's TrackChunk (INTERFACES §6.3); the kernel wrapper only reads attributes."""

    n_blocks: int
    tube_area: np.ndarray      # float64[n, n_ph + n_or]
    tube_min: np.ndarray       # float64[n, n_ph + n_or]
    tube_len: np.ndarray       # float64[n, 2]
    velum_area: np.ndarray     # float64[n]
    noise_n: np.ndarray        # int64[n]
    noise_idx: np.ndarray      # int64[n, 3, 3]
    noise_par: np.ndarray      # float64[n, 3, 3]
    noise_inherit: np.ndarray  # int64[n, 3]
    glot_par: np.ndarray       # float64[n, 17]
    p_lung: np.ndarray         # float64[n]


def empty_chunk(n_tract: int) -> TrackChunk:
    return make_chunk(0, n_tract)


def make_chunk(n: int, n_tract: int) -> TrackChunk:
    return TrackChunk(
        n, np.zeros((n, n_tract)), np.zeros((n, n_tract)), np.zeros((n, 2)), np.zeros(n),
        np.zeros(n, dtype=np.int64), np.full((n, NOISE_PAD, 3), -1, dtype=np.int64), np.zeros((n, NOISE_PAD, 3)),
        np.full((n, NOISE_PAD), -1, dtype=np.int64), np.zeros((n, 17)), np.zeros(n))


def concat_chunks(chunks: list[TrackChunk]) -> TrackChunk:
    return TrackChunk(sum(c.n_blocks for c in chunks), *[np.ascontiguousarray(np.concatenate([getattr(c, f) for c in chunks]))
                                                        for f in TrackChunk._fields[1:]])


def p1_inherit(noise_idx: np.ndarray, noise_n: np.ndarray, initial_idx: np.ndarray, initial_n: int) -> np.ndarray:
    """REVIEW-1 P1-01 emulation ("final list against the previous block"): source j of block b inherits from the
    first source of block b-1's FINAL list with the same (src, fwd); -1 if none. The template is different whenever a
    contact re-application put an intermediate list in between (that is what noise_inherit records)."""
    n_blocks = noise_idx.shape[0]
    out = np.full((n_blocks, NOISE_PAD), -1, dtype=np.int64)
    prev_idx, prev_n = initial_idx, initial_n
    for b in range(n_blocks):
        for j in range(int(noise_n[b])):
            for k in range(int(prev_n)):
                if prev_idx[k, 0] == noise_idx[b, j, 0] and prev_idx[k, 1] == noise_idx[b, j, 1]:
                    out[b, j] = k
                    break
        prev_idx, prev_n = noise_idx[b], noise_n[b]
    return out


class Track:
    """One V3 scenario: integer fields of every block, float fields where stored, the initial row."""

    def __init__(self, name: str, *, full: bool = False):
        self.name = name
        self.full = full
        if full:
            lab = support.lab_refs()
            assert lab is not None, "CHIMERA_LAB_REFS not set"
            d = support.load_json(lab / "tracks" / f"track-{name}.full.json")
        else:
            d = support.vector(f"track-{name}")
        self.raw = d
        self.scenario = d["scenario"]
        self.n_ph, self.n_or = d["n_ph"], d["n_or"]
        self.n_tract = self.n_ph + self.n_or
        self.n_blocks = d["n_blocks"]
        self.ctrl_every = d["ctrl_every"]
        self.n_samples = self.scenario["n_samples"]
        nb, nt = self.n_blocks, self.n_tract
        self.noise_n = np.zeros(nb, dtype=np.int64)
        self.noise_idx = np.full((nb, NOISE_PAD, 3), -1, dtype=np.int64)
        self.noise_inherit = np.full((nb, NOISE_PAD), -1, dtype=np.int64)
        self.has_float = np.zeros(nb, dtype=bool)
        self.tube_area = np.full((nb, nt), np.nan)
        self.tube_min = np.full((nb, nt), np.nan)
        self.tube_len = np.full((nb, 2), np.nan)
        self.velum_area = np.full(nb, np.nan)
        self.noise_par = np.zeros((nb, NOISE_PAD, 3))
        self.glot_par = np.full((nb, 17), np.nan)
        self.p_lung = np.full(nb, np.nan)
        self.clock = np.full(nb, np.nan)
        if full:
            blk = d["blocks"]
            for b in range(nb):
                self._ints(b, blk["noise_idx"][b], blk["noise_inherit"][b])
                self._floats(b, {k: blk[k][b] for k in ("tube_area", "tube_min", "tube_len", "velum_area",
                                                        "noise_par", "glot_par", "p_lung", "clock")})
            self.noise_inherit_p1_export = np.full((nb, NOISE_PAD), -1, dtype=np.int64)
            for b in range(nb):
                v = blk["noise_inherit_p1"][b]
                self.noise_inherit_p1_export[b, :len(v)] = v
        else:
            runs = d["int_runs"]
            for r, (first, idx, inh) in enumerate(runs):
                last = runs[r + 1][0] if r + 1 < len(runs) else nb
                for b in range(first, last):
                    self._ints(b, idx, inh)
            fl = d["float"]
            for i, b in enumerate(d["float_blocks"]):
                self._floats(b, {k: fl[k][i] for k in fl})
            self.noise_inherit_p1_export = None
        ini = d["initial"]
        self.initial = make_chunk(1, nt)
        self.initial.noise_n[0] = len(ini["noise_idx"])
        for j, t in enumerate(ini["noise_idx"]):
            self.initial.noise_idx[0, j] = t
        for k in ("tube_area", "tube_min", "tube_len", "glot_par"):
            getattr(self.initial, k)[0] = ini[k]
        self.initial.velum_area[0] = ini["velum_area"]
        self.initial.p_lung[0] = ini["p_lung"]
        for j, t in enumerate(ini["noise_par"]):
            self.initial.noise_par[0, j] = t
        self.p1 = p1_inherit(self.noise_idx, self.noise_n, self.initial.noise_idx[0], int(self.initial.noise_n[0]))
    def _ints(self, b, idx, inh):
        self.noise_n[b] = len(idx)
        for j, t in enumerate(idx):
            self.noise_idx[b, j] = t
        for j, v in enumerate(inh):
            self.noise_inherit[b, j] = v

    def _floats(self, b, f):
        self.has_float[b] = True
        self.tube_area[b] = f["tube_area"]
        self.tube_min[b] = f["tube_min"]
        self.tube_len[b] = f["tube_len"]
        self.velum_area[b] = f["velum_area"]
        for j, t in enumerate(f["noise_par"]):
            self.noise_par[b, j] = t
        self.glot_par[b] = f["glot_par"]
        self.p_lung[b] = f["p_lung"]
        self.clock[b] = f["clock"]

    def float_head(self) -> int:
        """Number of leading blocks whose float fields are all present (row b = state after control step b)."""
        n = 0
        while n < self.n_blocks and self.has_float[n]:
            n += 1
        return n

    def chunk(self, b0: int, b1: int, *, inherit: str = "track") -> TrackChunk:
        """Rows b0..b1-1 as a TrackChunk; inherit "track" = recorded noise_inherit, "p1" = REVIEW-1 emulation."""
        assert 0 <= b0 <= b1 <= self.n_blocks, (b0, b1, self.n_blocks)
        assert self.has_float[b0:b1].all(), f"{self.name}: float fields missing in blocks {b0}..{b1 - 1}"
        inh = self.noise_inherit if inherit == "track" else self.p1
        sl = slice(b0, b1)
        return TrackChunk(b1 - b0, *[np.ascontiguousarray(a[sl]) for a in (
            self.tube_area, self.tube_min, self.tube_len, self.velum_area, self.noise_n, self.noise_idx,
            self.noise_par, inh, self.glot_par, self.p_lung)])

    def posture(self, b: int) -> TrackChunk:
        """Block b's recorded tract and noise fields as a one-row chunk (glot/p_lung as recorded; SteadyControl
        replaces them with its own breath and pitch)."""
        assert self.has_float[b], f"{self.name}: no float fields in block {b}"
        sl = slice(b, b + 1)
        return TrackChunk(1, *[np.ascontiguousarray(a[sl]) for a in (
            self.tube_area, self.tube_min, self.tube_len, self.velum_area, self.noise_n, self.noise_idx,
            self.noise_par, self.noise_inherit, self.glot_par, self.p_lung)])

    def source(self) -> SourceParams:
        """The performer source after the ops at sample 0 (set_source patches in order)."""
        s = DEFAULT_SOURCE
        for op in self.scenario["ops"]:
            if op["op"] == "set_source":
                s = SourceParams.from_json(op["patch"], base=s)
        return s

    def lab_audio(self) -> np.ndarray:
        lab = support.lab_refs()
        assert lab is not None
        return np.fromfile(lab / LAB_AUDIO[self.name], dtype="<f4")


_TRACKS: dict[tuple[str, bool], Track] = {}


def track(name: str, *, full: bool = False) -> Track:
    key = (name, full)
    if key not in _TRACKS:
        _TRACKS[key] = Track(name, full=full)
    return _TRACKS[key]


class RecordedControl:
    """ControlModel stand-in that replays a recorded track (performer commands only update `source`; their effect
    is already in the recorded rows). advance() raises if the recording (or its float part) runs out."""

    def __init__(self, trk: Track, *, inherit: str = "track", dt: float = 128.0 / 96000.0):
        self.track = trk
        self.inherit = inherit
        self.pos = 0
        self.dt = dt
        self.clock = 0.0
        self.source = trk.source()
        self.f0 = float(trk.initial.glot_par[0, gp.G_F0])
        self.calls: list[tuple] = []

    def initial(self) -> TrackChunk:
        return self.track.initial

    def advance(self, n_blocks: int) -> TrackChunk:
        c = self.track.chunk(self.pos, self.pos + n_blocks, inherit=self.inherit)
        self.pos += n_blocks
        for _ in range(n_blocks):
            self.clock += self.dt
        if n_blocks:
            self.f0 = float(c.glot_par[-1, gp.G_F0])
        return c

    def set_source(self, patch):
        self.calls.append(("set_source", dict(patch)))
        self.source = _patched(self.source, patch)

    def set_gate(self, level):
        self.calls.append(("set_gate", level))

    def set_articulation(self, patch, tip_free=None):
        self.calls.append(("set_articulation", dict(patch), tip_free))

    def schedule(self, gestures):
        self.calls.append(("schedule", len(gestures)))

    def release(self, tag):
        self.calls.append(("release", tag))

    def cancel_score(self):
        self.calls.append(("cancel_score",))

    def settle_pitch(self):
        self.calls.append(("settle_pitch",))

    def set_clock_block(self, k):
        self.calls.append(("set_clock_block", k))
        self.clock = 0.0
        for _ in range(k):
            self.clock += self.dt


def _patched(src: SourceParams, patch) -> SourceParams:
    camel = {j: p for p, j in (("vibrato_depth", "vibratoDepth"), ("vibrato_rate", "vibratoRate"))}
    return src.patched(**{camel.get(k, k): v for k, v in dict(patch).items()})


def recorded_factory(name: str, *, full: bool = False, inherit: str = "track"):
    """A CONTROL_FACTORY replacement that ignores its arguments and replays track `name`."""

    def factory(body, index, geometry, fs_out, oversample):
        return RecordedControl(track(name, full=full), inherit=inherit, dt=64.0 * oversample / (fs_out * oversample))

    return factory


class SteadyControl:
    """Synthetic ControlModel for a held posture: the tract rows of one recorded block (default: sustained /ɑ/ male,
    block 79) and the template's breath gate, lung pressure, effort and pitch glide (human_voice.rs:477-501), the
    voice clock by repeated += dt, settle_pitch and set_clock_block per INTERFACES §6.3."""

    def __init__(self, posture: TrackChunk, *, body_scale: float = 1.0, fs: float = 96000.0, ctrl_every: int = 128,
                 source: SourceParams = DEFAULT_SOURCE):
        assert posture.n_blocks == 1
        self.posture = posture
        self.k2 = body_scale * body_scale
        self.body_scale = body_scale
        self.dt = ctrl_every / fs
        self.source = source
        self.gate = 0.0
        self.gate_target = 0.0
        self.f0 = DEFAULT_SOURCE.f0
        self.clock = 0.0
        self.hold = 0.0
        self.calls: list[tuple] = []
        self._first = True

    def _row(self, out: TrackChunk, i: int) -> None:
        for f in ("tube_area", "tube_min", "tube_len", "velum_area", "noise_n", "noise_idx", "noise_par"):
            getattr(out, f)[i] = getattr(self.posture, f)[0]
        n = int(self.posture.noise_n[0])
        out.noise_inherit[i] = -1
        if not self._first:
            out.noise_inherit[i, :n] = np.arange(n)
        values, mask, base_abd = gp.apply_source_params(self.source, self.body_scale)
        row = np.array(gp.DEFAULT_GLOTTIS, dtype=np.float64)
        row[mask] = values[mask]
        row[gp.G_ABDUCTION] = base_abd
        row[gp.G_CLOSURE] = 0.0
        row[gp.G_VIBRATO_DEPTH] = self.source.vibrato_depth
        p_lung = self.source.pressure * CM_H2O * self.gate
        row[gp.G_EFFORT] = jsmath.clamp((abs(self.source.pressure * self.gate) - 3.0) / 10.0, 0.0, 1.0)
        row[gp.G_F0] = self.f0
        out.glot_par[i] = row
        out.p_lung[i] = p_lung

    def initial(self) -> TrackChunk:
        out = make_chunk(1, self.posture.tube_area.shape[1])
        self._row(out, 0)
        out.p_lung[0] = 0.0
        self._first = True
        return out

    def advance(self, n_blocks: int) -> TrackChunk:
        out = make_chunk(n_blocks, self.posture.tube_area.shape[1])
        dt = self.dt
        for i in range(n_blocks):
            self.clock += dt
            goal = jsmath.js_max(self.gate_target, 0.0)
            self.hold = jsmath.js_max(0.0, self.hold - dt)
            tau = 0.025 if goal > self.gate else (0.02 if self.hold > 0.0 else 0.07)
            self.gate += (goal - self.gate) * (1.0 - jsmath.exp(-dt / tau))
            g = jsmath.js_max(0.002, self.source.glide)
            self.f0 *= jsmath.pow(self.source.f0 / self.f0, 1.0 - jsmath.exp(-dt / g))
            self._row(out, i)
            self._first = False
        return out

    def set_source(self, patch):
        self.calls.append(("set_source", dict(patch)))
        self.source = _patched(self.source, patch)

    def set_gate(self, level):
        self.calls.append(("set_gate", level))
        self.gate_target = jsmath.clamp(level, 0.0, 1.5)

    def set_articulation(self, patch, tip_free=None):
        self.calls.append(("set_articulation", dict(patch), tip_free))

    def schedule(self, gestures):
        self.calls.append(("schedule", len(gestures)))

    def release(self, tag):
        self.calls.append(("release", tag))

    def cancel_score(self):
        self.calls.append(("cancel_score",))

    def settle_pitch(self):
        self.calls.append(("settle_pitch",))
        self.f0 = self.source.f0

    def set_clock_block(self, k):
        self.calls.append(("set_clock_block", k))
        self.clock = 0.0
        for _ in range(k):
            self.clock += self.dt


def steady_posture(name: str = "sustain-aa-male", block: int = 79) -> TrackChunk:
    return track(name).posture(block)


def steady_factory(posture: TrackChunk | None = None, **kw):
    def factory(body, index, geometry, fs_out, oversample):
        return SteadyControl(posture if posture is not None else steady_posture(), body_scale=body.scale,
                             ctrl_every=64 * oversample, **kw)

    return factory


def cents(f: np.ndarray, ref: float) -> np.ndarray:
    return 1200.0 * np.log2(np.asarray(f, dtype=np.float64) / ref)


def scenario_json(name: str) -> str:
    return json.dumps(track(name).scenario, sort_keys=True)


__all__ = ["TrackChunk", "Track", "track", "RecordedControl", "SteadyControl", "recorded_factory", "steady_factory",
           "steady_posture", "p1_inherit", "make_chunk", "concat_chunks", "empty_chunk", "cents", "SCENARIOS",
           "math", "Path"]
