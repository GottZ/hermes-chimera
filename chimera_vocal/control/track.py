"""ControlModel -> TrackChunk: everything HumanVoice::control() does except acoustics
(voice/human_voice.rs:426-542): gesture blend, articulator step, source channels, pruning, clock, geometry and
slew limit, contact constraint with re-application, noise sources with filter inheritance, breath gate, lung
pressure target, pitch glide and the glottis parameter row of each block.

The control step runs at the start of every block of ctrl_every = 64*m internal samples (dt = 64/fs_out); the clock
grows by repeated ``+= dt`` (human_voice.rs:467), never ``k*dt``. Commands (set_source, schedule, ...) apply between
``advance`` calls, i.e. at block boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, NamedTuple, Sequence

import math

import numpy as np

from ..core.body import HumanBody
from ..core.glottis_params import (DEFAULT_GLOTTIS, G_ABDUCTION, G_CLOSURE, G_EFFORT, G_F0, G_VIBRATO_DEPTH,
                                   apply_source_params)
from ..core.jsmath import pow as js_pow
from ..core.keys import CONTROL_INDEX
from ..core.nbmath import clamp, js_max, js_min, js_trunc_i32
from ..core.physics import CM_H2O
from ..core.types import DEFAULT_SOURCE, Gesture, SourceParams
from .airway import Airway, block_inherit
from .articulators import ArticulatorBank
from .geometry import GeometryImpl
from .score import GestureScore

if TYPE_CHECKING:  # the acoustics' topology index; annotations only, no import at run time
    from ..dsp.network import TractIndex


class TrackChunk(NamedTuple):
    """One row per control block, in order; arrays C-contiguous. Row b holds the state after the complete control
    step b, which the acoustic kernel applies at the start of block b. Plain container: numba caches the argument
    types of compiled functions by module path, so package classes never reach a compiled function."""
    n_blocks: int
    tube_area: np.ndarray      # float64[n_blocks, n_ph + n_or]  slew-limited section areas (pharynx then oral)
    tube_min: np.ndarray       # float64[n_blocks, n_ph + n_or]  slew-limited sub-section minima
    tube_len: np.ndarray       # float64[n_blocks, 2]            (ph_len, or_len)
    velum_area: np.ndarray     # float64[n_blocks]               velum port target (cm^2)
    noise_n: np.ndarray        # int64[n_blocks]                 0..3 sources (final list of the block)
    noise_idx: np.ndarray      # int64[n_blocks, 3, 3]           (src, fwd, bwd) branch indices, -1 padded
    noise_par: np.ndarray      # float64[n_blocks, 3, 3]         (gain, lowpass_hz, highpass_hz), 0 padded
    noise_inherit: np.ndarray  # int64[n_blocks, 3]              index into the PREVIOUS block's final list, -1
    glot_par: np.ndarray       # float64[n_blocks, 17]           GlottisParams (G_* order) at the block start
    p_lung: np.ndarray         # float64[n_blocks]               lung pressure target of the block (dyn/cm^2)


class ControlModel:
    """Control side of HumanVoice.

    Public: ``source`` (SourceParams), ``clock`` (s), ``f0`` (glide state). Internal, for tests: ``dt``,
    ``gate``, ``score`` (GestureScore), ``artic`` (ArticulatorBank), ``airway`` (Airway).
    ``set_source`` takes SourceParams field names (``vibrato_depth``) or the template's serde names
    (``vibratoDepth``).
    """

    source: SourceParams
    clock: float
    f0: float
    dt: float
    gate: float

    def __init__(self, body: HumanBody, index: "TractIndex", *, geometry: GeometryImpl,
                 fs_out: float = 48000.0, oversample: int = 2) -> None:
        if (float(fs_out), int(oversample)) not in ((48000.0, 2), (96000.0, 1)):
            raise ValueError(f"unsupported (fs_out, oversample) = ({fs_out}, {oversample})")
        fs = float(fs_out) * int(oversample)
        if float(index.fs) != fs:
            raise ValueError(f"TractIndex built for {index.fs} Hz, voice runs at {fs} Hz")
        self.body = body
        self.ctrl_every = 64 * int(oversample)
        self.dt = self.ctrl_every / fs
        self.airway = Airway(body, index, geometry)
        self.artic = ArticulatorBank()
        self.score = GestureScore()
        self.source = DEFAULT_SOURCE
        self.gate = 0.0
        self.gate_target = 0.0
        self.score_breath_hold = 0.0
        self.p_lung = 0.0
        self.f0 = DEFAULT_SOURCE.f0
        self.clock = 0.0
        self.score_f0 = None
        self.max_penetration = 0.1
        self.glot = np.array(DEFAULT_GLOTTIS, dtype=np.float64)
        self.base_abduction = 0.0
        self.base_closure = 0.0
        self.airway.apply(self.artic.x.copy(), 0.0)
        self._apply_source()
        self._prev = [tuple(s) for s in self.airway.noise]
        self._initial = self._rows(1)
        self._write_row(self._initial, 0, [-1] * len(self.airway.noise))

    # ---- rows
    def _rows(self, n: int) -> TrackChunk:
        nt = self.airway.n_ph + self.airway.n_or
        return TrackChunk(n, np.zeros((n, nt)), np.zeros((n, nt)), np.zeros((n, 2)), np.zeros(n),
                          np.zeros(n, dtype=np.int64), np.full((n, 3, 3), -1, dtype=np.int64), np.zeros((n, 3, 3)),
                          np.full((n, 3), -1, dtype=np.int64), np.zeros((n, 17)), np.zeros(n))

    def _write_row(self, ch: TrackChunk, r: int, inherit) -> None:
        aw = self.airway
        n_ph = aw.n_ph
        ch.tube_area[r, :n_ph] = aw.ph_areas
        ch.tube_area[r, n_ph:] = aw.or_areas
        ch.tube_min[r, :n_ph] = aw.ph_min
        ch.tube_min[r, n_ph:] = aw.or_min
        ch.tube_len[r, 0] = aw.ph_len
        ch.tube_len[r, 1] = aw.or_len
        ch.velum_area[r] = aw.velum_target
        ch.noise_n[r] = len(aw.noise)
        for j, s in enumerate(aw.noise):
            ch.noise_idx[r, j] = (s.src, s.fwd, s.bwd)
            ch.noise_par[r, j] = (s.gain, s.lowpass_hz, s.highpass_hz)
            ch.noise_inherit[r, j] = inherit[j]
        ch.glot_par[r] = self.glot
        ch.p_lung[r] = self.p_lung

    def initial(self) -> TrackChunk:
        """One row: the state after construction, apply(x, 0) without slew (human_voice.rs:254-256); noise_inherit
        all -1."""
        return TrackChunk(1, *[a.copy() for a in self._initial[1:]])

    def advance(self, n_blocks: int) -> TrackChunk:
        """``n_blocks`` consecutive control steps (human_voice.rs:426-509); block 0 of the first call inherits
        from the list of ``initial``."""
        ch = self._rows(int(n_blocks))
        for r in range(int(n_blocks)):
            self._control()
            inherit = block_inherit(self._prev, self.airway.calls)
            self._write_row(ch, r, inherit)
            self._prev = [tuple(s) for s in self.airway.noise]
        return ch

    # ---- commands
    def set_source(self, patch: Mapping[str, float | bool]) -> None:
        changes = {}
        for k, v in patch.items():
            name = _SOURCE_NAMES.get(k, k)
            changes[name] = v
        self.source = self.source.patched(**changes)
        self._apply_source()

    def set_gate(self, level: float) -> None:
        self.gate_target = clamp(float(level), 0.0, 1.5)

    def set_articulation(self, patch: Mapping[str, float], tip_free: bool | None = None) -> None:
        if tip_free is not None:
            self.artic.tip_free = bool(tip_free)
        self.artic.set_target(patch)

    def schedule(self, gestures: Sequence[Gesture]) -> None:
        self.score.add_all(gestures)

    def release(self, tag: float) -> None:
        self.score.release(tag, self.clock)

    def cancel_score(self) -> None:
        self.score.clear()

    def settle_pitch(self) -> None:
        """Glide state f0 := source.f0 (human_voice.rs:225, :499-501), so a voice that starts mid-phrase has no
        pitch onset glide from the 110 Hz default."""
        self.f0 = self.source.f0

    def set_clock_block(self, k: int) -> None:
        """clock := 0.0 plus dt added k times, the exact clock of a voice that ran k blocks (a phrase rendered on
        its own lines up sample for sample with the continuous voice)."""
        c = 0.0
        for _ in range(int(k)):
            c += self.dt
        self.clock = c

    # ---- internals
    def _apply_source(self) -> None:
        vals, mask, base_ab = apply_source_params(self.source, self.body.scale)
        self.glot[mask] = vals[mask]
        self.base_abduction = base_ab
        self.base_closure = 0.0

    def _control(self) -> None:
        dt = self.dt
        t = self.clock
        g = self.glot
        breath_act = 0.0
        breath_val = 0.0
        f0_target = self.source.f0
        if self.score.size() > 0:
            b = self.score.blend_all(t)
            self.artic.step(dt, b, 0.05)
            k2 = self.body.scale * self.body.scale
            ab_w = float(b.weight[_ABD])
            ab_act = js_min(1.0, ab_w / 2.0) if ab_w > 0.0 else 0.0
            g[G_ABDUCTION] = self.base_abduction + float(b.value[_ABD]) * ab_act * k2
            cl_w = float(b.weight[_CLO])
            g[G_CLOSURE] = js_max(self.base_closure, float(b.value[_CLO]) * js_min(1.0, cl_w / 2.0) if cl_w > 0.0 else 0.0)
            br_w = float(b.weight[_BRE])
            if br_w > 0.0:
                breath_act = js_min(1.0, br_w)
                breath_val = float(b.value[_BRE])
            vb_w = float(b.weight[_VIB])
            if vb_w > 0.0:
                g[G_VIBRATO_DEPTH] = (self.source.vibrato_depth * 0.05 + float(b.value[_VIB]) * vb_w) / (0.05 + vb_w)
            else:
                g[G_VIBRATO_DEPTH] = self.source.vibrato_depth
            pf_w = float(b.weight[_F0])
            if pf_w > 0.0:
                f0_target = (self.source.f0 * 0.02 + float(b.value[_F0]) * pf_w) / (0.02 + pf_w)
                self.score_f0 = f0_target
            else:
                self.score_f0 = None
            if js_trunc_i32(self.clock * 10.0) != js_trunc_i32((self.clock - dt) * 10.0):
                self.score.prune(t - 0.2)
        else:
            self.artic.step(dt, None, 1.0)
            g[G_ABDUCTION] = self.base_abduction
            g[G_CLOSURE] = self.base_closure
            self.score_f0 = None
        self.clock += dt
        self.airway.apply(self.artic.x.copy(), dt)
        if self._enforce_contacts():
            self.airway.reapply(self.artic.x.copy())
        score_breath = breath_val * breath_act
        gate_goal = score_breath if score_breath < 0.0 else js_max(self.gate_target, score_breath)
        self.score_breath_hold = 0.15 if breath_act > 0.0 else js_max(0.0, self.score_breath_hold - dt)
        if gate_goal > self.gate:
            tau = 0.025
        elif self.score_breath_hold > 0.0:
            tau = 0.02
        else:
            tau = 0.07
        self.gate += (gate_goal - self.gate) * (1.0 - math.exp(-dt / tau))
        self.p_lung = self.source.pressure * CM_H2O * self.gate
        g[G_EFFORT] = clamp((abs(self.source.pressure * self.gate) - 3.0) / 10.0, 0.0, 1.0)
        gl = js_max(0.002, 0.03 if self.score_f0 is not None else self.source.glide)
        self.f0 *= js_pow(f0_target / self.f0, 1.0 - math.exp(-dt / gl))
        g[G_F0] = self.f0

    def _enforce_contacts(self) -> bool:
        """human_voice.rs:515-542."""
        k = self.body.scale
        lim = -self.max_penetration * k
        ct, ctn, cb, cbn = self.airway.contacts()
        changed = False
        x, v = self.artic.x, self.artic.v
        for d, n, kx, ky in ((ct, ctn, _TTX, _TTY), (cb, cbn, _TBX, _TBY)):
            if not (d < lim):
                continue
            changed = True
            depth = lim - d
            sx = (1.0 / 0.86 if x[kx] > 0.5 else 1.0) / k
            sy = 1.0 / k
            x[kx] = x[kx] + n[0] * depth * sx
            x[ky] = x[ky] + n[1] * depth * sy
            vn = v[kx] * n[0] / sx + v[ky] * n[1] / sy
            if vn < 0.0:
                v[kx] = v[kx] - vn * n[0] * sx
                v[ky] = v[ky] - vn * n[1] * sy
        return changed


_ABD, _CLO, _BRE, _F0, _VIB = (CONTROL_INDEX[k] for k in ("abduction", "closure", "breath", "f0", "vibrato"))
_TTX, _TTY, _TBX, _TBY = (CONTROL_INDEX[k] for k in ("ttX", "ttY", "tbX", "tbY"))
_SOURCE_NAMES = {"vibratoDepth": "vibrato_depth", "vibratoRate": "vibrato_rate"}
