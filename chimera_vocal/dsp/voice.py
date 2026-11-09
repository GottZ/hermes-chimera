"""HumanVoice: render loop ControlModel -> kernel -> output chain.

Ported from chimera-vocal-synth crates/chimera/src/voice/human_voice.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from typing import NamedTuple

from ..core.body import BodyMix, HumanBody
from ..core.glottis_params import apply_source_params
from ..core.types import Gesture, Seeds
from . import layout as L
from .kernel_numpy import KERNEL_NUMPY, KernelImpl
from .network import TractIndex, VoiceState, apply_initial, configure_output, human_net

_SOURCE_ALIASES = {"vibratoDepth": "vibrato_depth", "vibratoRate": "vibrato_rate"}


class _Rows(NamedTuple):
    n_blocks: int
    tube_area: np.ndarray
    tube_min: np.ndarray
    tube_len: np.ndarray
    velum_area: np.ndarray
    noise_n: np.ndarray
    noise_idx: np.ndarray
    noise_par: np.ndarray
    noise_inherit: np.ndarray
    glot_par: np.ndarray
    p_lung: np.ndarray


def _no_rows(n_tract: int) -> _Rows:
    return _Rows(0, np.zeros((0, n_tract)), np.zeros((0, n_tract)), np.zeros((0, 2)), np.zeros(0),
                 np.zeros(0, dtype=np.int64), np.zeros((0, 3, 3), dtype=np.int64), np.zeros((0, 3, 3)),
                 np.zeros((0, 3), dtype=np.int64), np.zeros((0, 17)), np.zeros(0))

if TYPE_CHECKING:
    from ..control.track import ControlModel, GeometryImpl, TrackChunk

ALLOWED_RATES: tuple[tuple[float, int], ...] = ((48000.0, 2), (96000.0, 1))
"""(fs_out, oversample) combinations HumanVoice accepts; internal rate 96 000 in both. Lower internal rates are
unstable for the smaller bodies; 96 kHz output exists only to reproduce the template's 96 kHz reference render."""

MAX_BLOCKS_PER_CHUNK = 750
"""Longest TrackChunk requested from the control model in one go (≈ 1 s at 48 kHz, ≈ 1 MB of track)."""


def default_control_factory(body: HumanBody, index: TractIndex, geometry: "GeometryImpl | None", fs_out: float,
                            oversample: int) -> "ControlModel":
    """The real control model: ControlModel(body, index, geometry=geometry or GEOMETRY_VEC, ...). Imported here,
    not at module level, so this module loads without the control package."""
    from ..control.track import ControlModel
    if geometry is None:
        from ..control.geometry import GEOMETRY_VEC
        geometry = GEOMETRY_VEC
    return ControlModel(body, index, geometry=geometry, fs_out=fs_out, oversample=oversample)


CONTROL_FACTORY: Callable[..., Any] = default_control_factory
"""Seam for tests: the callable HumanVoice uses to build its control model (same arguments as
default_control_factory). The acoustics tests replace it with a model that replays a recorded control track."""


def control_steps(st: VoiceState, n_internal: int) -> int:
    """Number of control steps that fall into the next n_internal internal samples (a step runs before every
    internal sample at which the control counter is <= 0)."""
    if n_internal <= 0:
        return 0
    first = max(int(st.si[L.SI_CTRL_COUNTER]), 0)
    if first >= n_internal:
        return 0
    return 1 + (n_internal - first - 1) // int(st.si[L.SI_CTRL_EVERY])


def run(k: KernelImpl, st: VoiceState, idx: TractIndex, trk: "TrackChunk", n_internal: int, raw: np.ndarray) -> None:
    """Unpack VoiceState/TractIndex/TrackChunk into arrays and call k.run_samples (rows of trk are consumed from row
    0); raise IndexError if the kernel set si[SI_ERROR] (track exhausted)."""
    c = np.ascontiguousarray
    st.si[L.SI_BLOCK_POS] = 0
    k.run_samples(st.nf, st.ni, st.nbr_start, st.nbr, st.bf, st.bi, st.rf, st.ri, st.noise_f, st.noise_i, st.glot,
                  st.glot_i, st.sc, st.si, idx.ph_nodes, idx.or_nodes,
                  c(trk.tube_area, dtype=np.float64), c(trk.tube_min, dtype=np.float64),
                  c(trk.tube_len, dtype=np.float64), c(trk.velum_area, dtype=np.float64),
                  c(trk.noise_n, dtype=np.int64), c(trk.noise_idx, dtype=np.int64), c(trk.noise_par, dtype=np.float64),
                  c(trk.noise_inherit, dtype=np.int64), c(trk.glot_par, dtype=np.float64),
                  c(trk.p_lung, dtype=np.float64), int(n_internal), raw)
    err = int(st.si[L.SI_ERROR])
    if err:
        st.si[L.SI_ERROR] = 0
        raise IndexError(f"control track exhausted after {int(st.si[L.SI_BLOCK_POS])} of {trk.n_blocks} rows")
    used = int(st.si[L.SI_BLOCK_POS])
    st.si[L.SI_BLOCK_POS] = 0           # a per-call cursor, not state: keep the state independent of call sizes
    if used != trk.n_blocks:
        raise ValueError(f"{trk.n_blocks} track rows passed, {used} used")


class HumanVoice:
    """= voice/human_voice.rs, CLI path (no Engine wrapper)."""

    def __init__(self, fs_out: float = 48000.0, body: "BodyMix | HumanBody" = BodyMix(), oversample: int = 2, *,
                 kernel: KernelImpl = KERNEL_NUMPY, geometry: "GeometryImpl | None" = None, seeds: Seeds = Seeds(),
                 start_block: int = 0) -> None:
        """(fs_out, oversample) in ALLOWED_RATES else ValueError. ``geometry=None`` means GEOMETRY_VEC (resolved by
        the control factory; the default is not imported at module level). start_block k > 0:
        ControlModel.set_clock_block(k); the caller places this voice's output at output sample 64·k (phrase windows and
        late score voices start on a control-block boundary so their gestures line up with the voice clock)."""
        key = (float(fs_out), oversample)
        if not isinstance(oversample, int) or key not in ALLOWED_RATES:
            raise ValueError(f"unsupported (fs_out, oversample) {key!r}; allowed: {ALLOWED_RATES}")
        if isinstance(body, HumanBody):
            hb = body
        elif isinstance(body, BodyMix):
            hb = body.body()
        else:
            hb = BodyMix.parse(body).body()
        self._body = hb
        self._m = oversample
        self._fs_out = float(fs_out)
        self._kernel = kernel
        self._st, self._idx = human_net(hb, self._fs_out * oversample, seeds)
        configure_output(self._st, self._fs_out, oversample)
        self._control = CONTROL_FACTORY(hb, self._idx, geometry, self._fs_out, oversample)
        apply_initial(self._st, self._idx, self._control.initial())
        if start_block:
            self._control.set_clock_block(int(start_block))
        self._raw = np.zeros(0)

    def settle_pitch(self) -> None:
        """ControlModel.settle_pitch() + glot[G_F0] = source.f0; call before the first render."""
        self._control.settle_pitch()
        self._st.glot[L.G_F0] = self._control.source.f0

    def set_source(self, patch: Mapping[str, float | bool]) -> None:
        """ControlModel.set_source + immediate write of the masked apply_source_params() values into the glottis
        (human_voice.rs:283-286, :308-341 at once, mid-block)."""
        p = {_SOURCE_ALIASES.get(k, k): v for k, v in dict(patch).items()}
        self._control.set_source(p)
        values, mask, _ = apply_source_params(self._control.source, self._body.scale)
        self._st.glot[: L.GLOTTIS_NPARAMS][mask] = values[mask]

    def set_gate(self, level: float) -> None:
        self._control.set_gate(level)

    def set_articulation(self, patch: Mapping[str, float], tip_free: bool | None = None) -> None:
        self._control.set_articulation(patch, tip_free)

    def schedule(self, gestures: Sequence[Gesture]) -> None:
        self._control.schedule(gestures)

    def release(self, tag: float) -> None:
        self._control.release(tag)

    def cancel_score(self) -> None:
        self._control.cancel_score()

    def render(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """float32[n]; any n (resumable mid-block); control steps falling into this call are computed now."""
        if out is None:
            out = np.zeros(n, dtype=np.float32)
        st, idx, k = self._st, self._idx, self._kernel
        m = self._m
        per_call = MAX_BLOCKS_PER_CHUNK * int(st.si[L.SI_CTRL_EVERY]) // m
        done = 0
        while done < n:
            n_out = min(n - done, per_call)
            n_int = n_out * m
            steps = control_steps(st, n_int)
            trk = self._control.advance(steps) if steps else _no_rows(idx.n_ph + idx.n_or)
            if self._raw.shape[0] < n_int:
                self._raw = np.zeros(n_int)
            raw = self._raw[:n_int]
            run(k, st, idx, trk, n_int, raw)
            got = k.output_chain(st.sc, st.si, st.dec_c, st.dec_buf, raw, n_int, out[done:done + n_out])
            if got != n_out:
                raise RuntimeError(f"output chain produced {got} of {n_out} samples")
            done += n_out
        return out

    @property
    def clock(self) -> float:
        return self._control.clock

    @property
    def state(self) -> VoiceState:
        return self._st

    @property
    def index(self) -> TractIndex:
        return self._idx

    @property
    def control(self) -> Any:
        """The control model in use (internal; tests)."""
        return self._control
