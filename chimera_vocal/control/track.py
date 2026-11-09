"""ControlModel -> TrackChunk (INTERFACES §6.3): everything HumanVoice::control() does except acoustics
(voice/human_voice.rs:426-542): gesture blend, articulator step, source channels, pruning, clock, geometry and
slew limit, contact constraint with re-application, noise sources with filter inheritance, breath gate, lung
pressure target, pitch glide and the glottis parameter row of each block.

The control step runs at the start of every block of ctrl_every = 64*m internal samples (dt = 64/fs_out); the clock
grows by repeated ``+= dt`` (human_voice.rs:467), never ``k*dt``. Commands (set_source, schedule, ...) apply between
``advance`` calls, i.e. at block boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, NamedTuple, Sequence

import numpy as np

from ..core.body import HumanBody
from ..core.types import Gesture, SourceParams
from .geometry import GeometryImpl

if TYPE_CHECKING:  # C3 type, only used for annotations here
    from ..dsp.network import TractIndex


class TrackChunk(NamedTuple):
    """One row per control block, in order; arrays C-contiguous. Plain container, never passed into a JIT
    function (INTERFACES §6.3, §6.4a rule 4)."""
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
    """Control side of HumanVoice (INTERFACES §6.3).

    Public: ``source`` (SourceParams), ``clock`` (s), ``f0`` (glide state). C2-internal, for tests: ``dt``,
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
        raise NotImplementedError

    def initial(self) -> TrackChunk:
        """One row: the state after construction, apply(x, 0) without slew (human_voice.rs:254-256); noise_inherit
        all -1."""
        raise NotImplementedError

    def advance(self, n_blocks: int) -> TrackChunk:
        """``n_blocks`` consecutive control steps (human_voice.rs:426-509); block 0 of the first call inherits
        from the list of ``initial``."""
        raise NotImplementedError

    def set_source(self, patch: Mapping[str, float | bool]) -> None:
        raise NotImplementedError

    def set_gate(self, level: float) -> None:
        raise NotImplementedError

    def set_articulation(self, patch: Mapping[str, float], tip_free: bool | None = None) -> None:
        raise NotImplementedError

    def schedule(self, gestures: Sequence[Gesture]) -> None:
        raise NotImplementedError

    def release(self, tag: float) -> None:
        raise NotImplementedError

    def cancel_score(self) -> None:
        raise NotImplementedError

    def settle_pitch(self) -> None:
        """Glide state f0 := source.f0 (human_voice.rs:225, :499-501; D§3.3)."""
        raise NotImplementedError

    def set_clock_block(self, k: int) -> None:
        """clock := 0.0 plus dt added k times (start_block, D§3.3)."""
        raise NotImplementedError
