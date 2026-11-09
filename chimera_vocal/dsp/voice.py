"""HumanVoice: render loop ControlModel -> kernel -> output chain (INTERFACES §6.5; voice/human_voice.rs).

Ported from chimera-vocal-synth crates/chimera/src/voice/human_voice.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from ..core.body import BodyMix, HumanBody
from ..core.types import Gesture, Seeds
from .kernel_numpy import KERNEL_NUMPY, KernelImpl
from .network import TractIndex, VoiceState

if TYPE_CHECKING:
    from ..control.track import ControlModel, GeometryImpl, TrackChunk

ALLOWED_RATES: tuple[tuple[float, int], ...] = ((48000.0, 2), (96000.0, 1))
"""(fs_out, oversample) combinations HumanVoice accepts (REVIEW-1 P1-15/P1-16); internal rate 96 000 in both."""

MAX_BLOCKS_PER_CHUNK = 750
"""Longest TrackChunk requested from the control model in one go (≈ 1 s at 48 kHz; DESIGN §2.3)."""


def default_control_factory(body: HumanBody, index: TractIndex, geometry: "GeometryImpl | None", fs_out: float,
                            oversample: int) -> "ControlModel":
    """The real control model (C2): ControlModel(body, index, geometry=geometry or GEOMETRY_VEC, ...). Imported here,
    not at module level, so this module loads without the control package."""
    raise NotImplementedError


CONTROL_FACTORY: Callable[..., Any] = default_control_factory
"""Seam for tests: the callable HumanVoice uses to build its control model (same arguments as
default_control_factory). C3's tests replace it with a recorded-track model (tests/dsp/recorded_control.py)."""


def control_steps(st: VoiceState, n_internal: int) -> int:
    """Number of control steps that fall into the next n_internal internal samples (a step runs before every
    internal sample at which the control counter is <= 0)."""
    raise NotImplementedError


def run(k: KernelImpl, st: VoiceState, idx: TractIndex, trk: "TrackChunk", n_internal: int, raw: np.ndarray) -> None:
    """Unpack VoiceState/TractIndex/TrackChunk into arrays and call k.run_samples (rows of trk are consumed from row
    0); raise IndexError if the kernel set si[SI_ERROR] (track exhausted)."""
    raise NotImplementedError


class HumanVoice:
    """= voice/human_voice.rs, CLI path (no Engine wrapper)."""

    def __init__(self, fs_out: float = 48000.0, body: "BodyMix | HumanBody" = BodyMix(), oversample: int = 2, *,
                 kernel: KernelImpl = KERNEL_NUMPY, geometry: "GeometryImpl | None" = None, seeds: Seeds = Seeds(),
                 start_block: int = 0) -> None:
        """(fs_out, oversample) in ALLOWED_RATES else ValueError. ``geometry=None`` means GEOMETRY_VEC (resolved by
        the control factory; the default is not imported at module level). start_block k > 0:
        ControlModel.set_clock_block(k); the caller places this voice's output at output sample 64·k (DESIGN §3.3)."""
        raise NotImplementedError

    def settle_pitch(self) -> None:
        """ControlModel.settle_pitch() + glot[G_F0] = source.f0; call before the first render."""
        raise NotImplementedError

    def set_source(self, patch: Mapping[str, float | bool]) -> None:
        """ControlModel.set_source + immediate write of the masked apply_source_params() values into the glottis
        (human_voice.rs:283-286, :308-341 at once, mid-block)."""
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

    def render(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """float32[n]; any n (resumable mid-block); control steps falling into this call are computed now."""
        raise NotImplementedError

    @property
    def clock(self) -> float:
        raise NotImplementedError

    @property
    def state(self) -> VoiceState:
        raise NotImplementedError

    @property
    def index(self) -> TractIndex:
        raise NotImplementedError

    @property
    def control(self) -> Any:
        """The control model in use (C3-internal; tests)."""
        raise NotImplementedError
