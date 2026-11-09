"""Network construction: the human airway topology and the packed kernel state (INTERFACES §6.1, §6.2).

Ported from chimera-vocal-synth crates/chimera/src/voice/human_airway.rs (HumanAirway::new) and
airway/network.rs (construction part) (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..core.body import HumanBody
from ..core.types import Seeds

if TYPE_CHECKING:  # TrackChunk is C2's type (control/track.py); only its attributes are used here
    from ..control.track import TrackChunk


@dataclass(frozen=True)
class TractIndex:
    """Topology facts the control model needs (human_airway.rs:69-150), at internal rate fs = 96 000."""

    fs: float
    n_ph: int
    n_or: int
    n_nasal: int
    n_trachea: int
    n_nodes: int
    n_branches: int
    ph_nodes: np.ndarray          # int64 node index per pharynx section
    or_nodes: np.ndarray          # int64 node index per oral section
    glottis_port: int             # branch index
    velum_port: int               # branch index
    lungs: int                    # node index of the pressure node
    sub_node: int                 # trachea.last_node()
    supra_node: int               # pharynx.first_node()

    def branch_of(self, tube: Literal["trachea", "pharynx", "oral"], i: int) -> int:
        """Tube::branch(clamp(i, 0, n - 2)) (human_airway.rs:156-159); tubes are created trachea, pharynx, oral,
        so their internal branches start at 0, n_trachea - 1 and n_trachea - 1 + n_ph - 1."""
        raise NotImplementedError


@dataclass
class VoiceState:
    """All mutable state and constants of network, glottis and output chain in contiguous arrays (INTERFACES §6.2;
    rows/fields: dsp/layout.py). Python objects hold references only; both kernels read and write these arrays."""

    nf: np.ndarray        # float64[NF, n_nodes]
    ni: np.ndarray        # int64[NI, n_nodes]
    nbr_start: np.ndarray  # int64[n_nodes + 1]
    nbr: np.ndarray       # int64[n_nbr]
    bf: np.ndarray        # float64[BF, n_branches]
    bi: np.ndarray        # int64[BI, n_branches]
    rf: np.ndarray        # float64[RF, n_rad]
    ri: np.ndarray        # int64[n_rad] radiator nodes
    noise_f: np.ndarray   # float64[MAX_NOISE, NZ]
    noise_i: np.ndarray   # int64[MAX_NOISE, 3] (src, fwd, bwd)
    glot: np.ndarray      # float64[G]
    glot_i: np.ndarray    # int64[GI]
    sc: np.ndarray        # float64[S]
    si: np.ndarray        # int64[SI]
    dec_c: np.ndarray     # float64[taps] decimator coefficients (49 at m = 2, [1.0] at m = 1)
    dec_buf: np.ndarray   # float64[2 * taps] decimator ring, stored twice as decimator.rs:58 (pos in si)

    def copy(self) -> "VoiceState":
        """Deep copy of every array."""
        raise NotImplementedError


def section_counts(body: HumanBody, fs_internal: float) -> tuple[int, int]:
    """(n_ph, n_or) from the shortest expected tube lengths (human_airway.rs:73-78)."""
    raise NotImplementedError


def human_net(body: HumanBody, fs_internal: float, seeds: Seeds) -> tuple[VoiceState, TractIndex]:
    """Build the human airway network (human_airway.rs:69-154 + network.rs:426-617) and a fresh glottis
    (glottis.rs:108-136) with the given seeds. The output chain is configured for (fs_internal / 2, m = 2); call
    configure_output() for another combination. Tube areas are the TubeSpec initial values: the voice applies the
    control model's initial row with apply_initial() (HumanVoice::new: apply + snap_to_targets)."""
    raise NotImplementedError


def configure_output(st: VoiceState, fs_out: float, oversample: int) -> None:
    """Output chain and control rate of HumanVoice::new (human_voice.rs:203-269): ctrl_every = 64·m, decimator for m,
    output_gain 5.5e-7 with the body-scale ramp, DC-blocker coefficient, limiter hold/release; resets their state."""
    raise NotImplementedError


def apply_initial(st: VoiceState, idx: TractIndex, trk: "TrackChunk") -> None:
    """HumanVoice::new after the network exists: apply the first (only) row of `trk` (tube targets, velum target,
    noise list from an empty list), snap_to_targets (network.rs:674-691), glottis params from the row, lung-pressure
    target; control counter 0 so the first sample runs a control step."""
    raise NotImplementedError


def noise_coefs(lowpass_hz: float, highpass_hz: float, fs: float) -> tuple[float, float]:
    """(lp_coef, hp_coef) as set_noise_sources (network.rs:739-740)."""
    raise NotImplementedError
