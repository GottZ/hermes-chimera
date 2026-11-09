"""The control side of the human airway (voice/human_airway.rs:162-325, airway/network.rs:725-745): geometry ->
slew-limited tube targets, velum port target, turbulence noise sources and their filter inheritance.

Nothing here touches acoustic state: the network's ``set_noise_sources`` keeps the filter state (hp, lp1, lp2) of
a source whose (src, fwd) pair is in the list active at the call (first match, network.rs:727-745). A control step
with a contact correction calls it twice (apply -> list A, reapply -> list B; human_voice.rs:468-474), so the final
list inherits through A, not from the previous block's list. ``block_inherit`` composes the calls into one index
per source into the PREVIOUS block's final list (TrackChunk.noise_inherit, REVIEW-1 P1-01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

from ..core.body import HumanBody
from .geometry import GeometryImpl

if TYPE_CHECKING:  # C3 type, only used for annotations here
    from ..dsp.network import TractIndex


class NoiseSpec(NamedTuple):
    """airway/network.rs:165-186: one turbulence source."""
    src: int             # branch whose Reynolds number drives the source
    fwd: int             # branch to inject into for positive flow
    bwd: int             # branch to inject into for negative flow
    gain: float
    lowpass_hz: float
    highpass_hz: float


# human_airway.rs:142-148
NOISE_GAIN_GLOTTIS = 3.2e-6
NOISE_GAIN_CONSTRICTION = 2.2e-6
NOISE_GAIN_TEETH = 3.0e-6
MAX_AREA_RATE = 45.0
AREA_RATE_GAIN = 40.0
SEAL_RATE = 5.0
SEAL_LO = 0.03
SEAL_HI = 0.4
SEAL_EASE = 0.03


def inherit_from(active: Sequence[Sequence[int]], new: Sequence[Sequence[int]]) -> list[int]:
    """One ``set_noise_sources`` call (network.rs:727-745): for each source of ``new`` (entries start with
    (src, fwd)), the index of the FIRST source of ``active`` with the same (src, fwd), else -1."""
    raise NotImplementedError


def compose_inherit(first: Sequence[int], second: Sequence[int]) -> list[int]:
    """Chain two calls: ``first`` maps the intermediate list into the previous one, ``second`` maps the new list
    into the intermediate one; result[j] = first[second[j]] if second[j] >= 0 else -1."""
    raise NotImplementedError


def block_inherit(prev: Sequence[Sequence[int]], calls: Sequence[Sequence[Sequence[int]]]) -> list[int]:
    """``noise_inherit`` of one control block: ``calls`` are the lists set by the block's set_noise_sources calls
    in order (one, or two with a contact re-application); the result indexes ``prev`` (the previous block's final
    list) for each source of the last call (-1 = filter state starts at 0)."""
    raise NotImplementedError


def section_slew(cur: np.ndarray, old: np.ndarray, mins: np.ndarray, old_mins: np.ndarray, dt: float,
                 body_scale: float) -> None:
    """The slew limit of HumanAirway::commit (human_airway.rs:187-253) on one tube, in place: area rises at most
    (open_r + gain*A)*dt per step, falls at most (close_r(old) + gain*old)*dt; the same for the sub-section
    minima, which are then clamped to [0, area]."""
    raise NotImplementedError


class Airway:
    """HumanAirway minus the network (human_airway.rs:19-65, :162-325).

    State after apply/reapply: ``ph_areas``/``or_areas``/``ph_min``/``or_min`` (slew-limited raw arrays, before the
    network's js_max(0, .)), ``ph_len``/``or_len``, ``velum_target`` (frame velum port), ``frame`` (last geometry),
    ``noise`` (the active list, NoiseSpec), ``calls`` (the lists set during the current control step, oldest
    first; apply() starts a new step).
    """

    n_ph: int
    n_or: int
    ph_areas: np.ndarray
    or_areas: np.ndarray
    ph_min: np.ndarray
    or_min: np.ndarray
    ph_len: float
    or_len: float
    velum_target: float
    frame: np.ndarray
    noise: list[NoiseSpec]
    calls: list[list[NoiseSpec]]

    def __init__(self, body: HumanBody, index: "TractIndex", geometry: GeometryImpl) -> None:
        raise NotImplementedError

    def apply(self, artic: np.ndarray, dt: float) -> None:
        """Geometry -> targets for the next block; dt > 0 enables the slew limit against the areas committed at
        the start of this step (human_airway.rs:163-173)."""
        raise NotImplementedError

    def reapply(self, artic: np.ndarray) -> None:
        """Recompute within the same control step, slew-limited against the same baseline (human_airway.rs:178-180)."""
        raise NotImplementedError

    def contacts(self) -> tuple[float, tuple[float, float], float, tuple[float, float]]:
        """(contact_tip, tip normal, contact_body, body normal) of the last geometry (human.rs:634-639)."""
        raise NotImplementedError
