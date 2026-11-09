"""The control side of the human airway (voice/human_airway.rs:162-325, airway/network.rs:725-745): geometry ->
slew-limited tube targets, velum port target, turbulence noise sources and their filter inheritance.

Nothing here touches acoustic state: the network's ``set_noise_sources`` keeps the filter state (hp, lp1, lp2) of
a source whose (src, fwd) pair is in the list active at the call (first match, network.rs:727-745). A control step
with a contact correction calls it twice (apply -> list A, reapply -> list B; human_voice.rs:468-474), so the final
list inherits through A, not from the previous block's list. ``block_inherit`` composes the calls into one index
per source into the PREVIOUS block's final list (TrackChunk.noise_inherit). Comparing the final list with the
previous block's list instead would keep filter states the first call dropped: the template's "Hello world."
then differs from about sample 40 066 by up to 1.3 % of its peak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Sequence

import math

import numpy as np

from ..core.body import HumanBody
from ..core.nbmath import js_max, js_min, js_round, smoothstep
from . import frame_layout as fl
from .geometry import GeometryImpl

if TYPE_CHECKING:  # the acoustics' topology index; annotations only, no import at run time
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
    out = []
    for s in new:
        k = -1
        for i, o in enumerate(active):
            if o[0] == s[0] and o[1] == s[1]:
                k = i
                break
        out.append(k)
    return out


def compose_inherit(first: Sequence[int], second: Sequence[int]) -> list[int]:
    """Chain two calls: ``first`` maps the intermediate list into the previous one, ``second`` maps the new list
    into the intermediate one; result[j] = first[second[j]] if second[j] >= 0 else -1."""
    return [int(first[j]) if j >= 0 else -1 for j in second]


def block_inherit(prev: Sequence[Sequence[int]], calls: Sequence[Sequence[Sequence[int]]]) -> list[int]:
    """``noise_inherit`` of one control block: ``calls`` are the lists set by the block's set_noise_sources calls
    in order (one, or two with a contact re-application); the result indexes ``prev`` (the previous block's final
    list) for each source of the last call (-1 = filter state starts at 0)."""
    if not calls:
        raise ValueError("a control block sets the noise sources at least once")
    m = inherit_from(prev, calls[0])
    for a, b in zip(calls, calls[1:]):
        m = compose_inherit(m, inherit_from(a, b))
    return m


def section_slew(cur: np.ndarray, old: np.ndarray, mins: np.ndarray, old_mins: np.ndarray, dt: float,
                 body_scale: float) -> None:
    """The slew limit of HumanAirway::commit (human_airway.rs:187-253) on one tube, in place: area rises at most
    (open_r + gain*A)*dt per step, falls at most (close_r(old) + gain*old)*dt; the same for the sub-section
    minima, which are then clamped to [0, area]."""
    k2 = body_scale * body_scale
    open_r = MAX_AREA_RATE * k2
    gain = AREA_RATE_GAIN
    seal_r = SEAL_RATE * k2
    lo = SEAL_LO * k2
    hi = SEAL_HI * k2
    ease = SEAL_EASE * k2

    def close_r(area: float) -> float:
        if area < ease:
            return seal_r * math.sqrt(js_max(0.0, area) / ease)
        if area <= lo:
            u = 0.0
        elif area >= hi:
            u = 1.0
        else:
            u = (area - lo) / (hi - lo)
        return seal_r + (open_r - seal_r) * u * u * (3.0 - 2.0 * u)

    for i in range(len(cur)):
        c, o = float(cur[i]), float(old[i])
        d = c - o
        if d > 0.0:
            lim = (open_r + gain * c) * dt
            if d > lim:
                cur[i] = o + lim
        else:
            lim = (close_r(o) + gain * o) * dt
            if d < -lim:
                cur[i] = o - lim
        m, om = float(mins[i]), float(old_mins[i])
        dm = m - om
        if dm > 0.0:
            lim_m = (open_r + gain * m) * dt
            if dm > lim_m:
                mins[i] = om + lim_m
        else:
            lim_m = (close_r(om) + gain * om) * dt
            if dm < -lim_m:
                mins[i] = om - lim_m
        if mins[i] > cur[i]:
            mins[i] = cur[i]
        if mins[i] < 0.0:
            mins[i] = 0.0


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
        self.body = body
        self.index = index
        self.geometry = geometry
        self.n_ph = int(index.n_ph)
        self.n_or = int(index.n_or)
        self.tract = fl.tract_constants(body)
        self.frame = fl.alloc_frame()
        self.ph_areas = np.zeros(self.n_ph)
        self.or_areas = np.zeros(self.n_or)
        self.ph_min = np.zeros(self.n_ph)
        self.or_min = np.zeros(self.n_or)
        self.ph_len = 8.8
        self.or_len = 8.3
        self.velum_target = 0.0
        self.noise = []
        self.calls = []
        self._base = None
        self._base_dt = 0.0
        self._lens = np.zeros(2)
        g_tr = index.branch_of("trachea", index.n_trachea - 2)
        self._glottal = NoiseSpec(int(index.glottis_port), index.branch_of("pharynx", 1), g_tr, NOISE_GAIN_GLOTTIS,
                                  3500.0, 300.0)

    def apply(self, artic: np.ndarray, dt: float) -> None:
        """Geometry -> targets for the next block; dt > 0 enables the slew limit against the areas committed at
        the start of this step (human_airway.rs:163-173)."""
        if dt > 0.0:
            self._base = (self.ph_areas.copy(), self.or_areas.copy(), self.ph_min.copy(), self.or_min.copy())
        self._base_dt = dt
        self.calls = []
        self._commit(artic, dt)

    def reapply(self, artic: np.ndarray) -> None:
        """Recompute within the same control step, slew-limited against the same baseline (human_airway.rs:178-180)."""
        self._commit(artic, self._base_dt)

    def contacts(self) -> tuple[float, tuple[float, float], float, tuple[float, float]]:
        """(contact_tip, tip normal, contact_body, body normal) of the last geometry (human.rs:634-639)."""
        f = self.frame
        return (float(f[fl.F_CONTACT_TIP]), (float(f[fl.F_CONTACT_TIP_NX]), float(f[fl.F_CONTACT_TIP_NY])),
                float(f[fl.F_CONTACT_BODY]), (float(f[fl.F_CONTACT_BODY_NX]), float(f[fl.F_CONTACT_BODY_NY])))

    def _commit(self, artic: np.ndarray, dt: float) -> None:
        a = np.ascontiguousarray(artic, dtype=np.float64)
        self.geometry.compute(a, self.tract, self.frame)
        self.geometry.resample(a, self.tract, self.frame, self.ph_areas, self.or_areas, self.ph_min, self.or_min,
                               self._lens)
        if dt > 0.0 and self._base is not None:
            b = self._base
            section_slew(self.ph_areas, b[0], self.ph_min, b[2], dt, self.body.scale)
            section_slew(self.or_areas, b[1], self.or_min, b[3], dt, self.body.scale)
        self.ph_len = float(self._lens[0])
        self.or_len = float(self._lens[1])
        self.velum_target = float(self.frame[fl.F_VELUM_PORT])
        self.noise = self._noise_list()
        self.calls.append(self.noise)

    def _noise_list(self) -> list[NoiseSpec]:
        """HumanAirway::update_noise (human_airway.rs:266-325)."""
        idx = self.index
        out = [self._glottal]
        n_ph, n_or = self.n_ph, self.n_or
        min_a, min_i = math.inf, -1
        for i in range(n_ph + n_or):
            area = float(self.ph_min[i]) if i < n_ph else float(self.or_min[i - n_ph])
            if area > 0.0 and area < min_a:
                min_a, min_i = area, i
        scale = self.body.scale
        k2 = scale * scale
        w_area = 1.0 - smoothstep(0.35 * k2, 0.9 * k2, min_a) if min_i >= 0 else 0.0
        if w_area > 0.0:
            n_tot = n_ph + n_or
            dx_ph = self.ph_len / n_ph
            dx_or = self.or_len / n_or

            def to_branch(gi: int) -> int:
                if gi < n_ph:
                    return idx.branch_of("pharynx", gi)
                if gi == n_ph:
                    return -1
                return idx.branch_of("oral", gi - n_ph)

            dx = dx_ph if min_i < n_ph else dx_or
            off = int(js_max(1.0, js_round((0.9 * scale) / dx)))
            src = to_branch(min(min_i, n_tot - 2))
            fwd = to_branch(min(n_tot - 2, min_i + off))
            bwd = to_branch(max(0, min_i - off))
            if src >= 0 and fwd >= 0 and bwd >= 0:
                out.append(NoiseSpec(src, fwd, bwd, NOISE_GAIN_CONSTRICTION * w_area, 2600.0, 350.0))
                if min_i < n_ph:
                    s_min = (min_i + 0.5) * dx_ph
                else:
                    s_min = self.ph_len + ((min_i - n_ph) + 0.5) * dx_or
                teeth_s = float(self.frame[fl.F_TEETH_S])
                behind = (teeth_s - s_min) / scale
                w_dist = smoothstep(0.15, 0.5, behind) * (1.0 - smoothstep(2.6, 4.0, behind))
                w_jet = 1.0 - smoothstep(0.2 * k2, 0.5 * k2, min_a)
                wt = w_dist * w_jet
                if wt > 1e-3:
                    teeth_i = n_ph + int(js_min(float(n_or - 2), js_max(0.0, js_round((teeth_s - self.ph_len) / dx_or))))
                    tb = to_branch(teeth_i)
                    if tb >= 0:
                        out.append(NoiseSpec(src, tb, bwd, NOISE_GAIN_TEETH * wt, 5000.0, 1200.0))
        return out
