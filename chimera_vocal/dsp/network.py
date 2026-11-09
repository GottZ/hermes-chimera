"""Network construction: the human airway topology and the packed kernel state.

Ported from chimera-vocal-synth crates/chimera/src/voice/human_airway.rs (HumanAirway::new) and
airway/network.rs (construction part) (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..core.body import HumanBody
from ..core.nbmath import js_max, js_round
from ..core.physics import C_SOUND, RHO, WALL_NASAL, WALL_SOFT, WALL_TRACHEA, WallParams
from ..core.types import Seeds
from . import decimator
from . import kernel_numba as K
from . import layout as L
from .glottis import init_glottis

PI = math.pi

if TYPE_CHECKING:  # TrackChunk lives in control/track.py; only its attributes are used here
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
        n, base = {"trachea": (self.n_trachea, 0), "pharynx": (self.n_ph, self.n_trachea - 1),
                   "oral": (self.n_or, self.n_trachea - 1 + self.n_ph - 1)}[tube]
        return base + min(max(int(i), 0), n - 2)


@dataclass
class VoiceState:
    """All mutable state and constants of network, glottis and output chain in contiguous arrays (rows/fields:
    dsp/layout.py). Python objects hold references only; both kernels read and write these arrays, so the kernel may
    change between two calls."""

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
        return VoiceState(*[getattr(self, f).copy() for f in self.__dataclass_fields__])


def section_counts(body: HumanBody, fs_internal: float) -> tuple[int, int]:
    """(n_ph, n_or) from the shortest expected tube lengths (human_airway.rs:73-78)."""
    k = body.scale
    dx_min = (C_SOUND / fs_internal) / 0.93
    ph_min_len = 8.7 * k * (0.55 + 0.45 * body.pharynx) - 1.6 * k
    or_min_len = 8.3 * k - 1.0 * k
    return int(max(8.0, math.floor(ph_min_len / dx_min))), int(max(8.0, math.floor(or_min_len / dx_min)))


class _Builder:
    """AirwayNetwork construction (network.rs:426-617)."""

    def __init__(self, fs: float):
        self.fs = fs
        self.tubes: list[dict] = []
        self.n_nodes = 0
        self.b_a: list[int] = []
        self.b_b: list[int] = []
        self.b_kind: list[int] = []
        self.ports: list[dict | None] = []
        self.pending_fixed: list[int] = []
        self.rad_nodes: list[int] = []

    def add_tube(self, name, n, length, area, wall=None, perimeter=1.0, viscous=1.0, radiating=0.0) -> dict:
        t = {"name": name, "index": len(self.tubes), "n": n, "first": self.n_nodes, "branch_base": len(self.b_a),
             "length": length, "area": area, "wall": wall, "perimeter": perimeter, "viscous": viscous,
             "radiating": radiating}
        self.tubes.append(t)
        self.n_nodes += n
        for i in range(max(n - 1, 0)):
            self._push(t["first"] + i, t["first"] + i + 1, K.KIND_SERIES, None)
        return t

    def _push(self, a, b, kind, spec) -> int:
        self.b_a.append(a)
        self.b_b.append(b)
        self.b_kind.append(kind)
        self.ports.append(spec)
        return len(self.b_a) - 1

    def connect(self, a, b) -> int:
        return self._push(a, b, K.KIND_SERIES, None)

    def add_port(self, a, b, area, length, k_entrance=0.4, k_exit=1.0, slit=0.0, half=True, extra_r=0.0) -> int:
        return self._push(a, b, K.KIND_PORT, {"area": area, "length": length, "kin": k_entrance, "kex": k_exit,
                                               "slit": slit, "half": half, "extra_r": extra_r})

    def add_cavity(self, name, volume, wall) -> int:
        ln = math.cbrt(volume)
        return self.add_tube(name, 1, ln, volume / ln, wall=wall, perimeter=1.2)["first"]

    def add_pressure_node(self, name) -> int:
        node = self.add_tube(name, 1, 1.0, 100.0)["first"]
        self.pending_fixed.append(node)
        return node

    def add_radiation(self, node) -> int:
        self.rad_nodes.append(node)
        return len(self.rad_nodes) - 1


def _finalize(bld: _Builder) -> VoiceState:
    n, nb, nr = bld.n_nodes, len(bld.b_a), len(bld.rad_nodes)
    nf = np.zeros((L.NF, n), dtype=np.float64)
    ni = np.zeros((L.NI, n), dtype=np.int64)
    for t in bld.tubes:
        dx = t["length"] / t["n"]
        for i in range(t["n"]):
            k = t["first"] + i
            a = t["area"][i] if isinstance(t["area"], (list, tuple)) else t["area"]
            for row in (L.N_AREA, L.N_AREA_T, L.N_AMIN, L.N_AMIN_T):
                nf[row, k] = a
            nf[L.N_DX, k] = dx
            nf[L.N_DX_T, k] = dx
            ni[L.NI_TUBE, k] = t["index"]
            nf[L.N_VISC_F, k] = t["viscous"]
            w: WallParams | None = t["wall"]
            if w is not None:
                ni[L.NI_HAS_WALL, k] = 1
                nf[L.N_WALL_M, k] = w.mass
                nf[L.N_WALL_R, k] = w.resistance
                nf[L.N_WALL_K, k] = w.stiffness
            nf[L.N_WALL_PERIM, k] = t["perimeter"] * 2.0 * math.sqrt(PI)
            nf[L.N_WALL_RAD, k] = t["radiating"]
    for k in bld.pending_fixed:
        ni[L.NI_FIXED, k] = 1
    degree = [0] * n
    for a, b in zip(bld.b_a, bld.b_b):
        degree[a] += 1
        degree[b] += 1
    for k in range(n):
        ni[L.NI_JUNCTION, k] = 1 if degree[k] >= 3 else 0
    nbr_start = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        nbr_start[k + 1] = nbr_start[k] + degree[k]
    nbr = np.zeros(int(nbr_start[n]), dtype=np.int64)
    fill = nbr_start[:n].copy()
    for a, b in zip(bld.b_a, bld.b_b):
        nbr[fill[a]] = b
        fill[a] += 1
        nbr[fill[b]] = a
        fill[b] += 1
    bf = np.zeros((L.BF, nb), dtype=np.float64)
    bi = np.zeros((L.BI, nb), dtype=np.int64)
    bi[L.BI_A] = bld.b_a
    bi[L.BI_B] = bld.b_b
    bi[L.BI_KIND] = bld.b_kind
    for j, spec in enumerate(bld.ports):
        if spec is not None:
            bf[L.B_PORT_AREA, j] = spec["area"]
            bf[L.B_PORT_AREA_T, j] = spec["area"]
            bf[L.B_PORT_LEN, j] = spec["length"]
            bf[L.B_PORT_KIN, j] = spec["kin"]
            bf[L.B_PORT_KEX, j] = spec["kex"]
            bf[L.B_PORT_SLIT, j] = spec["slit"]
            bi[L.BI_PORT_HALF, j] = 1 if spec["half"] else 0
            bf[L.B_PORT_EXTRA_R, j] = spec["extra_r"]
    ni[L.NI_RAD_OF] = -1
    for r, node in enumerate(bld.rad_nodes):
        ni[L.NI_RAD_OF, node] = r
    sc = np.zeros(L.S, dtype=np.float64)
    sc[L.S_FS] = bld.fs
    sc[L.S_T] = 1.0 / bld.fs
    sc[L.S_INV_T] = bld.fs
    sc[L.S_CLOSED_AREA] = 2e-4
    sc[L.S_MIN_NODE_AREA] = 0.02
    sc[L.S_SQUEEZE_AREA] = 0.12
    sc[L.S_VISCOUS_SCALE] = 1.0
    sc[L.S_VISCOUS_OMEGA] = 2.0 * PI * 700.0
    sc[L.S_NOISE_SCALE] = 1.0
    sc[L.S_VOL_LP_COEF] = 1.0 - math.exp((-2.0 * PI * 60.0) / bld.fs)
    min_a = sc[L.S_MIN_NODE_AREA]
    for k in range(n):
        nf[L.N_VOL, k] = js_max(nf[L.N_AREA, k], min_a) * nf[L.N_DX, k]
        nf[L.N_INV_A, k] = 1.0 / js_max(nf[L.N_AREA, k], 1e-6)
    si = np.zeros(L.SI, dtype=np.int64)
    si[L.SI_N_NODES] = n
    si[L.SI_N_BRANCHES] = nb
    si[L.SI_N_RAD] = nr
    si[L.SI_FLAGS] = L.F_DEFAULT
    noise_i = np.full((L.MAX_NOISE, 3), -1, dtype=np.int64)
    return VoiceState(nf, ni, nbr_start, nbr, bf, bi, np.zeros((L.RF, nr)), np.array(bld.rad_nodes, dtype=np.int64),
                      np.zeros((L.MAX_NOISE, L.NZ)), noise_i, np.zeros(L.G), np.zeros(L.GI, dtype=np.int64), sc, si,
                      np.ones(1), np.zeros(2))


def human_net(body: HumanBody, fs_internal: float, seeds: Seeds) -> tuple[VoiceState, TractIndex]:
    """Build the human airway network (human_airway.rs:69-154 + network.rs:426-617) and a fresh glottis
    (glottis.rs:108-136) with the given seeds. The output chain is configured for (fs_internal / 2, m = 2); call
    configure_output() for another combination. Tube areas are the TubeSpec initial values: the voice applies the
    control model's initial row with apply_initial() (HumanVoice::new: apply + snap_to_targets)."""
    fs = float(fs_internal)
    k = body.scale
    dx_min = (C_SOUND / fs) / 0.93
    n_ph, n_or = section_counts(body, fs)
    bld = _Builder(fs)
    tr_len = 12.0 * k
    trachea = bld.add_tube("trachea", int(js_max(6.0, math.floor(tr_len / js_max(dx_min, 0.9 * k)))), tr_len,
                           2.5 * k * k, wall=WALL_TRACHEA)
    pharynx = bld.add_tube("pharynx", n_ph, 8.8 * k, 3.0 * k * k, wall=WALL_SOFT, perimeter=1.25, radiating=0.25)
    oral = bld.add_tube("oral", n_or, 8.3 * k, 3.0 * k * k, wall=WALL_SOFT, perimeter=1.5, radiating=0.25)
    na_len = 11.4 * k
    n_na = int(js_max(8.0, math.floor(na_len / js_max(dx_min, 0.5 * k))))
    na_areas = []
    for i in range(n_na):
        u = (i + 0.5) / n_na
        a = 2.6 if u < 0.18 else (2.0 - 0.6 * (u - 0.18) if u < 0.75 else (1.4 if u < 0.9 else 1.0))
        na_areas.append(a * k * k)
    nasal = bld.add_tube("nasal", n_na, na_len, na_areas, wall=WALL_NASAL, perimeter=3.2, viscous=2.5)
    fossa = bld.add_tube("piriform", 3, 2.0 * k, 0.9 * k * k, wall=WALL_SOFT, perimeter=1.3)
    lungs = bld.add_pressure_node("lungs")
    tr_a = 2.5 * k * k
    bld.add_port(lungs, trachea["first"], tr_a, 1.0, 0.0, 0.0, extra_r=(0.5 * RHO * C_SOUND) / tr_a)
    sub_node = trachea["first"] + trachea["n"] - 1
    supra_node = pharynx["first"]
    glottis_port = bld.add_port(sub_node, supra_node, 0.0, 0.3 * k, 0.37, 1.0, slit=1.4 * k)
    ph_last = pharynx["first"] + n_ph - 1
    bld.connect(ph_last, oral["first"])
    velum_port = bld.add_port(ph_last, nasal["first"], 0.0, 1.0 * k, 0.3, 0.8)
    fossa_node = pharynx["first"] + int(min(float(n_ph - 1), js_round((2.6 / 8.8) * n_ph)))
    bld.connect(fossa_node, fossa["first"])
    sinus = bld.add_cavity("sinus", 14.0 * k * k * k, None)
    bld.add_port(nasal["first"] + int(js_round(n_na * 0.45)), sinus, 0.12 * k * k, 0.5 * k, 0.0, 0.0, half=False)
    bld.add_radiation(oral["first"] + n_or - 1)
    bld.add_radiation(nasal["first"] + n_na - 1)
    st = _finalize(bld)
    st.glot, st.glot_i = init_glottis(fs, seeds)
    st.si[L.SI_NET_RNG] = (seeds.net & 0xFFFFFFFF) or 1
    st.si[L.SI_GLOTTIS_PORT] = glottis_port
    st.si[L.SI_VELUM_PORT] = velum_port
    st.si[L.SI_LUNGS] = lungs
    st.si[L.SI_SUB_NODE] = sub_node
    st.si[L.SI_SUPRA_NODE] = supra_node
    st.sc[L.S_BODY_SCALE] = k
    configure_output(st, fs / 2.0, 2)
    idx = TractIndex(fs, n_ph, n_or, n_na, trachea["n"], st.nf.shape[1], st.bf.shape[1],
                     np.arange(pharynx["first"], pharynx["first"] + n_ph, dtype=np.int64),
                     np.arange(oral["first"], oral["first"] + n_or, dtype=np.int64),
                     glottis_port, velum_port, lungs, sub_node, supra_node)
    return st, idx


def configure_output(st: VoiceState, fs_out: float, oversample: int) -> None:
    """Output chain and control rate of HumanVoice::new (human_voice.rs:203-269): ctrl_every = 64·m, decimator for m,
    output_gain 5.5e-7 with the body-scale ramp, DC-blocker coefficient, limiter hold/release; resets their state."""
    m = int(oversample)
    st.dec_c = decimator.coefficients(m)
    st.dec_buf = decimator.new_buffer(st.dec_c)
    si, sc = st.si, st.sc
    si[L.SI_M] = m
    si[L.SI_DEC_POS] = 0
    si[L.SI_DEC_COUNT] = 0
    si[L.SI_CTRL_EVERY] = 64 * m
    si[L.SI_CTRL_COUNTER] = 0
    sc[L.S_FS_OUT] = fs_out
    k = sc[L.S_BODY_SCALE]
    gain = 5.5e-7
    gain *= 1.0 if k >= 1.0 else (0.55 if k <= 0.9 else 0.55 + 0.45 * (k - 0.9) / 0.1)
    sc[L.S_OUTPUT_GAIN] = gain
    sc[L.S_DC_COEF] = math.exp((-2.0 * PI * 25.0) / fs_out)
    sc[L.S_DC_X] = 0.0
    sc[L.S_DC_Y] = 0.0
    si[L.SI_LIM_HOLD_N] = int(js_round(0.025 * fs_out))
    si[L.SI_LIM_HOLD] = 0
    sc[L.S_LIM_RELEASE] = math.exp(-1.0 / (0.2 * fs_out))
    sc[L.S_LIM_ENV] = 0.0
    sc[L.S_DEC_VALUE] = 0.0


def apply_initial(st: VoiceState, idx: TractIndex, trk: "TrackChunk") -> None:
    """HumanVoice::new after the network exists: apply the first (only) row of `trk` (tube targets, velum target,
    noise list from an empty list), snap_to_targets (network.rs:674-691), glottis params from the row, lung-pressure
    target; control counter 0 so the first sample runs a control step."""
    row = 0
    inherit = np.full((1, L.MAX_NOISE), -1, dtype=np.int64)    # from an empty list: every source starts at 0
    K.apply_targets(st.nf, st.bf, st.noise_f, st.noise_i, st.sc, st.si, idx.ph_nodes, idx.or_nodes,
                    np.asarray(trk.tube_area, dtype=np.float64), np.asarray(trk.tube_min, dtype=np.float64),
                    np.asarray(trk.tube_len, dtype=np.float64), np.asarray(trk.velum_area, dtype=np.float64),
                    np.asarray(trk.noise_n, dtype=np.int64), np.asarray(trk.noise_idx, dtype=np.int64),
                    np.asarray(trk.noise_par, dtype=np.float64), inherit, row)
    K.snap_to_targets(st.nf, st.bf, st.bi, st.sc)
    st.glot[: L.GLOTTIS_NPARAMS] = np.asarray(trk.glot_par, dtype=np.float64)[row]
    st.si[L.SI_CTRL_COUNTER] = 0
    st.si[L.SI_BLOCK_POS] = 0


def noise_coefs(lowpass_hz: float, highpass_hz: float, fs: float) -> tuple[float, float]:
    """(lp_coef, hp_coef) as set_noise_sources (network.rs:739-740)."""
    return 1.0 - math.exp((-2.0 * PI * lowpass_hz) / fs), math.exp((-2.0 * PI * highpass_hz) / fs)
