"""K-3.1 test_topology: human_net per body = V2 (tests/vectors/topology.json) — counts, index, branches, neighbour
lists, constants, and the state right after HumanVoice::new (apply + snap_to_targets + apply_source)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import recorded_control as rc
from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.types import Seeds
from chimera_vocal.dsp import glottis as glot_mod
from chimera_vocal.dsp import layout as L
from chimera_vocal.dsp import network
from support import assert_exact, assert_t2, rel, vector

V2 = vector("topology")
LABELS = [b["label"] for b in V2["bodies"]]
FS = 96000.0
GLOTTIS_BAND = (3500.0, 300.0)   # NoiseSpec band of the aspiration source (human_airway.rs:271)

# V2 `initial.zero_fields` -> where the field lives in VoiceState
ZERO_FIELDS = {
    "p": ("nf", L.N_P), "u": ("bf", L.B_U), "psrc": ("bf", L.B_PSRC), "wall_y": ("nf", L.N_WALL_Y),
    "wall_u": ("nf", L.N_WALL_U), "rad_u": ("rf", L.R_U), "ext_flow": ("nf", L.N_EXT_FLOW),
    "area_step": ("nf", L.N_AREA_S), "amin_step": ("nf", L.N_AMIN_S), "dx_step": ("nf", L.N_DX_S),
    "half_l": ("nf", L.N_HALF_L), "half_r": ("nf", L.N_HALF_R), "sqrt_a": ("nf", L.N_SQRT_A),
    "inv_amin": ("nf", L.N_INV_AMIN), "inv_lsum": ("nf", L.N_INV_LSUM), "spill": ("nf", L.N_SPILL),
    "vol_src1": ("nf", L.N_VOL_SRC1), "vol_src2": ("nf", L.N_VOL_SRC2), "wall_v": ("nf", L.N_WALL_V),
    "mom": ("bf", L.B_MOM), "l_prev": ("bf", L.B_L_PREV), "rad_ul": ("rf", L.R_UL), "rad_l_prev": ("rf", L.R_L_PREV),
    "rad_prev_total": ("sc", L.S_RAD_PREV_TOTAL),
}


def _body(v):
    return BodyMix(v["body_mix"]["female"], v["body_mix"]["child"]).body()


def _v2(label):
    return next(b for b in V2["bodies"] if b["label"] == label)


def initial_row(v) -> rc.TrackChunk:
    """The control model's constructor row as V2 records it (ControlModel.initial())."""
    ini = v["initial"]
    row = rc.make_chunk(1, v["n_ph"] + v["n_or"])
    row.tube_area[0] = ini["ph_areas"] + ini["or_areas"]
    row.tube_min[0] = ini["ph_min"] + ini["or_min"]
    row.tube_len[0] = (ini["ph_len"], ini["or_len"])
    row.velum_area[0] = ini["frame_velum_port"]
    row.noise_n[0] = len(ini["noise"])
    for j, (src, fwd, bwd, gain, _lp, _hp) in enumerate(ini["noise"]):
        row.noise_idx[0, j] = (src, fwd, bwd)
        row.noise_par[0, j] = (gain,) + GLOTTIS_BAND
    row.glot_par[0] = ini["glottis_params"]
    row.p_lung[0] = ini["voice"]["p_lung"]
    return row


@pytest.mark.parametrize("label", LABELS)
def test_topology_index(label):
    v = _v2(label)
    body = _body(v)
    assert network.section_counts(body, FS) == (v["n_ph"], v["n_or"])
    st, idx = network.human_net(body, FS, Seeds())
    assert idx.fs == FS
    got = (idx.n_ph, idx.n_or, idx.n_trachea, idx.n_nasal, idx.n_nodes, idx.n_branches)
    assert got == (v["n_ph"], v["n_or"], v["n_trachea"], v["n_nasal"], v["n_nodes"], v["n_branches"])
    assert idx.ph_nodes.dtype == np.int64 and idx.or_nodes.dtype == np.int64
    assert_exact(idx.ph_nodes.tolist(), v["ph_nodes"], "ph_nodes")
    assert_exact(idx.or_nodes.tolist(), v["or_nodes"], "or_nodes")
    assert (idx.glottis_port, idx.velum_port, idx.lungs, idx.sub_node, idx.supra_node) == (
        v["glottis_port"], v["velum_port"], v["lungs"], v["sub_node"], v["supra_node"])
    for tube, want in v["branch_of"].items():
        assert [idx.branch_of(tube, i) for i in range(len(want))] == want, tube
        assert idx.branch_of(tube, -3) == want[0] and idx.branch_of(tube, len(want) + 4) == want[-1], tube


@pytest.mark.parametrize("label", LABELS)
def test_topology_structure(label):
    v = _v2(label)
    st, idx = network.human_net(_body(v), FS, Seeds())
    n, nb = v["n_nodes"], v["n_branches"]
    assert st.nf.shape == (L.NF, n) and st.ni.shape == (L.NI, n)
    assert st.bf.shape == (L.BF, nb) and st.bi.shape == (L.BI, nb)
    assert st.rf.shape == (L.RF, v["n_rad"]) and st.ri.shape == (v["n_rad"],)
    for a, dt in ((st.nf, np.float64), (st.bf, np.float64), (st.rf, np.float64), (st.noise_f, np.float64),
                  (st.glot, np.float64), (st.sc, np.float64), (st.dec_c, np.float64), (st.dec_buf, np.float64),
                  (st.ni, np.int64), (st.bi, np.int64), (st.ri, np.int64), (st.noise_i, np.int64),
                  (st.glot_i, np.int64), (st.si, np.int64), (st.nbr_start, np.int64), (st.nbr, np.int64)):
        assert a.dtype == dt and a.flags.c_contiguous
    br = v["branches"]
    assert_exact(st.bi[L.BI_A].tolist(), br["a"], "branch a")
    assert_exact(st.bi[L.BI_B].tolist(), br["b"], "branch b")
    assert_exact(st.bi[L.BI_KIND].tolist(), br["kind"], "branch kind")
    nodes = v["nodes"]
    assert_exact(np.flatnonzero(st.ni[L.NI_JUNCTION]).tolist(), nodes["junction"], "junction")
    assert_exact(np.flatnonzero(st.ni[L.NI_FIXED]).tolist(), nodes["fixed"], "fixed")
    assert_exact(np.flatnonzero(st.ni[L.NI_HAS_WALL]).tolist(), nodes["has_wall"], "has_wall")
    assert_exact(st.ni[L.NI_TUBE].tolist(), nodes["node_tube"], "node_tube")
    assert_exact(st.ni[L.NI_RAD_OF].tolist(), nodes["rad_of"], "rad_of")
    assert_exact(st.nbr_start.tolist(), nodes["nbr_start"], "nbr_start")
    assert_exact(st.nbr.tolist(), nodes["nbr"], "nbr")
    assert_exact(st.ri.tolist(), v["radiators"], "radiators")
    for key, row in (("wall_m", L.N_WALL_M), ("wall_r", L.N_WALL_R), ("wall_k", L.N_WALL_K),
                     ("wall_perim", L.N_WALL_PERIM), ("wall_rad", L.N_WALL_RAD), ("visc_f", L.N_VISC_F)):
        assert_t2(st.nf[row].tolist(), nodes[key], key)
    ports = {p["branch"]: p for p in v["ports"]}
    for j in range(nb):
        p = ports.get(j)
        want = ((p["length"], p["k_entrance"], p["k_exit"], p["slit_width"], p["extra_resistance"])
                if p else (0.0, 0.0, 0.0, 0.0, 0.0))
        got = tuple(st.bf[r, j] for r in (L.B_PORT_LEN, L.B_PORT_KIN, L.B_PORT_KEX, L.B_PORT_SLIT, L.B_PORT_EXTRA_R))
        assert_t2(got, want, f"port spec of branch {j}")
        assert st.bi[L.BI_PORT_HALF, j] == (1 if p and p["half_sections"] else 0), j
        assert st.bf[L.B_PORT_AREA, j] == (p["spec_area"] if p else 0.0), j
    c = v["constants"]
    si = st.si
    assert (si[L.SI_N_NODES], si[L.SI_N_BRANCHES], si[L.SI_N_RAD]) == (n, nb, v["n_rad"])
    assert (si[L.SI_GLOTTIS_PORT], si[L.SI_VELUM_PORT], si[L.SI_LUNGS], si[L.SI_SUB_NODE], si[L.SI_SUPRA_NODE]) == (
        v["glottis_port"], v["velum_port"], v["lungs"], v["sub_node"], v["supra_node"])
    for key, field in (("fs", L.S_FS), ("t", L.S_T), ("inv_t", L.S_INV_T), ("closed_area", L.S_CLOSED_AREA),
                       ("min_node_area", L.S_MIN_NODE_AREA), ("squeeze_area", L.S_SQUEEZE_AREA),
                       ("viscous_scale", L.S_VISCOUS_SCALE), ("viscous_omega", L.S_VISCOUS_OMEGA),
                       ("noise_scale", L.S_NOISE_SCALE), ("vol_lp_coef", L.S_VOL_LP_COEF)):
        assert rel(float(st.sc[field]), c[key], 1e-300) <= 1e-15, (key, st.sc[field], c[key])
    flags = int(si[L.SI_FLAGS])
    assert bool(flags & L.F_VOLUME_SOURCE) == c["enable_volume_source"]
    assert bool(flags & L.F_JUNCTION_FLOOR) == c["enable_junction_floor"]
    assert bool(flags & L.F_ENERGY_SAFE) == c["energy_safe"]
    assert bool(flags & L.F_SYMMETRIC_KINETIC) == c["symmetric_kinetic"]
    assert flags & L.F_FLUSH_SUBNORMALS, "subnormal flush must be on by default"
    assert si[L.SI_NET_RNG] == 12345 and si[L.SI_ERROR] == 0


@pytest.mark.parametrize("label", LABELS)
def test_topology_after_constructor(label):
    """apply_initial + configure_output(48000, 2) reproduce the state right after HumanVoice::new."""
    v = _v2(label)
    ini = v["initial"]
    st, idx = network.human_net(_body(v), FS, Seeds())
    network.configure_output(st, 48000.0, 2)
    network.apply_initial(st, idx, initial_row(v))
    nf = st.nf
    assert_t2(nf[L.N_AREA].tolist(), ini["area"], "area")
    assert_t2(nf[L.N_AMIN].tolist(), ini["amin"], "amin")
    assert_t2(nf[L.N_DX].tolist(), ini["dx"], "dx")
    for cur, tgt in ((L.N_AREA, L.N_AREA_T), (L.N_AMIN, L.N_AMIN_T), (L.N_DX, L.N_DX_T)):
        assert_exact(nf[tgt].tolist(), nf[cur].tolist(), f"target row {tgt}")
    min_a = st.sc[L.S_MIN_NODE_AREA]
    vol = [max(a, min_a) * d for a, d in zip(nf[L.N_AREA], nf[L.N_DX])]
    inv_a = [1.0 / max(a, 1e-6) for a in nf[L.N_AREA]]
    assert_exact(nf[L.N_VOL].tolist(), vol, "vol (network.rs:679)")
    assert_exact(nf[L.N_INV_A].tolist(), inv_a, "inv_a (network.rs:680)")
    missing = set(ini["zero_fields"]) - set(ZERO_FIELDS)
    assert not missing, f"V2 zero fields without a VoiceState row: {missing}"
    for name in ini["zero_fields"]:
        arr, row = ZERO_FIELDS[name]
        a = getattr(st, arr)[row]
        assert np.all(a == 0.0), name
    ports = {p["branch"]: p for p in v["ports"]}
    for j, p in ports.items():
        assert st.bf[L.B_PORT_AREA, j] == p["port_area"] and st.bf[L.B_PORT_AREA_T, j] == p["port_area_target"], j
        assert st.bf[L.B_PORT_AREA_S, j] == 0.0
    assert st.si[L.SI_NOISE_N] == len(ini["noise"])
    for j, (src, fwd, bwd, gain, lp, hp) in enumerate(ini["noise"]):
        assert st.noise_i[j].tolist() == [src, fwd, bwd]
        assert st.noise_f[j, L.Z_GAIN] == gain
        assert rel(st.noise_f[j, L.Z_LP_COEF], lp, 1e-300) <= 1e-15 and rel(st.noise_f[j, L.Z_HP_COEF], hp, 1e-300) <= 1e-15
        assert st.noise_f[j, L.Z_HP] == st.noise_f[j, L.Z_LP1] == st.noise_f[j, L.Z_LP2] == 0.0
    assert st.si[L.SI_NET_RNG] == ini["net_rng_state"]
    assert_exact(st.glot[:17].tolist(), ini["glottis_params"], "glottis params")
    gs = glot_mod.glottis_state(st.glot, st.glot_i)
    for key, want in ini["glottis_state"].items():
        got = gs[key]
        if isinstance(want, list):
            assert_t2(got, want, f"glottis {key}", tol=1e-15, floor=1e-300)
        elif isinstance(want, int) and not isinstance(want, bool):
            assert got == want, key
        else:
            assert rel(float(got), float(want), 1e-300) <= 1e-15, (key, got, want)
    vo = ini["voice"]
    assert st.si[L.SI_CTRL_EVERY] == vo["ctrl_every"] and st.si[L.SI_CTRL_COUNTER] == vo["ctrl_counter"]
    assert st.si[L.SI_LIM_HOLD_N] == vo["lim_hold_n"] and st.si[L.SI_LIM_HOLD] == 0
    for key, field in (("output_gain", L.S_OUTPUT_GAIN), ("dc_coef", L.S_DC_COEF), ("lim_release", L.S_LIM_RELEASE),
                       ("p_lung_now", L.S_P_LUNG_NOW), ("p_lung_step", L.S_P_LUNG_STEP)):
        assert rel(float(st.sc[field]), vo[key], 1e-300) <= 1e-15, (key, st.sc[field], vo[key])
    assert st.sc[L.S_DC_X] == st.sc[L.S_DC_Y] == st.sc[L.S_LIM_ENV] == 0.0
    dec = ini["decimator"]
    assert st.si[L.SI_M] == dec["factor"] and st.si[L.SI_DEC_POS] == dec["pos"] and st.si[L.SI_DEC_COUNT] == dec["count"]
    assert st.dec_buf.shape == (dec["buf_len"],) and not st.dec_buf.any()
    assert st.dec_c.shape == (dec["buf_len"] // 2,)


def test_fossa_and_sinus_rounding_rule():
    """The piriform fossa hangs off pharynx node min(n_ph-1, js_round(2.6/8.8·n_ph)) and the sinus off nasal node
    js_round(0.45·n_na) (human_airway.rs:111,115): checked on every V2 body through the connect/port branches."""
    for v in V2["bodies"]:
        st, idx = network.human_net(_body(v), FS, Seeds())
        n_ph = v["n_ph"]
        fossa_first = next(t["first"] for t in v["tubes"] if t["name"] == "piriform")
        ph_first = next(t["first"] for t in v["tubes"] if t["name"] == "pharynx")
        x = (2.6 / 8.8) * n_ph
        want = ph_first + min(n_ph - 1, int(math.floor(x) + (1 if x - math.floor(x) >= 0.5 else 0)))
        js = [(a, b) for a, b in zip(st.bi[L.BI_A].tolist(), st.bi[L.BI_B].tolist()) if b == fossa_first]
        assert js == [(want, fossa_first)], (v["label"], js, want)
