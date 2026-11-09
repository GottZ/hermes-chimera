"""Acoustic kernel, numpy form: the fallback that runs wherever numpy runs.

Same three entry points and the same semantics as ``kernel_numba`` on the same state arrays. Within a control block
everything that depends only on geometry (area ramps, section terms, the moving-wall volume source, branch
inertances, resistances and loss coefficients, compliances, radiation constants) is computed for all samples at once;
the per-sample loop keeps only what depends on the acoustic state, as numpy operations over nodes and branches. All
element-wise arithmetic and every ordered sum follow the scalar kernel's operation order.

Ported from chimera-vocal-synth crates/chimera/src/airway/network.rs, sources/glottis.rs, airway/decimator.rs and
voice/human_voice.rs (MIT, (c) 2026 Jameson Hodge).
"""

from __future__ import annotations

import math
import weakref
from typing import Callable, Literal, NamedTuple

import numpy as np

from . import kernel_numba as _k
from . import layout as L

RHO, MU, C_SOUND, RHO_C2, NU, RE_CRIT, PI = _k.RHO, _k.MU, _k.C_SOUND, _k.RHO_C2, _k.NU, _k.RE_CRIT, _k.PI
_SQRT_PI = math.sqrt(PI)


class KernelImpl(NamedTuple):
    """Plain container for one kernel's entry points; never passed into compiled code."""

    name: Literal["numba", "numpy"]
    run_samples: Callable
    output_chain: Callable
    flush_subnormals: Callable


# ------------------------------------------------------------------------------------------------ topology cache

class _Topo:
    """Index arrays derived from the (constant) network structure of one state."""

    def __init__(self, ni, nbr_start, nbr, bi, ri, si):
        n = ni.shape[1]
        nb = bi.shape[1]
        self.n, self.nb = n, nb
        self.A = bi[L.BI_A].copy()
        self.B = bi[L.BI_B].copy()
        kind = bi[L.BI_KIND]
        self.gp = int(si[L.SI_GLOTTIS_PORT])
        self.series = np.flatnonzero(kind == L.KIND_SERIES)
        self.ports = np.array([j for j in np.flatnonzero(kind == L.KIND_PORT) if j != self.gp], dtype=np.int64)
        self.port_mask = kind == L.KIND_PORT
        self.is_j = ni[L.NI_JUNCTION] != 0
        self.fixed = ni[L.NI_FIXED] != 0
        self.rad_of = ni[L.NI_RAD_OF].copy()
        self.rad_nodes = [int(k) for k in np.flatnonzero(self.rad_of >= 0)]       # node order (= update order)
        self.wall_nodes = np.flatnonzero(ni[L.NI_HAS_WALL] != 0)
        self.nbr = [list(nbr[nbr_start[k]:nbr_start[k + 1]]) for k in range(n)]
        # flow events per node in branch order: (branch, -1) where the node is a, (branch, +1) where it is b
        ev = [[] for _ in range(n)]
        for j in range(nb):
            ev[int(self.A[j])].append((j, -1.0))
            ev[int(self.B[j])].append((j, 1.0))
        d = max(len(e) for e in ev)
        self.ev_br = np.full((d, n), nb, dtype=np.int64)          # nb = a zero slot appended to u
        self.ev_sg = np.ones((d, n))
        for k, e in enumerate(ev):
            for s, (j, sg) in enumerate(e):
                self.ev_br[s, k] = j
                self.ev_sg[s, k] = sg
        # junction compliance floor: branches with a junction at either end, per junction node in branch order
        self.junctions = np.flatnonzero(self.is_j)
        self.junction_branches = {int(k): [j for j in range(nb) if (self.A[j] == k or self.B[j] == k)]
                                  for k in self.junctions}
        self.jbranch = np.array([bool(self.is_j[self.A[j]] or self.is_j[self.B[j]]) for j in range(nb)])
        # every node an open junction branch touches accumulates 1/l (only junction nodes use it)
        self.lsum_branches = {k: [j for j in range(nb) if self.jbranch[j] and (self.A[j] == k or self.B[j] == k)]
                              for k in range(n)}
        self.lsum_branches = {k: v for k, v in self.lsum_branches.items() if v}
        for k, brs in self.junction_branches.items():
            assert self.gp not in brs, "the glottis port must not touch a junction node"
        self.interior = ~self.fixed & (self.rad_of < 0)


_TOPO: dict[int, tuple] = {}


def _topo(ni, nbr_start, nbr, bi, ri, si) -> _Topo:
    key = id(bi)
    hit = _TOPO.get(key)
    if hit is not None and hit[0]() is bi:
        return hit[1]
    t = _Topo(ni, nbr_start, nbr, bi, ri, si)
    _TOPO[key] = (weakref.ref(bi), t)
    if len(_TOPO) > 64:
        for k2 in [k2 for k2, (r, _) in _TOPO.items() if r() is None]:
            del _TOPO[k2]
    return t


# ------------------------------------------------------------------------------------------------ helpers

def _ramp(x0, step, n_steps, clamp):
    """Values after each of n_steps repeated additions x += step (sequential, as the scalar loop); with clamp, a
    value below 0 becomes 0 and stays there (it can only go below 0 with a negative step)."""
    m = np.empty((n_steps + 1,) + x0.shape)
    m[0] = x0
    m[1:] = step
    acc = np.add.accumulate(m, axis=0)[1:]
    if clamp:
        neg = acc < 0.0
        if neg.any():
            acc[np.logical_or.accumulate(neg, axis=0)] = 0.0
    return acc


def _ffill(values, is_set, initial):
    """Per column: the last value where is_set, else initial (rows = time). Returns the value AFTER each row."""
    rows = np.arange(values.shape[0])[:, None]
    idx = np.where(is_set, rows, -1)
    idx = np.maximum.accumulate(idx, axis=0)
    cols = np.arange(values.shape[1])[None, :]
    return np.where(idx >= 0, values[np.maximum(idx, 0), cols], initial[None, :])


def glottis_block(glot, glot_i, dp, area, f0_now, voicing):
    """Same contract as kernel_numba.glottis_block (one glottis step per dp value)."""
    g = glot.tolist()
    gi = glot_i.tolist()
    nxt = _k.glottis_next
    for i in range(dp.shape[0]):
        area[i] = nxt(g, gi, float(dp[i]))
        f0_now[i] = g[L.G_F0_NOW]
        voicing[i] = g[L.G_VOICING]
    glot[:] = g
    glot_i[:] = gi


def flush_subnormals(nf, bf, rf, noise_f, glot, sc):
    """|x| < FLUSH_BELOW -> 0.0 in every state row that decays."""
    lim = L.FLUSH_BELOW
    for r in L.FLUSH_NODE_ROWS:
        row = nf[r]
        row[np.abs(row) < lim] = 0.0
    for r in L.FLUSH_BRANCH_ROWS:
        row = bf[r]
        row[np.abs(row) < lim] = 0.0
    for r in L.FLUSH_RAD_ROWS:
        row = rf[r]
        row[np.abs(row) < lim] = 0.0
    for c in L.FLUSH_NOISE_COLS:
        col = noise_f[:, c]
        noise_f[np.abs(col) < lim, c] = 0.0
    for i in L.FLUSH_GLOT:
        if abs(glot[i]) < lim:
            glot[i] = 0.0
    for i in L.FLUSH_SC:
        if abs(sc[i]) < lim:
            sc[i] = 0.0


# ------------------------------------------------------------------------------------------------ control step

def _control_step(tp, nf, ni, bf, bi, rf, noise_f, noise_i, glot, sc, si, ph_nodes, or_nodes, t_area, t_min, t_len,
                  t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung, row):
    """What control() leaves in the acoustic state for one track row (vectorised kernel_numba.control_step)."""
    n_ph = ph_nodes.shape[0]
    for nodes, off, length in ((ph_nodes, 0, t_len[row, 0]), (or_nodes, n_ph, t_len[row, 1])):
        cnt = nodes.shape[0]
        a = np.maximum(t_area[row, off:off + cnt], 0.0)          # js_max(0, a): no NaN in a track
        nf[L.N_AREA_T, nodes] = a
        nf[L.N_AMIN_T, nodes] = np.maximum(np.minimum(a, t_min[row, off:off + cnt]), 0.0)
        nf[L.N_DX_T, nodes] = length / cnt
    bf[L.B_PORT_AREA_T, si[L.SI_VELUM_PORT]] = max(0.0, float(t_velum[row]))
    _k._set_noise_sources(noise_f, noise_i, sc, si, int(t_noise_n[row]), t_noise_idx[row], t_noise_par[row],
                          t_noise_inherit[row])
    every = int(si[L.SI_CTRL_EVERY])
    inv = 1.0 / max(1.0, float(every))
    nf[L.N_AREA_S] = (nf[L.N_AREA_T] - nf[L.N_AREA]) * inv
    nf[L.N_AMIN_S] = (nf[L.N_AMIN_T] - nf[L.N_AMIN]) * inv
    nf[L.N_DX_S] = (nf[L.N_DX_T] - nf[L.N_DX]) * inv
    p = nf[L.N_P]
    max_p = float(np.max(np.abs(p[~tp.fixed]))) if (~tp.fixed).any() else 0.0
    if max_p < 1e-9:
        if float(np.max(np.abs(bf[L.B_U]))) < 1e-9:
            p[~tp.fixed] = 0.0
            nf[L.N_WALL_V] = 0.0
            wy = nf[L.N_WALL_Y]
            wy[np.abs(wy) < 1e-12] = 0.0
            bf[L.B_U] = 0.0
            bf[L.B_MOM] = 0.0
            rf[L.R_UL] = 0.0
    pm = tp.port_mask
    bf[L.B_PORT_AREA_S, pm] = (bf[L.B_PORT_AREA_T, pm] - bf[L.B_PORT_AREA, pm]) * inv
    si[L.SI_CTRL_COUNTER] = every
    sc[L.S_P_LUNG_STEP] = (float(t_plung[row]) - sc[L.S_P_LUNG_NOW]) / float(every)
    glot[: L.GLOTTIS_NPARAMS] = t_glot[row]


# ------------------------------------------------------------------------------------------------ one segment

def _segment(tp: _Topo, nf, ni, bf, bi, rf, noise_f, noise_i, glot, glot_i, sc, si, n_s, raw, i0):
    """n_s internal samples inside one control block (no control step inside)."""
    n, nb = tp.n, tp.nb
    t = float(sc[L.S_T])
    inv_t = float(sc[L.S_INV_T])
    min_a = float(sc[L.S_MIN_NODE_AREA])
    sq = float(sc[L.S_SQUEEZE_AREA])
    closed = float(sc[L.S_CLOSED_AREA])
    flags = int(si[L.SI_FLAGS])
    vol_on = (flags & L.F_VOLUME_SOURCE) != 0
    energy_safe = (flags & L.F_ENERGY_SAFE) != 0
    symmetric = (flags & L.F_SYMMETRIC_KINETIC) != 0
    junction_floor = (flags & L.F_JUNCTION_FLOOR) != 0
    half_rho = 0.5 * RHO
    visc = math.sqrt((RHO * MU * float(sc[L.S_VISCOUS_OMEGA])) / 2.0) * float(sc[L.S_VISCOUS_SCALE])

    # ---- geometry trajectories (row j = values used by sample j)
    area0 = nf[L.N_AREA].copy()
    amin0 = nf[L.N_AMIN].copy()
    AR = _ramp(area0, nf[L.N_AREA_S], n_s, True)
    AM = _ramp(amin0, nf[L.N_AMIN_S], n_s, True)
    bad = AM > AR
    if bad.any():                                    # the minimum caught up with the area: exact recurrence there
        steps = nf[L.N_AMIN_S]
        for k in np.flatnonzero(bad.any(axis=0)):
            x = amin0[k]
            for j in range(n_s):
                am = x + steps[k]
                if am < 0.0:
                    am = 0.0
                x = AR[j, k] if am > AR[j, k] else am
                AM[j, k] = x
    DX = _ramp(nf[L.N_DX].copy(), nf[L.N_DX_S], n_s, False)
    # max/min below equal the scalar kernel's compare-and-select for finite operands
    IA = 1.0 / np.maximum(AR, 1e-6)
    IAM = 1.0 / np.maximum(AM, 1e-6)
    SA = np.sqrt(AR)
    HL = DX * IA
    HR = nf[L.N_WALL_PERIM] * DX * SA * IA * IA * nf[L.N_VISC_F]
    V = np.maximum(AR, min_a) * DX
    VP = np.empty_like(V)
    VP[0] = nf[L.N_VOL]
    VP[1:] = V[:-1]

    # ---- moving-wall volume source: raw -dV/dt, split near closures (scalar loop order kept per node)
    if vol_on:
        RAW = -(V - VP) * inv_t
        AE = np.minimum(AM, AR)
        SQ = (AE < sq) & (RAW != 0.0)
        if SQ.any():
            TT = AE / sq
            KEEP = TT * TT * (3.0 - 2.0 * TT)
            SPILL = np.where(SQ, RAW * KEEP, RAW) + 0.0
            SPILL = _spill_shares(tp, SPILL, SQ, RAW, KEEP, AR, AM, area0, amin0, nf[L.N_AMIN_T], sq, n_s)
        else:
            SPILL = RAW + 0.0
    else:
        SPILL = None

    # ---- branches: geometry-only coefficients (series formula everywhere, then the port columns; the glottis
    #      port is filled per sample)
    Aa, Bb = tp.A, tp.B
    ma, mb = AM[:, Aa], AM[:, Bb]
    AEFF = np.minimum(ma, mb)
    CL = (np.minimum(AR[:, Aa], AR[:, Bb]) < closed) | (AEFF < closed)
    ia, ib, ima, imb = IA[:, Aa], IA[:, Bb], IAM[:, Aa], IAM[:, Bb]
    LB = half_rho * (HL[:, Aa] + HL[:, Bb])
    RB = 0.5 * visc * (HR[:, Aa] + HR[:, Bb])
    ef = np.maximum(ima - ib, 0.0)
    eb = np.maximum(imb - ia, 0.0)
    cf = np.maximum(imb - ia, 0.0)
    cb = np.maximum(ima - ib, 0.0)
    KF = half_rho * ef * ef + 0.1 * half_rho * cf * cf
    KB = half_rho * eb * eb + 0.1 * half_rho * cb * cb
    pt = tp.ports
    PA_END = None
    if pt.size:
        PA = _ramp(bf[L.B_PORT_AREA, pt].copy(), bf[L.B_PORT_AREA_S, pt], n_s, True)
        PA_END = PA[-1]
        a_, b_ = Aa[pt], Bb[pt]
        cl = PA < closed
        with np.errstate(divide="ignore", invalid="ignore"):
            ip = 1.0 / PA
            ln = bf[L.B_PORT_LEN, pt]
            ll = RHO * ln * ip
            half = bi[L.BI_PORT_HALF, pt] != 0
            ll = np.where(half, ll + half_rho * (HL[:, a_] + HL[:, b_]), ll)
            w = bf[L.B_PORT_SLIT, pt]
            r_slit = (12.0 * MU * ln * w * w) * ip * ip * ip
            r_visc = visc * 2.0 * _SQRT_PI * ln * np.sqrt(PA) * ip * ip
            rr = np.where(w > 0.0, r_slit, r_visc) + bf[L.B_PORT_EXTRA_R, pt]
            ex_f = ip - IA[:, b_]
            ex_b = ip - IA[:, a_]
            kin = bf[L.B_PORT_KIN, pt] * half_rho * ip * ip
            kex = bf[L.B_PORT_KEX, pt]
            kf = kin + kex * half_rho * np.where(ex_f > 0.0, ex_f * ex_f, 0.0)
            kb = kin + kex * half_rho * np.where(ex_b > 0.0, ex_b * ex_b, 0.0)
        LB[:, pt] = np.where(cl, 0.0, ll)
        RB[:, pt] = np.where(cl, 0.0, rr)
        KF[:, pt] = np.where(cl, 0.0, kf)
        KB[:, pt] = np.where(cl, 0.0, kb)
        AEFF[:, pt] = PA
        CL[:, pt] = cl
    gp = tp.gp
    LB[:, gp] = 0.0
    RB[:, gp] = 0.0
    KF[:, gp] = 0.0
    KB[:, gp] = 0.0
    CL[:, gp] = True
    if symmetric:
        km = np.maximum(KF, KB)
        KF = km
        KB = km.copy()
    OPENF = (~CL).astype(np.float64)
    BQ = LB * inv_t + RB
    # l_prev before each sample (only open branches update it); glottis column handled per sample
    lp0 = bf[L.B_L_PREV].copy()
    not_g = np.arange(nb) != gp
    if CL[:, not_g].any():
        LPN = _ffill(LB, ~CL, lp0)                    # after each sample
    else:
        LPN = LB.copy()
    LPN[:, gp] = lp0[gp]
    LP = np.empty_like(LPN)
    LP[0] = lp0
    LP[1:] = LPN[:-1]
    # momentum carried into this sample: mom = l_prev·u (both from the last open sample, 0 when it was closed),
    # so "l·u if l < l_prev else mom" is min(l, l_prev)·u (energy-safe), else l_prev·u
    LMIN = np.minimum(LB, LP) if energy_safe else LP.copy()

    # ---- nodes: compliance with the junction floor, sealed cells
    C = V / RHO_C2
    INV_LSUM = np.zeros((n_s, n))
    il_all = OPENF / (LB + (1.0 - OPENF))             # 1/l on open branches, 0 on closed ones
    for k, brs in tp.lsum_branches.items():
        acc = np.zeros(n_s)
        for j in brs:
            acc = acc + il_all[:, j]
        INV_LSUM[:, k] = acc
    if junction_floor and tp.junctions.size:
        t2 = t * t
        jn = tp.junctions
        cmin = (t2 * INV_LSUM[:, jn]) / 2.4
        C[:, jn] = np.maximum(C[:, jn], cmin)
    SEALED = (AR < closed) | (AM < closed)
    OPENINT = tp.interior[None, :] & ~SEALED
    TC = t / C

    # ---- radiators
    rad = []
    for k in tp.rad_nodes:
        r = int(tp.rad_of[k])
        a = np.where(AR[:, k] > 1e-4, AR[:, k], 1e-4)
        lr = (8.0 * RHO) / (3.0 * PI * np.sqrt(PI * a))
        rres = (128.0 * RHO * C_SOUND) / (9.0 * PI * PI * a)
        h = t / (2.0 * lr)
        g = 0.5 / rres
        cc = C[:, k]
        num = cc * inv_t - 0.5 * h - g
        den = cc * inv_t + 0.5 * h + g
        open_ = ~SEALED[:, k]
        lrp0 = np.array([float(rf[L.R_L_PREV, r])])
        lrn = _ffill(lr[:, None], open_[:, None], lrp0)[:, 0]
        lrp = np.empty(n_s)
        lrp[0] = lrp0[0]
        lrp[1:] = lrn[:-1]
        scale = (lrp > 0.0) & (lr > lrp)
        fac = np.where(scale, lrp / np.where(scale, lr, 1.0), 1.0)
        rad.append((k, r, open_.tolist(), scale.tolist(), fac.tolist(), h.tolist(), g.tolist(), num.tolist(),
                    den.tolist(), lrn[-1]))

    rad_by_node = {k: rd[1:] for rd in rad for k in [rd[0]]}
    rad_only = list(tp.rad_nodes)
    sealed_int = SEALED & tp.interior[None, :]
    any_sealed_int = sealed_int.any(axis=1).tolist()
    special_rows = [sorted(rad_only + np.flatnonzero(sealed_int[j]).tolist()) if any_sealed_int[j] else rad_only
                    for j in range(n_s)]

    # ---- walls over all nodes; non-wall nodes get zero coefficients (their local values never leave here)
    has_w = ni[L.NI_HAS_WALL] != 0
    hw = has_w.astype(np.float64)
    wn = tp.wall_nodes
    m_w = nf[L.N_WALL_M]
    r_w = nf[L.N_WALL_R]
    A_w = (m_w * inv_t - 0.5 * r_w) * hw
    D_w = np.where(has_w, m_w * inv_t + 0.5 * r_w, 1.0)
    K_w = nf[L.N_WALL_K] * hw
    W = (nf[L.N_WALL_PERIM] * np.sqrt(np.maximum(AR, min_a)) * DX) * hw
    wrad = nf[L.N_WALL_RAD]
    rad_w = np.flatnonzero(has_w & (wrad != 0.0))
    if rad_w.size and rad_w[-1] - rad_w[0] + 1 == rad_w.size:
        rad_w = slice(int(rad_w[0]), int(rad_w[-1]) + 1)
    wrad_r = wrad[rad_w]
    any_rad_w = wrad_r.size > 0

    # ---- branch solve terms; a closed branch gets neutral terms and a zero factor (its flow and momentum are 0)
    np.copyto(BQ, 1.0, where=CL)
    BQ2 = BQ * BQ
    KF4 = (4.0 * KF) * OPENF
    KB4 = (4.0 * KB) * OPENF
    KFP = KF4 > 0.0
    KBP = KB4 > 0.0
    TCO = TC * OPENINT                        # p += (t/c)·flow_sum on open interior cells, + 0.0 elsewhere

    # ---- per-sample loop (state-dependent part)
    p = nf[L.N_P]
    u_ext = np.zeros(nb + 1)                  # branch flows + one zero slot for the flow-event gather
    u_ext[:nb] = bf[L.B_U]
    u = u_ext[:nb]
    psrc = np.zeros(nb)
    wv = nf[L.N_WALL_V].copy()
    wy = nf[L.N_WALL_Y].copy()
    vs1 = nf[L.N_VOL_SRC1]
    vs2 = nf[L.N_VOL_SRC2]
    vlc = float(sc[L.S_VOL_LP_COEF])
    ev_br, ev_sg = tp.ev_br, tp.ev_sg
    gp = tp.gp
    ga, gb = int(Aa[gp]), int(Bb[gp])
    g_len = float(bf[L.B_PORT_LEN, gp])
    g_half = bi[L.BI_PORT_HALF, gp] != 0
    g_slit = float(bf[L.B_PORT_SLIT, gp])
    g_extra = float(bf[L.B_PORT_EXTRA_R, gp])
    g_kin = float(bf[L.B_PORT_KIN, gp])
    g_kex = float(bf[L.B_PORT_KEX, gp])
    g_lp = float(bf[L.B_L_PREV, gp])
    g_ap = float(bf[L.B_PORT_AREA, gp])
    g_ap_prev = float(bf[L.B_AREA, gp])
    HLga = HL[:, ga].tolist()
    HLgb = HL[:, gb].tolist()
    IAga = IA[:, ga].tolist()
    IAgb = IA[:, gb].tolist()
    lungs, sub, supra = int(si[L.SI_LUNGS]), int(si[L.SI_SUB_NODE]), int(si[L.SI_SUPRA_NODE])
    p_now = float(sc[L.S_P_LUNG_NOW])
    p_step = float(sc[L.S_P_LUNG_STEP])
    gl = glot.tolist()
    gli = glot_i.tolist()
    gnext = _k.glottis_next
    n_noise = int(si[L.SI_NOISE_N])
    nz = []
    for i in range(n_noise):
        sb = int(noise_i[i, L.ZI_SRC])
        col = None if sb == gp else AEFF[:, sb].tolist()
        nz.append([sb, int(noise_i[i, L.ZI_FWD]), int(noise_i[i, L.ZI_BWD]), float(noise_f[i, L.Z_GAIN]),
                   float(noise_f[i, L.Z_HP]), float(noise_f[i, L.Z_LP1]), float(noise_f[i, L.Z_LP2]),
                   float(noise_f[i, L.Z_LP_COEF]), float(noise_f[i, L.Z_HP_COEF]), float(noise_f[i, L.Z_LEVEL]),
                   col, float(bf[L.B_AREA, sb])])
    noise_scale = float(sc[L.S_NOISE_SCALE]) * math.sqrt(96000.0 / float(sc[L.S_FS]))
    sat = 12000.0 * 12000.0 - RE_CRIT * RE_CRIT
    rng = int(si[L.SI_NET_RNG])
    rng_next = _k.nbm.rng_next
    prev_total = float(sc[L.S_RAD_PREV_TOTAL])
    rul = rf[L.R_UL].tolist()
    ru = rf[L.R_U].tolist()
    touched: list[int] = []
    fs = np.zeros(n)
    uw = np.zeros(n)
    y = float(sc[L.S_OUT])
    OFr, LBr, BQr, BQ2r, KF4r, KB4r, KFPr, KBPr, LMr = OPENF, LB, BQ, BQ2, KF4, KB4, KFP, KBP, LMIN
    err = np.seterr(divide="ignore", invalid="ignore")
    try:
        for j in range(n_s):
            # lungs and glottis
            p_now += p_step
            p[lungs] = p_now
            a = gnext(gl, gli, float(p[sub]) - float(p[supra]))
            g_ap = 0.0 if a < 0.0 else a
            # moving-wall volume source -> flow_sum base
            if vol_on:
                vs1 += vlc * (SPILL[j] - vs1)
                vs2 += vlc * (vs1 - vs2)
                np.add(vs2, 0.0, out=fs)
            else:
                fs[:] = 0.0
            # turbulence noise (previous sample's flow and branch area)
            for i in touched:
                psrc[i] = 0.0
            touched = []
            for src in nz:
                s_b = src[0]
                if src[10] is None:
                    ab_ = g_ap_prev
                else:
                    ab_ = src[10][j - 1] if j > 0 else src[11]
                uu = float(u[s_b])
                amp = 0.0
                if ab_ > closed:
                    re = (abs(uu) * 2.0) / (NU * math.sqrt(PI * ab_))
                    if re > RE_CRIT:
                        x = re * re - RE_CRIT * RE_CRIT
                        amp = (src[3] * noise_scale * x) / (1.0 + x / sat)
                rng, rv = rng_next(rng)
                wn_ = (rv * 2.0 - 1.0) * 1.732
                hp = src[4]
                hp += (1.0 - src[8]) * (wn_ - hp)
                src[4] = hp
                hp_out = wn_ - hp
                lp1 = src[5]
                lp1 += src[7] * (hp_out - lp1)
                src[5] = lp1
                lp2 = src[6]
                lp2 += src[7] * (lp1 - lp2)
                src[6] = lp2
                src[9] = amp
                if amp == 0.0:
                    continue
                v_ = amp * lp2
                if uu >= 0.0:
                    psrc[src[1]] += v_
                    touched.append(src[1])
                else:
                    psrc[src[2]] -= v_
                    touched.append(src[2])
            # glottis port: coefficients from its area of this sample
            if g_ap < closed:
                OFr[j, gp] = 0.0
                LBr[j, gp] = 0.0
                BQr[j, gp] = 1.0
                BQ2r[j, gp] = 1.0
                KF4r[j, gp] = 0.0
                KB4r[j, gp] = 0.0
                KFPr[j, gp] = False
                KBPr[j, gp] = False
                LMr[j, gp] = 0.0
            else:
                ip = 1.0 / g_ap
                ll = RHO * g_len * ip
                if g_half:
                    ll += half_rho * (HLga[j] + HLgb[j])
                if g_slit > 0.0:
                    rr = (12.0 * MU * g_len * g_slit * g_slit) * ip * ip * ip
                else:
                    rr = visc * 2.0 * _SQRT_PI * g_len * math.sqrt(g_ap) * ip * ip
                rr += g_extra
                ex_f = ip - IAgb[j]
                ex_b = ip - IAga[j]
                kin = g_kin * half_rho * ip * ip
                kf_ = kin + g_kex * half_rho * (ex_f * ex_f if ex_f > 0.0 else 0.0)
                kb_ = kin + g_kex * half_rho * (ex_b * ex_b if ex_b > 0.0 else 0.0)
                if symmetric:
                    km_ = kf_ if kf_ > kb_ else kb_
                    kf_ = kb_ = km_
                bq_ = ll * inv_t + rr
                OFr[j, gp] = 1.0
                LBr[j, gp] = ll
                BQr[j, gp] = bq_
                BQ2r[j, gp] = bq_ * bq_
                KF4r[j, gp] = 4.0 * kf_
                KB4r[j, gp] = 4.0 * kb_
                KFPr[j, gp] = kf_ > 0.0
                KBPr[j, gp] = kb_ > 0.0
                LMr[j, gp] = ll if (energy_safe and ll < g_lp) else g_lp
                g_lp = ll
            g_ap_prev = g_ap
            # branch flows (exact quadratic, scalar kernel's operation order)
            bq = BQr[j]
            c = p[Aa] - p[Bb]
            if touched:
                c += psrc
            c += (LMr[j] * u) * inv_t
            pos = c >= 0.0
            uq = (2.0 * c) / (bq + np.sqrt(BQ2r[j] + np.where(pos, KF4r[j], KB4r[j]) * np.abs(c)))
            np.multiply(np.where(np.where(pos, KFPr[j], KBPr[j]), uq, c / bq), OFr[j], out=u)
            # flow sums: each node's events in branch order
            G = u_ext[ev_br]
            G *= ev_sg
            for sl in range(G.shape[0]):
                fs += G[sl]
            # walls
            v = (A_w * wv + p - K_w * wy) / D_w
            wv = v
            wy += t * v
            np.multiply(W[j], v, out=uw)
            fs -= uw
            wall_rad_sum = float(np.add.accumulate(uw[rad_w] * wrad_r)[-1]) if any_rad_w else 0.0
            # nodes: open interior cells at once, then sealed cells and radiators in node order
            sealed_now = any_sealed_int[j]
            p_old = p.copy() if sealed_now else p
            p += TCO[j] * fs
            rad_total = wall_rad_sum
            for k in (special_rows[j] if sealed_now else rad_only):
                rd = rad_by_node.get(k)
                if rd is None:                              # sealed interior cell: follow open neighbours
                    ssum = 0.0
                    cnt = 0
                    for m in tp.nbr[k]:
                        if AR[j, m] >= closed and AM[j, m] >= closed:
                            ssum += float(p[m]) if m < k else float(p_old[m])
                            cnt += 1
                    if cnt > 0:
                        po = float(p_old[k])
                        p[k] = po + 0.02 * (ssum / cnt - po)
                    continue
                r, open_, scale, fac, h, g, num, den, _lrn = rd
                if not open_[j]:
                    p[k] = float(p_old[k]) * 0.98
                    ul_ = rul[r] * 0.995
                    rul[r] = ul_
                    ru[r] = ul_
                    rad_total += ul_
                    continue
                po = float(p_old[k])
                ul_prev = rul[r]
                if scale[j]:
                    ul_prev *= fac[j]
                pn = (po * num[j] + float(fs[k]) - ul_prev) / den[j]
                p[k] = pn
                ul_ = ul_prev + h[j] * (po + pn)
                rul[r] = ul_
                ur = 0.5 * (ul_prev + ul_) + g[j] * (pn + po)
                ru[r] = ur
                rad_total += ur
            y = (rad_total - prev_total) * inv_t
            prev_total = rad_total
            raw[i0 + j] = y
    finally:
        np.seterr(**err)

    # ---- write back the state of the last sample
    nf[L.N_AREA] = AR[-1]
    nf[L.N_AMIN] = AM[-1]
    nf[L.N_DX] = DX[-1]
    nf[L.N_INV_A] = IA[-1]
    nf[L.N_INV_AMIN] = IAM[-1]
    nf[L.N_SQRT_A] = SA[-1]
    nf[L.N_HALF_L] = HL[-1]
    nf[L.N_HALF_R] = HR[-1]
    nf[L.N_VOL] = V[-1]
    nf[L.N_INV_LSUM] = INV_LSUM[-1]
    nf[L.N_FLOW_SUM] = fs
    nf[L.N_EXT_FLOW] = 0.0
    if vol_on:
        nf[L.N_SPILL] = SPILL[-1]
    nf[L.N_WALL_V, wn] = wv[wn]
    nf[L.N_WALL_Y, wn] = wy[wn]
    nf[L.N_WALL_U, wn] = uw[wn]
    bf[L.B_U] = u
    bf[L.B_MOM] = LB[-1] * u
    bf[L.B_PSRC] = psrc
    lpn = LPN[-1].copy()
    lpn[gp] = g_lp
    bf[L.B_L_PREV] = lpn
    barea = AEFF[-1].copy()
    barea[gp] = g_ap
    bf[L.B_AREA] = barea
    if PA_END is not None:
        bf[L.B_PORT_AREA, pt] = PA_END
    bf[L.B_PORT_AREA, gp] = g_ap
    bf[L.B_PORT_AREA_T, gp] = g_ap
    bf[L.B_PORT_AREA_S, gp] = 0.0
    rf[L.R_UL] = rul
    rf[L.R_U] = ru
    for rd in rad:
        rf[L.R_L_PREV, rd[1]] = rd[-1]
    for i, src in enumerate(nz):
        noise_f[i, L.Z_HP], noise_f[i, L.Z_LP1], noise_f[i, L.Z_LP2], noise_f[i, L.Z_LEVEL] = (
            src[4], src[5], src[6], src[9])
    si[L.SI_NET_RNG] = rng
    sc[L.S_P_LUNG_NOW] = p_now
    sc[L.S_RAD_PREV_TOTAL] = prev_total
    sc[L.S_OUT] = y
    glot[:] = gl
    glot_i[:] = gli


def _spill_shares(tp, SPILL, SQ, RAW, KEEP, AR, AM, area0, amin0, amin_t, sq, n_s):
    """Add the squeezed volume of nearly closed cells to their open neighbours, in the scalar loop's order: for each
    receiving node m the contributions are folded in node order (own share at m, others at their node index); a
    neighbour counts as open with its area of this sample if it comes earlier in the node loop, else of the last."""
    sqcols = np.flatnonzero(SQ.any(axis=0))
    ARp = np.empty_like(AR)
    ARp[0] = area0
    ARp[1:] = AR[:-1]
    AMp = np.empty_like(AM)
    AMp[0] = amin0
    AMp[1:] = AM[:-1]
    share = {}
    for k in sqcols:
        k = int(k)
        nb_ = tp.nbr[k]
        opens = []
        for m in nb_:
            if m < k:
                o = (np.minimum(AR[:, m], AM[:, m]) >= sq) | (amin_t[m] >= sq)
            else:
                o = (np.minimum(ARp[:, m], AMp[:, m]) >= sq) | (amin_t[m] >= sq)
            opens.append(o)
        n_open = np.zeros(n_s)
        for o in opens:
            n_open = n_open + o
        sp = RAW[:, k] * (1.0 - KEEP[:, k])
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = sp / n_open
        for m, o in zip(nb_, opens):
            share.setdefault(int(m), []).append((k, np.where(SQ[:, k] & o & (n_open > 0), sh, 0.0)))
    out = SPILL.copy()
    for m, lst in share.items():
        own = SPILL[:, m]
        terms = sorted(lst + [(m, own)], key=lambda x: x[0])
        acc = np.zeros(n_s)
        for _, v in terms:
            acc = acc + v
        out[:, m] = acc
    return out


# ------------------------------------------------------------------------------------------------ entry points

def run_samples(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, glot, glot_i, sc, si,
                ph_nodes, or_nodes,
                t_area, t_min, t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung,
                n_internal, raw):
    """Same contract as kernel_numba.run_samples."""
    tp = _topo(ni, nbr_start, nbr, bi, ri, si)
    n_rows = t_area.shape[0]
    i = 0
    while i < n_internal:
        if si[L.SI_CTRL_COUNTER] <= 0:
            row = int(si[L.SI_BLOCK_POS])
            if row >= n_rows:
                si[L.SI_ERROR] = L.ERR_TRACK_EXHAUSTED
                return
            if (int(si[L.SI_FLAGS]) & L.F_FLUSH_SUBNORMALS) != 0:
                flush_subnormals(nf, bf, rf, noise_f, glot, sc)
            _control_step(tp, nf, ni, bf, bi, rf, noise_f, noise_i, glot, sc, si, ph_nodes, or_nodes, t_area, t_min,
                          t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung, row)
            si[L.SI_BLOCK_POS] = row + 1
            si[L.SI_BLOCKS_DONE] += 1
        n_s = int(min(int(si[L.SI_CTRL_COUNTER]), n_internal - i))
        _segment(tp, nf, ni, bf, bi, rf, noise_f, noise_i, glot, glot_i, sc, si, n_s, raw, i)
        si[L.SI_CTRL_COUNTER] -= n_s
        i += n_s


def output_chain(sc, si, dec_c, dec_buf, raw, n_internal, out):
    """Same contract as kernel_numba.output_chain."""
    m = int(si[L.SI_M])
    nt = dec_c.shape[0]
    pos0 = int(si[L.SI_DEC_POS])
    cnt0 = int(si[L.SI_DEC_COUNT])
    n = int(n_internal)
    if n <= 0:
        return 0
    x = raw[:n]
    hist = np.concatenate((dec_buf[pos0:pos0 + nt], x))
    first = m - cnt0 - 1
    pushes = np.arange(first, n, m)
    count = pushes.shape[0]
    # ring buffer after the call (each slot keeps its last write)
    last = np.arange(max(0, n - nt), n)
    slots = (pos0 + last) % nt
    dec_buf[slots] = x[last]
    dec_buf[slots + nt] = x[last]
    si[L.SI_DEC_POS] = (pos0 + n) % nt
    si[L.SI_DEC_COUNT] = (cnt0 + n) % m
    if count == 0:
        return 0
    acc = np.zeros(count)
    base = pushes + 1
    for q in range(nt):
        acc += dec_c[nt - 1 - q] * hist[base + q]
    sc[L.S_DEC_VALUE] = acc[-1]
    gain = float(sc[L.S_OUTPUT_GAIN])
    dc_x = float(sc[L.S_DC_X])
    dc_y = float(sc[L.S_DC_Y])
    dc_c = float(sc[L.S_DC_COEF])
    env = float(sc[L.S_LIM_ENV])
    rel = float(sc[L.S_LIM_RELEASE])
    hold = int(si[L.SI_LIM_HOLD])
    hold_n = int(si[L.SI_LIM_HOLD_N])
    tanh = math.tanh
    res = [0.0] * count
    for idx, av_ in enumerate(acc.tolist()):
        v = av_ * gain
        hp = v - dc_x + dc_c * dc_y
        dc_x = v
        dc_y = hp
        v = hp
        av = abs(v)
        if av >= env:
            env = av
            hold = hold_n
        elif hold > 0:
            hold -= 1
        else:
            env *= rel
        if env > 0.75:
            v *= 0.75 / env
        al = abs(v)
        if al > 0.85:
            v = (1.0 if v > 0.0 else -1.0) * (0.85 + 0.15 * tanh((al - 0.85) / 0.15))
        res[idx] = v
    out[:count] = res
    sc[L.S_DC_X] = dc_x
    sc[L.S_DC_Y] = dc_y
    sc[L.S_LIM_ENV] = env
    si[L.SI_LIM_HOLD] = hold
    return count


KERNEL_NUMPY = KernelImpl("numpy", run_samples, output_chain, flush_subnormals)

__all__ = ["KernelImpl", "KERNEL_NUMPY", "run_samples", "output_chain", "flush_subnormals", "glottis_block"]
