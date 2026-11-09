"""Acoustic kernel, scalar form: the numba fast path and the CPython reference.

Ported from chimera-vocal-synth crates/chimera/src/airway/network.rs, sources/glottis.rs,
airway/decimator.rs and voice/human_voice.rs (MIT, (c) 2026 Jameson Hodge).

JIT file: the worker's loader executes it under the fixed module name ``_chimera_kernel_numba`` with
``CHIMERA_JIT = numba.njit(...)``; imported as part of the package, ``jit`` is the identity and every function
runs as plain CPython. Entry points (``run_samples``, ``output_chain``, ``flush_subnormals``) take only numpy
arrays and Python scalars. The state layout constants live here (numba reads module globals as compile-time
constants and a JIT file may not import the package); ``dsp/layout.py`` re-exports them for everything else.
"""

import math

import numpy as np

try:
    import _chimera_nbmath as nbm
except ImportError:
    from ..core import nbmath as nbm

jit = globals().get("CHIMERA_JIT", lambda f: f)

# ---------------------------------------------------------------- node rows: nf float64[NF, n_nodes]
N_P = 0             # pressure p (dyn/cm2)
N_AREA = 1          # area (cm2)
N_AREA_T = 2        # area_target
N_AREA_S = 3        # area_step
N_AMIN = 4          # amin (sub-section minimum)
N_AMIN_T = 5        # amin_target
N_AMIN_S = 6        # amin_step
N_DX = 7            # section length dx (cm)
N_DX_T = 8          # dx_target
N_DX_S = 9          # dx_step
N_VOL = 10          # vol = js_max(area, min_node_area) * dx
N_INV_A = 11        # 1 / js_max(area, 1e-6)
N_INV_AMIN = 12     # 1 / js_max(amin, 1e-6) (per sample)
N_INV_LSUM = 13     # junction compliance floor accumulator (per sample)
N_HALF_L = 14       # dx / A (per sample)
N_HALF_R = 15       # wall_perim * dx * sqrt(A) / A^2 * visc_f (per sample)
N_SQRT_A = 16       # sqrt(A) (per sample)
N_FLOW_SUM = 17     # net inflow accumulator (per sample)
N_EXT_FLOW = 18     # external injection (always 0 in the human voice)
N_SPILL = 19        # raw moving-wall injection accumulator (per sample)
N_VOL_SRC1 = 20     # moving-wall volume source, 1st low-pass state
N_VOL_SRC2 = 21     # 2nd low-pass state
N_WALL_M = 22       # wall mass per unit area
N_WALL_R = 23       # wall resistance
N_WALL_K = 24       # wall stiffness
N_WALL_PERIM = 25   # perimeter factor * 2 sqrt(pi)
N_WALL_RAD = 26     # radiated fraction of the wall volume velocity
N_WALL_Y = 27       # wall displacement
N_WALL_V = 28       # wall velocity
N_WALL_U = 29       # wall volume velocity (per sample)
N_VISC_F = 30       # viscous factor
NF = 31

# ---------------------------------------------------------------- node ints: ni int64[NI, n_nodes]
NI_JUNCTION = 0     # >= 3 connections
NI_FIXED = 1        # prescribed-pressure node (lungs)
NI_HAS_WALL = 2
NI_TUBE = 3         # tube index of the node
NI_RAD_OF = 4       # radiator index or -1
NI = 5

# ---------------------------------------------------------------- branch rows: bf float64[BF, n_branches]
B_U = 0             # volume velocity U
B_MOM = 1           # momentum L*U
B_L_PREV = 2        # inertance of the previous sample
B_PSRC = 3          # series pressure source (noise, per sample)
B_PORT_AREA = 4     # port area (current)
B_PORT_AREA_T = 5   # port area target
B_PORT_AREA_S = 6   # port area step
B_PORT_LEN = 7      # port duct length
B_PORT_KIN = 8      # entrance loss coefficient
B_PORT_KEX = 9      # exit loss coefficient
B_PORT_SLIT = 10    # slit width (glottis) or 0
B_PORT_EXTRA_R = 11  # extra series resistance
B_AREA = 12         # branch_area (effective area, Reynolds number)
BF = 13

# ---------------------------------------------------------------- branch ints: bi int64[BI, n_branches]
BI_A = 0
BI_B = 1
BI_KIND = 2         # KIND_SERIES / KIND_PORT
BI_PORT_HALF = 3    # port includes the adjacent half-section inertances
BI = 4
KIND_SERIES = 0
KIND_PORT = 1

# ---------------------------------------------------------------- radiators: rf float64[RF, n_rad], ri int64[n_rad]
R_UL = 0            # inductive branch flow of the radiation load
R_L_PREV = 1        # radiation inertance of the previous sample
R_U = 2             # radiated volume velocity
RF = 3

# ---------------------------------------------------------------- noise sources: noise_f float64[MAX_NOISE, NZ]
MAX_NOISE = 3
Z_GAIN = 0
Z_HP = 1            # high-pass state (holds a low-passed copy)
Z_LP1 = 2
Z_LP2 = 3
Z_LP_COEF = 4
Z_HP_COEF = 5
Z_LEVEL = 6         # last amplitude (telemetry)
NZ = 7
ZI_SRC = 0          # noise_i int64[MAX_NOISE, 3]
ZI_FWD = 1
ZI_BWD = 2

# ---------------------------------------------------------------- glottis: glot float64[G], glot_i int64[GI]
G_F0 = 0            # 0..16 = GlottisParams in field order (= core.glottis_params.G_*)
G_AMPLITUDE = 1
G_ABDUCTION = 2
G_CHINK = 3
G_OPEN_QUOTIENT = 4
G_SKEW = 5
G_CLOSURE_SHARPNESS = 6
G_EFFORT = 7
G_JITTER = 8
G_SHIMMER = 9
G_VIBRATO_DEPTH = 10
G_VIBRATO_RATE = 11
G_DRIFT = 12
G_FRY = 13
G_DIPLOPHONIA = 14
G_CLOSURE = 15
G_CHAOS = 16
GLOTTIS_NPARAMS = 17
G_PHASE = 17
G_CYCLE_AMP = 18
G_CYCLE_PERIOD_SCALE = 19
G_CHAOS_X = 20
G_VIB_PHASE = 21
G_DRIFT_VALUE = 22
G_PITCH_FACTOR = 23
G_VOICING = 24
G_VOICING_SM = 25
G_DP_SM = 26
G_DP_COEF = 27
G_VOICE_COEF = 28
G_AREA = 29
G_F0_NOW = 30
G_THRESHOLD = 31
G_STATIC_SM = 32
G_CLOSURE_SM = 33
G_AMP_SM = 34
G_SM_COEF = 35
G_FS = 36
G_DRIFT_Y = 37      # 37, 38, 39: Drift.y[0..3]
G_DRIFT_A = 40      # 40, 41, 42: Drift.a[0..3]
G = 43
GI_CYCLE_INDEX = 0
GI_DRIFT_COUNTER = 1
GI_RNG = 2          # glottis jitter rng state (u32 in int64)
GI_DRIFT_RNG = 3    # drift rng state
GI = 4

# ---------------------------------------------------------------- scalars: sc float64[S]
S_FS = 0            # internal rate
S_T = 1
S_INV_T = 2
S_CLOSED_AREA = 3
S_MIN_NODE_AREA = 4
S_SQUEEZE_AREA = 5
S_VISCOUS_SCALE = 6
S_VISCOUS_OMEGA = 7
S_NOISE_SCALE = 8
S_VOL_LP_COEF = 9
S_OUT = 10          # last radiated output of the network
S_RAD_PREV_TOTAL = 11
S_P_LUNG_NOW = 12
S_P_LUNG_STEP = 13
S_DC_X = 14
S_DC_Y = 15
S_DC_COEF = 16
S_LIM_ENV = 17
S_LIM_RELEASE = 18
S_OUTPUT_GAIN = 19
S_BODY_SCALE = 20
S_FS_OUT = 21
S_DEC_VALUE = 22    # last decimator output (before the gain)
S = 23

# ---------------------------------------------------------------- scalar ints: si int64[SI]
SI_N_NODES = 0
SI_N_BRANCHES = 1
SI_N_RAD = 2
SI_NOISE_N = 3      # active noise sources (0..3)
SI_CTRL_EVERY = 4   # internal samples per control step (64 * m)
SI_CTRL_COUNTER = 5  # internal samples left until the next control step
SI_BLOCK_POS = 6    # next row of the track arrays passed to run_samples
SI_NET_RNG = 7      # network noise rng state
SI_LIM_HOLD = 8
SI_LIM_HOLD_N = 9
SI_M = 10           # oversampling factor
SI_DEC_POS = 11
SI_DEC_COUNT = 12
SI_GLOTTIS_PORT = 13
SI_VELUM_PORT = 14
SI_LUNGS = 15
SI_SUB_NODE = 16
SI_SUPRA_NODE = 17
SI_FLAGS = 18
SI_ERROR = 19       # 0 = ok; ERR_* set inside the JIT instead of raising
SI_BLOCKS_DONE = 20  # control steps applied since construction (diagnostics)
SI = 21

F_VOLUME_SOURCE = 1
F_JUNCTION_FLOOR = 2
F_ENERGY_SAFE = 4
F_SYMMETRIC_KINETIC = 8
F_FLUSH_SUBNORMALS = 16
F_DEFAULT = F_VOLUME_SOURCE | F_JUNCTION_FLOOR | F_ENERGY_SAFE | F_FLUSH_SUBNORMALS

ERR_TRACK_EXHAUSTED = 1

# physics.rs (a JIT file may not import core.physics; equality is tested)
RHO = 1.14e-3
C_SOUND = 35000.0
MU = 1.86e-4
NU = MU / RHO
RHO_C2 = RHO * C_SOUND * C_SOUND
RE_CRIT = 1800.0
PI = math.pi
TAU = PI * 2.0

FLUSH_BELOW = 1e-30  # |x| below this is set to 0 by flush_subnormals (subnormal operands cost 2-6x per operation)
# state that decays (or is a decayed copy) and is flushed per control step; everything else is either recomputed
# every sample, a constant, driven by noise, or geometry
FLUSH_NODE_ROWS = (N_P, N_VOL_SRC1, N_VOL_SRC2, N_WALL_Y, N_WALL_V, N_WALL_U)
FLUSH_BRANCH_ROWS = (B_U, B_MOM)
FLUSH_RAD_ROWS = (R_UL, R_U)
FLUSH_NOISE_COLS = (Z_HP, Z_LP1, Z_LP2, Z_LEVEL)
FLUSH_GLOT = (G_VOICING, G_VOICING_SM, G_DP_SM, G_STATIC_SM, G_CLOSURE_SM, G_AMP_SM, G_AREA)
FLUSH_SC = (S_OUT, S_RAD_PREV_TOTAL, S_P_LUNG_NOW, S_P_LUNG_STEP, S_DC_X, S_DC_Y, S_LIM_ENV, S_DEC_VALUE)


@jit
def _shape(ph, oq, skew, sharp):
    """Normalised area pulse (0..1) at cycle phase ph (glottis.rs:232-246)."""
    if ph >= oq:
        return 0.0
    t = ph / oq
    s = nbm.clamp(skew, 0.3, 0.9)
    if t < s:
        return 0.5 - 0.5 * math.cos((PI * t) / s)
    u = (t - s) / (1.0 - s)
    soft = 0.5 + 0.5 * math.cos(PI * u)
    hard = math.cos(0.5 * PI * u)
    return soft + (hard - soft) * sharp


@jit
def _drift_step(glot, glot_i):
    """Drift::step (math.rs:248-258) on the glottis' drift state."""
    s = 0.0
    for i in range(3):
        a = glot[G_DRIFT_A + i]
        y = a * glot[G_DRIFT_Y + i]
        st, g = nbm.rng_gauss(int(glot_i[GI_DRIFT_RNG]))
        glot_i[GI_DRIFT_RNG] = st
        y = y + math.sqrt(1.0 - a * a) * g
        glot[G_DRIFT_Y + i] = y
        s += y / (i + 1.0)
    return s / 1.4


@jit
def glottis_next(glot, glot_i, dp):
    """ParametricGlottis::next (glottis.rs:148-224); returns the glottal area."""
    fs = glot[G_FS]
    glot[G_DP_SM] += glot[G_DP_COEF] * (dp - glot[G_DP_SM])
    glot_i[GI_DRIFT_COUNTER] -= 1
    if glot_i[GI_DRIFT_COUNTER] <= 0:
        glot_i[GI_DRIFT_COUNTER] = 32
        dv = _drift_step(glot, glot_i)
        glot[G_DRIFT_VALUE] = dv
        glot[G_VIB_PHASE] += ((glot[G_VIBRATO_RATE] * 32.0) / fs) * (1.0 + 0.04 * dv)
        if glot[G_VIB_PHASE] > 1.0:
            glot[G_VIB_PHASE] -= 1.0
        cents = glot[G_VIBRATO_DEPTH] * math.sin(TAU * glot[G_VIB_PHASE]) + glot[G_DRIFT] * dv
        glot[G_PITCH_FACTOR] = math.pow(2.0, cents / 1200.0)
    f0 = glot[G_F0] * glot[G_PITCH_FACTOR]
    fry = glot[G_FRY]
    if fry > 0.0:
        f0 = f0 * (1.0 - fry) + nbm.js_min(f0, 48.0) * fry
    glot[G_F0_NOW] = f0
    th = glot[G_THRESHOLD]
    pressure_gate = nbm.smoothstep(th * 0.8, th * 1.6, glot[G_DP_SM])
    adduct_gate = 1.0 - nbm.smoothstep(0.08, 0.22, glot[G_STATIC_SM] - glot[G_CHINK])
    target = pressure_gate * adduct_gate * (1.0 - glot[G_CLOSURE_SM])
    glot[G_VOICING_SM] += glot[G_VOICE_COEF] * (target - glot[G_VOICING_SM])
    glot[G_VOICING] = glot[G_VOICING_SM]
    glot[G_PHASE] += (f0 / fs) / glot[G_CYCLE_PERIOD_SCALE]
    if glot[G_PHASE] >= 1.0:
        glot[G_PHASE] -= 1.0
        glot_i[GI_CYCLE_INDEX] += 1
        alt = 1.0 if (glot_i[GI_CYCLE_INDEX] & 1) == 1 else -1.0
        chaos_p = 0.0
        chaos_a = 0.0
        chaos = glot[G_CHAOS]
        if chaos > 0.0:
            r = 2.9 + 1.1 * nbm.js_min(1.0, chaos)
            x = r * glot[G_CHAOS_X] * (1.0 - glot[G_CHAOS_X])
            if not (x > 0.0 and x < 1.0):
                x = 0.37
            glot[G_CHAOS_X] = x
            dev = x - 0.5
            chaos_p = 0.45 * chaos * dev
            chaos_a = 1.1 * chaos * dev
        st, g1 = nbm.rng_gauss(int(glot_i[GI_RNG]))
        glot_i[GI_RNG] = st
        cps = 1.0 + glot[G_JITTER] * g1 + glot[G_DIPLOPHONIA] * 0.08 * alt + chaos_p
        glot[G_CYCLE_PERIOD_SCALE] = nbm.clamp(cps, 0.55, 1.8)
        st, g2 = nbm.rng_gauss(int(glot_i[GI_RNG]))
        glot_i[GI_RNG] = st
        up = 1.0 if alt > 0.0 else 0.0
        glot[G_CYCLE_AMP] = nbm.js_max(0.12, 1.0 + glot[G_SHIMMER] * g2 - glot[G_DIPLOPHONIA] * 0.35 * up + chaos_a)
    oq = nbm.clamp(glot[G_OPEN_QUOTIENT] - 0.18 * (glot[G_EFFORT] - 0.5), 0.25, 0.98)
    if fry > 0.0:
        oq = oq * (1.0 - fry) + 0.28 * fry
    pulse = _shape(glot[G_PHASE], oq, glot[G_SKEW], glot[G_CLOSURE_SHARPNESS])
    if fry > 0.3:
        ph2 = glot[G_PHASE] - 0.38
        if ph2 > 0.0:
            pulse += 0.55 * nbm.smoothstep(0.3, 0.8, fry) * _shape(ph2, oq * 0.8, glot[G_SKEW],
                                                                     glot[G_CLOSURE_SHARPNESS])
    effort_scale = 0.45 + 0.75 * glot[G_EFFORT]
    sm = glot[G_SM_COEF]
    glot[G_AMP_SM] += sm * (glot[G_AMPLITUDE] * effort_scale - glot[G_AMP_SM])
    glot[G_CLOSURE_SM] += sm * (glot[G_CLOSURE] - glot[G_CLOSURE_SM])
    glot[G_STATIC_SM] += sm * (glot[G_ABDUCTION] + glot[G_CHINK] - glot[G_STATIC_SM])
    vib = glot[G_AMP_SM] * glot[G_CYCLE_AMP] * glot[G_VOICING_SM] * pulse
    statics = glot[G_STATIC_SM] * (1.0 - 0.92 * glot[G_CLOSURE_SM])
    a = nbm.js_max(0.0, vib + statics)
    glot[G_AREA] = a
    return a


@jit
def glottis_block(glot, glot_i, dp, area, f0_now, voicing):
    """ParametricGlottis::next for every dp[i] (sources/glottis.rs:148-224); records the returned area, f0_now and
    voicing after each call. The kernels advance the glottis with exactly this arithmetic."""
    for i in range(dp.shape[0]):
        area[i] = glottis_next(glot, glot_i, dp[i])
        f0_now[i] = glot[G_F0_NOW]
        voicing[i] = glot[G_VOICING]


@jit
def flush_subnormals(nf, bf, rf, noise_f, glot, sc):
    """|x| < FLUSH_BELOW -> 0.0 in every state row that decays. The template only flushes when the whole network is
    silent; closed side cavities and walls otherwise decay into subnormals for seconds."""
    for r in FLUSH_NODE_ROWS:
        for k in range(nf.shape[1]):
            if abs(nf[r, k]) < FLUSH_BELOW:
                nf[r, k] = 0.0
    for r in FLUSH_BRANCH_ROWS:
        for j in range(bf.shape[1]):
            if abs(bf[r, j]) < FLUSH_BELOW:
                bf[r, j] = 0.0
    for r in FLUSH_RAD_ROWS:
        for j in range(rf.shape[1]):
            if abs(rf[r, j]) < FLUSH_BELOW:
                rf[r, j] = 0.0
    for c in FLUSH_NOISE_COLS:
        for i in range(noise_f.shape[0]):
            if abs(noise_f[i, c]) < FLUSH_BELOW:
                noise_f[i, c] = 0.0
    for i in FLUSH_GLOT:
        if abs(glot[i]) < FLUSH_BELOW:
            glot[i] = 0.0
    for i in FLUSH_SC:
        if abs(sc[i]) < FLUSH_BELOW:
            sc[i] = 0.0


@jit
def _set_tube_targets(nf, nodes, areas, mins, off, length):
    """AirwayNetwork::set_tube_targets with min areas and a length (network.rs:633-649)."""
    n = nodes.shape[0]
    for i in range(n):
        k = nodes[i]
        a = nbm.js_max(0.0, areas[off + i])
        nf[N_AREA_T, k] = a
        nf[N_AMIN_T, k] = nbm.js_max(0.0, nbm.js_min(a, mins[off + i]))
    d = length / n
    for i in range(n):
        nf[N_DX_T, nodes[i]] = d


@jit
def _set_noise_sources(noise_f, noise_i, sc, si, n_new, idx, par, inherit):
    """set_noise_sources (network.rs:727-745) with the filter states taken from the previous final list by index."""
    old = np.empty((MAX_NOISE, 3))
    for j in range(MAX_NOISE):
        old[j, 0] = noise_f[j, Z_HP]
        old[j, 1] = noise_f[j, Z_LP1]
        old[j, 2] = noise_f[j, Z_LP2]
    fs = sc[S_FS]
    for j in range(MAX_NOISE):
        if j < n_new:
            noise_i[j, ZI_SRC] = idx[j, 0]
            noise_i[j, ZI_FWD] = idx[j, 1]
            noise_i[j, ZI_BWD] = idx[j, 2]
            noise_f[j, Z_GAIN] = par[j, 0]
            k = inherit[j]
            if k >= 0:
                noise_f[j, Z_HP] = old[k, 0]
                noise_f[j, Z_LP1] = old[k, 1]
                noise_f[j, Z_LP2] = old[k, 2]
            else:
                noise_f[j, Z_HP] = 0.0
                noise_f[j, Z_LP1] = 0.0
                noise_f[j, Z_LP2] = 0.0
            noise_f[j, Z_LP_COEF] = 1.0 - math.exp((-2.0 * PI * par[j, 1]) / fs)
            noise_f[j, Z_HP_COEF] = math.exp((-2.0 * PI * par[j, 2]) / fs)
            noise_f[j, Z_LEVEL] = 0.0
        else:
            noise_i[j, ZI_SRC] = -1
            noise_i[j, ZI_FWD] = -1
            noise_i[j, ZI_BWD] = -1
            for c in range(NZ):
                noise_f[j, c] = 0.0
    si[SI_NOISE_N] = n_new


@jit
def apply_targets(nf, bf, noise_f, noise_i, sc, si, ph_nodes, or_nodes, t_area, t_min, t_len, t_velum, t_noise_n,
                  t_noise_idx, t_noise_par, t_noise_inherit, row):
    """The airway part of control() for one track row: tube targets (pharynx, oral), velum port target, noise list
    (human_airway.rs:255-260, :323)."""
    n_ph = ph_nodes.shape[0]
    _set_tube_targets(nf, ph_nodes, t_area[row], t_min[row], 0, t_len[row, 0])
    _set_tube_targets(nf, or_nodes, t_area[row], t_min[row], n_ph, t_len[row, 1])
    bf[B_PORT_AREA_T, si[SI_VELUM_PORT]] = nbm.js_max(0.0, t_velum[row])
    _set_noise_sources(noise_f, noise_i, sc, si, t_noise_n[row], t_noise_idx[row], t_noise_par[row],
                       t_noise_inherit[row])


@jit
def _flush_state(nf, ni, bf, rf):
    """network.rs:747-760 (everything silent)."""
    for k in range(nf.shape[1]):
        if ni[NI_FIXED, k] == 0:
            nf[N_P, k] = 0.0
        nf[N_WALL_V, k] = 0.0
        if abs(nf[N_WALL_Y, k]) < 1e-12:
            nf[N_WALL_Y, k] = 0.0
    for j in range(bf.shape[1]):
        bf[B_U, j] = 0.0
        bf[B_MOM, j] = 0.0
    for r in range(rf.shape[1]):
        rf[R_UL, r] = 0.0


@jit
def begin_block(nf, ni, bf, bi, rf, samples):
    """AirwayNetwork::begin_block (network.rs:694-723)."""
    inv = 1.0 / nbm.js_max(1.0, float(samples))
    max_p = 0.0
    for k in range(nf.shape[1]):
        nf[N_AREA_S, k] = (nf[N_AREA_T, k] - nf[N_AREA, k]) * inv
        nf[N_AMIN_S, k] = (nf[N_AMIN_T, k] - nf[N_AMIN, k]) * inv
        nf[N_DX_S, k] = (nf[N_DX_T, k] - nf[N_DX, k]) * inv
        if ni[NI_FIXED, k] == 0:
            ap = abs(nf[N_P, k])
            if ap > max_p:
                max_p = ap
    if max_p < 1e-9:
        max_u = 0.0
        for j in range(bf.shape[1]):
            max_u = nbm.js_max(max_u, abs(bf[B_U, j]))
        if max_u < 1e-9:
            _flush_state(nf, ni, bf, rf)
    for j in range(bf.shape[1]):
        if bi[BI_KIND, j] == KIND_PORT:
            bf[B_PORT_AREA_S, j] = (bf[B_PORT_AREA_T, j] - bf[B_PORT_AREA, j]) * inv


@jit
def snap_to_targets(nf, bf, bi, sc):
    """AirwayNetwork::snap_to_targets (network.rs:674-691)."""
    min_a = sc[S_MIN_NODE_AREA]
    for k in range(nf.shape[1]):
        nf[N_AREA, k] = nf[N_AREA_T, k]
        nf[N_AMIN, k] = nf[N_AMIN_T, k]
        nf[N_DX, k] = nf[N_DX_T, k]
        nf[N_VOL, k] = nbm.js_max(nf[N_AREA, k], min_a) * nf[N_DX, k]
        nf[N_INV_A, k] = 1.0 / nbm.js_max(nf[N_AREA, k], 1e-6)
        nf[N_AREA_S, k] = 0.0
        nf[N_AMIN_S, k] = 0.0
        nf[N_DX_S, k] = 0.0
    for j in range(bf.shape[1]):
        if bi[BI_KIND, j] == KIND_PORT:
            bf[B_PORT_AREA, j] = bf[B_PORT_AREA_T, j]
            bf[B_PORT_AREA_S, j] = 0.0


@jit
def _compute_noise(bf, noise_f, noise_i, sc, si):
    """network.rs:1097-1131."""
    scale = sc[S_NOISE_SCALE] * math.sqrt(96000.0 / sc[S_FS])
    sat = 12000.0 * 12000.0 - RE_CRIT * RE_CRIT
    closed = sc[S_CLOSED_AREA]
    for i in range(si[SI_NOISE_N]):
        src = noise_i[i, ZI_SRC]
        a = bf[B_AREA, src]
        u = bf[B_U, src]
        amp = 0.0
        if a > closed:
            re = (abs(u) * 2.0) / (NU * math.sqrt(PI * a))
            if re > RE_CRIT:
                x = re * re - RE_CRIT * RE_CRIT
                amp = (noise_f[i, Z_GAIN] * scale * x) / (1.0 + x / sat)
        st, r = nbm.rng_next(int(si[SI_NET_RNG]))
        si[SI_NET_RNG] = st
        w = (r * 2.0 - 1.0) * 1.732
        hp = noise_f[i, Z_HP]
        hp += (1.0 - noise_f[i, Z_HP_COEF]) * (w - hp)
        noise_f[i, Z_HP] = hp
        hp_out = w - hp
        lc = noise_f[i, Z_LP_COEF]
        lp1 = noise_f[i, Z_LP1]
        lp1 += lc * (hp_out - lp1)
        noise_f[i, Z_LP1] = lp1
        lp2 = noise_f[i, Z_LP2]
        lp2 += lc * (lp1 - lp2)
        noise_f[i, Z_LP2] = lp2
        noise_f[i, Z_LEVEL] = amp
        if amp == 0.0:
            continue
        v = amp * lp2
        if u >= 0.0:
            bf[B_PSRC, noise_i[i, ZI_FWD]] += v
        else:
            bf[B_PSRC, noise_i[i, ZI_BWD]] -= v


@jit
def net_step(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, sc, si):
    """AirwayNetwork::step (network.rs:769-1095); returns the radiated output."""
    n = si[SI_N_NODES]
    nb = si[SI_N_BRANCHES]
    t = sc[S_T]
    inv_t = sc[S_INV_T]
    min_a = sc[S_MIN_NODE_AREA]
    flags = si[SI_FLAGS]
    sq = sc[S_SQUEEZE_AREA]
    vol_on = (flags & F_VOLUME_SOURCE) != 0
    p = nf[N_P]
    area = nf[N_AREA]
    amin = nf[N_AMIN]
    amin_t = nf[N_AMIN_T]
    dx = nf[N_DX]
    vol = nf[N_VOL]
    inv_a = nf[N_INV_A]
    inv_amin = nf[N_INV_AMIN]
    inv_lsum = nf[N_INV_LSUM]
    half_l = nf[N_HALF_L]
    half_r = nf[N_HALF_R]
    sqrt_a = nf[N_SQRT_A]
    flow_sum = nf[N_FLOW_SUM]
    spill = nf[N_SPILL]
    wall_perim = nf[N_WALL_PERIM]
    visc_f = nf[N_VISC_F]
    # 1) geometry ramps and the moving-wall volume source
    if vol_on:
        for k in range(n):
            spill[k] = 0.0
    for k in range(n):
        a = area[k] + nf[N_AREA_S, k]
        if a < 0.0:
            a = 0.0
        area[k] = a
        am = amin[k] + nf[N_AMIN_S, k]
        if am < 0.0:
            am = 0.0
        amk = a if am > a else am
        amin[k] = amk
        dx[k] += nf[N_DX_S, k]
        d = dx[k]
        ia = 1.0 / (a if a > 1e-6 else 1e-6)
        inv_a[k] = ia
        inv_amin[k] = 1.0 / (amk if amk > 1e-6 else 1e-6)
        sa = math.sqrt(a)
        sqrt_a[k] = sa
        half_l[k] = d * ia
        half_r[k] = wall_perim[k] * d * sa * ia * ia * visc_f[k]
        v = (a if a > min_a else min_a) * d
        if vol_on:
            raw = -(v - vol[k]) * inv_t
            ae = amin[k] if amin[k] < a else a
            if raw != 0.0 and ae < sq:
                tt = ae / sq
                keep = tt * tt * (3.0 - 2.0 * tt)
                spill[k] += raw * keep
                sp = raw * (1.0 - keep)
                s0 = nbr_start[k]
                s1 = nbr_start[k + 1]
                n_open = 0
                for e in range(s0, s1):
                    m = nbr[e]
                    if nbm.js_min(area[m], amin[m]) >= sq or amin_t[m] >= sq:
                        n_open += 1
                if n_open > 0:
                    share = sp / n_open
                    for e in range(s0, s1):
                        m = nbr[e]
                        if nbm.js_min(area[m], amin[m]) >= sq or amin_t[m] >= sq:
                            spill[m] += share
            else:
                spill[k] += raw
        vol[k] = v
        inv_lsum[k] = 0.0
        flow_sum[k] = nf[N_EXT_FLOW, k]
        nf[N_EXT_FLOW, k] = 0.0
    if vol_on:
        c = sc[S_VOL_LP_COEF]
        for k in range(n):
            nf[N_VOL_SRC1, k] += c * (spill[k] - nf[N_VOL_SRC1, k])
            s1v = nf[N_VOL_SRC1, k]
            nf[N_VOL_SRC2, k] += c * (s1v - nf[N_VOL_SRC2, k])
            flow_sum[k] += nf[N_VOL_SRC2, k]
    # 2) turbulence noise -> series pressure sources
    for j in range(nb):
        bf[B_PSRC, j] = 0.0
    if si[SI_NOISE_N] > 0:
        _compute_noise(bf, noise_f, noise_i, sc, si)
    # 3) branches
    visc = math.sqrt((RHO * MU * sc[S_VISCOUS_OMEGA]) / 2.0) * sc[S_VISCOUS_SCALE]
    closed = sc[S_CLOSED_AREA]
    half_rho = 0.5 * RHO
    energy_safe = (flags & F_ENERGY_SAFE) != 0
    symmetric = (flags & F_SYMMETRIC_KINETIC) != 0
    is_j = ni[NI_JUNCTION]
    for j in range(nb):
        a = bi[BI_A, j]
        b = bi[BI_B, j]
        if bi[BI_KIND, j] == KIND_SERIES:
            aa = area[a]
            ab = area[b]
            ma = amin[a]
            mb = amin[b]
            a_eff = ma if ma < mb else mb
            if (aa if aa < ab else ab) < closed or a_eff < closed:
                bf[B_U, j] = 0.0
                bf[B_MOM, j] = 0.0
                bf[B_AREA, j] = a_eff
                continue
            ia = inv_a[a]
            ib = inv_a[b]
            ll = half_rho * (half_l[a] + half_l[b])
            r = 0.5 * visc * (half_r[a] + half_r[b])
            ima = inv_amin[a]
            imb = inv_amin[b]
            ef = ima - ib
            eb = imb - ia
            kf = half_rho * ef * ef if ef > 0.0 else 0.0
            kb = half_rho * eb * eb if eb > 0.0 else 0.0
            cf = imb - ia
            cb = ima - ib
            if cf > 0.0:
                kf += 0.1 * half_rho * cf * cf
            if cb > 0.0:
                kb += 0.1 * half_rho * cb * cb
        else:
            bf[B_PORT_AREA, j] += bf[B_PORT_AREA_S, j]
            ap = bf[B_PORT_AREA, j]
            if ap < 0.0:
                bf[B_PORT_AREA, j] = 0.0
                ap = 0.0
            a_eff = ap
            if ap < closed:
                bf[B_U, j] = 0.0
                bf[B_MOM, j] = 0.0
                bf[B_AREA, j] = ap
                continue
            ip = 1.0 / ap
            ln = bf[B_PORT_LEN, j]
            ll = RHO * ln * ip
            if bi[BI_PORT_HALF, j] != 0:
                ll += half_rho * (half_l[a] + half_l[b])
            w = bf[B_PORT_SLIT, j]
            if w > 0.0:
                r = (12.0 * MU * ln * w * w) * ip * ip * ip
            else:
                r = visc * 2.0 * math.sqrt(PI) * ln * math.sqrt(ap) * ip * ip
            r += bf[B_PORT_EXTRA_R, j]
            ex_f = ip - inv_a[b]
            ex_b = ip - inv_a[a]
            kin = bf[B_PORT_KIN, j] * half_rho * ip * ip
            kf = kin + bf[B_PORT_KEX, j] * half_rho * (ex_f * ex_f if ex_f > 0.0 else 0.0)
            kb = kin + bf[B_PORT_KEX, j] * half_rho * (ex_b * ex_b if ex_b > 0.0 else 0.0)
        bf[B_AREA, j] = a_eff
        if symmetric:
            km = kf if kf > kb else kb
            kf = km
            kb = km
        lp = bf[B_L_PREV, j]
        if energy_safe and ll < lp:
            m0 = ll * bf[B_U, j]
        else:
            m0 = bf[B_MOM, j]
        c = p[a] - p[b] + bf[B_PSRC, j] + m0 * inv_t
        bq = ll * inv_t + r
        if c >= 0.0:
            if kf > 0.0:
                u = (2.0 * c) / (bq + math.sqrt(bq * bq + 4.0 * kf * c))
            else:
                u = c / bq
        elif kb > 0.0:
            u = (2.0 * c) / (bq + math.sqrt(bq * bq - 4.0 * kb * c))
        else:
            u = c / bq
        bf[B_U, j] = u
        bf[B_MOM, j] = ll * u
        bf[B_L_PREV, j] = ll
        flow_sum[a] -= u
        flow_sum[b] += u
        if is_j[a] != 0 or is_j[b] != 0:
            il = 1.0 / ll
            inv_lsum[a] += il
            inv_lsum[b] += il
    # 4) walls
    wall_rad_sum = 0.0
    for k in range(n):
        if ni[NI_HAS_WALL, k] == 0:
            continue
        m = nf[N_WALL_M, k]
        r = nf[N_WALL_R, k]
        v = ((m * inv_t - 0.5 * r) * nf[N_WALL_V, k] + p[k] - nf[N_WALL_K, k] * nf[N_WALL_Y, k]) / (m * inv_t + 0.5 * r)
        nf[N_WALL_V, k] = v
        nf[N_WALL_Y, k] += t * v
        sa = sqrt_a[k] if area[k] > min_a else math.sqrt(min_a)
        uw = wall_perim[k] * sa * dx[k] * v
        nf[N_WALL_U, k] = uw
        flow_sum[k] -= uw
        wall_rad_sum += uw * nf[N_WALL_RAD, k]
    # 5) nodes (radiation loads semi-implicit)
    t2 = t * t
    rad_total = wall_rad_sum
    junction_floor = (flags & F_JUNCTION_FLOOR) != 0
    for k in range(n):
        if ni[NI_FIXED, k] != 0:
            continue
        if area[k] < closed or amin[k] < closed:
            rr = ni[NI_RAD_OF, k]
            if rr >= 0:
                p[k] *= 0.98
            else:
                ssum = 0.0
                cnt = 0
                for e in range(nbr_start[k], nbr_start[k + 1]):
                    m2 = nbr[e]
                    if area[m2] >= closed and amin[m2] >= closed:
                        ssum += p[m2]
                        cnt += 1
                if cnt > 0:
                    p[k] += 0.02 * (ssum / cnt - p[k])
            if rr >= 0:
                rf[R_UL, rr] *= 0.995
                rf[R_U, rr] = rf[R_UL, rr]
                rad_total += rf[R_UL, rr]
            continue
        c = vol[k] / RHO_C2
        if is_j[k] != 0 and junction_floor:
            c_min = (t2 * inv_lsum[k]) / 2.4
            if c < c_min:
                c = c_min
        r = ni[NI_RAD_OF, k]
        if r < 0:
            p[k] += (t / c) * flow_sum[k]
        else:
            a = area[k] if area[k] > 1e-4 else 1e-4
            lr = (8.0 * RHO) / (3.0 * PI * math.sqrt(PI * a))
            rres = (128.0 * RHO * C_SOUND) / (9.0 * PI * PI * a)
            p_old = p[k]
            lp = rf[R_L_PREV, r]
            ul_prev = rf[R_UL, r]
            if lp > 0.0 and lr > lp:
                ul_prev *= lp / lr
            rf[R_L_PREV, r] = lr
            h = t / (2.0 * lr)
            g = 0.5 / rres
            p_new = (p_old * (c * inv_t - 0.5 * h - g) + flow_sum[k] - ul_prev) / (c * inv_t + 0.5 * h + g)
            p[k] = p_new
            ul = ul_prev + h * (p_old + p_new)
            rf[R_UL, r] = ul
            ur = 0.5 * (ul_prev + ul) + g * (p_new + p_old)
            rf[R_U, r] = ur
            rad_total += ur
    out = (rad_total - sc[S_RAD_PREV_TOTAL]) * inv_t
    sc[S_RAD_PREV_TOTAL] = rad_total
    sc[S_OUT] = out
    return out


@jit
def control_step(nf, ni, bf, bi, rf, noise_f, noise_i, glot, sc, si, ph_nodes, or_nodes, t_area, t_min, t_len,
                 t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung, row):
    """What control() leaves in the acoustic state for one track row (human_voice.rs:426-501)."""
    apply_targets(nf, bf, noise_f, noise_i, sc, si, ph_nodes, or_nodes, t_area, t_min, t_len, t_velum, t_noise_n,
                  t_noise_idx, t_noise_par, t_noise_inherit, row)
    every = si[SI_CTRL_EVERY]
    begin_block(nf, ni, bf, bi, rf, every)
    si[SI_CTRL_COUNTER] = every
    sc[S_P_LUNG_STEP] = (t_plung[row] - sc[S_P_LUNG_NOW]) / float(every)
    for i in range(GLOTTIS_NPARAMS):
        glot[i] = t_glot[row, i]


@jit
def run_samples(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, glot, glot_i, sc, si,
                ph_nodes, or_nodes,
                t_area, t_min, t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung,
                n_internal, raw):
    """Advance n_internal internal samples, writing the radiated value of each into raw. A control step (flush, next
    track row: tube targets, velum target, noise sources, block ramps, lung-pressure step, glottis parameters) runs
    before every sample at which the control counter is <= 0. A missing track row sets si[SI_ERROR]."""
    n_rows = t_area.shape[0]
    lungs = si[SI_LUNGS]
    sub = si[SI_SUB_NODE]
    supra = si[SI_SUPRA_NODE]
    gp = si[SI_GLOTTIS_PORT]
    for i in range(n_internal):
        if si[SI_CTRL_COUNTER] <= 0:
            row = si[SI_BLOCK_POS]
            if row >= n_rows:
                si[SI_ERROR] = ERR_TRACK_EXHAUSTED
                return
            if (si[SI_FLAGS] & F_FLUSH_SUBNORMALS) != 0:
                flush_subnormals(nf, bf, rf, noise_f, glot, sc)
            control_step(nf, ni, bf, bi, rf, noise_f, noise_i, glot, sc, si, ph_nodes, or_nodes, t_area, t_min,
                         t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung, row)
            si[SI_BLOCK_POS] = row + 1
            si[SI_BLOCKS_DONE] += 1
        si[SI_CTRL_COUNTER] -= 1
        sc[S_P_LUNG_NOW] += sc[S_P_LUNG_STEP]
        nf[N_P, lungs] = sc[S_P_LUNG_NOW]
        a = glottis_next(glot, glot_i, nf[N_P, sub] - nf[N_P, supra])
        pa = 0.0 if a < 0.0 else a
        bf[B_PORT_AREA, gp] = pa
        bf[B_PORT_AREA_T, gp] = pa
        bf[B_PORT_AREA_S, gp] = 0.0
        raw[i] = net_step(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, sc, si)


@jit
def output_chain(sc, si, dec_c, dec_buf, raw, n_internal, out):
    """Decimator, gain, DC blocker, limiter, soft clip, float32 (human_voice.rs:366-393); returns the count."""
    m = si[SI_M]
    ntaps = dec_c.shape[0]
    count = 0
    for i in range(n_internal):
        x = raw[i]
        pos = si[SI_DEC_POS]
        dec_buf[pos] = x
        dec_buf[pos + ntaps] = x
        pos = 0 if pos + 1 == ntaps else pos + 1
        si[SI_DEC_POS] = pos
        si[SI_DEC_COUNT] += 1
        if si[SI_DEC_COUNT] < m:
            continue
        si[SI_DEC_COUNT] = 0
        acc = 0.0
        for k in range(ntaps):
            acc += dec_c[ntaps - 1 - k] * dec_buf[pos + k]
        sc[S_DEC_VALUE] = acc
        v = acc * sc[S_OUTPUT_GAIN]
        hp = v - sc[S_DC_X] + sc[S_DC_COEF] * sc[S_DC_Y]
        sc[S_DC_X] = v
        sc[S_DC_Y] = hp
        v = hp
        av = abs(v)
        if av >= sc[S_LIM_ENV]:
            sc[S_LIM_ENV] = av
            si[SI_LIM_HOLD] = si[SI_LIM_HOLD_N]
        elif si[SI_LIM_HOLD] > 0:
            si[SI_LIM_HOLD] -= 1
        else:
            sc[S_LIM_ENV] *= sc[S_LIM_RELEASE]
        if sc[S_LIM_ENV] > 0.75:
            v *= 0.75 / sc[S_LIM_ENV]
        al = abs(v)
        if al > 0.85:
            v = nbm.js_sign(v) * (0.85 + 0.15 * math.tanh((al - 0.85) / 0.15))
        out[count] = v
        count += 1
    return count
