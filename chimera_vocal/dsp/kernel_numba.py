"""Acoustic kernel, scalar form: the numba fast path and the CPython reference (INTERFACES §6.4, §6.4a).

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

FLUSH_BELOW = 1e-30  # |x| below this is set to 0 by flush_subnormals (DESIGN §2.9)
# state that decays (or is a decayed copy) and is flushed per control step; everything else is either recomputed
# every sample, a constant, driven by noise, or geometry
FLUSH_NODE_ROWS = (N_P, N_VOL_SRC1, N_VOL_SRC2, N_WALL_Y, N_WALL_V, N_WALL_U)
FLUSH_BRANCH_ROWS = (B_U, B_MOM)
FLUSH_RAD_ROWS = (R_UL, R_U)
FLUSH_NOISE_COLS = (Z_HP, Z_LP1, Z_LP2, Z_LEVEL)
FLUSH_GLOT = (G_VOICING, G_VOICING_SM, G_DP_SM, G_STATIC_SM, G_CLOSURE_SM, G_AMP_SM, G_AREA)
FLUSH_SC = (S_OUT, S_RAD_PREV_TOTAL, S_P_LUNG_NOW, S_P_LUNG_STEP, S_DC_X, S_DC_Y, S_LIM_ENV, S_DEC_VALUE)


@jit
def glottis_block(glot, glot_i, dp, area, f0_now, voicing):
    """ParametricGlottis::next for every dp[i] (sources/glottis.rs:148-224); records the returned area, f0_now and
    voicing after each call. The kernels advance the glottis with exactly this arithmetic."""
    raise NotImplementedError


@jit
def flush_subnormals(nf, bf, rf, noise_f, glot, sc):
    """|x| < FLUSH_BELOW -> 0.0 in every state row that decays (DESIGN §2.9)."""
    raise NotImplementedError


@jit
def run_samples(nf, ni, nbr_start, nbr, bf, bi, rf, ri, noise_f, noise_i, glot, glot_i, sc, si,
                ph_nodes, or_nodes,
                t_area, t_min, t_len, t_velum, t_noise_n, t_noise_idx, t_noise_par, t_noise_inherit, t_glot, t_plung,
                n_internal, raw):
    """Advance n_internal internal samples, writing the radiated value of each into raw (INTERFACES §6.4)."""
    raise NotImplementedError


@jit
def output_chain(sc, si, dec_c, dec_buf, raw, n_internal, out):
    """Decimator, gain, DC blocker, limiter, soft clip, float32 (human_voice.rs:366-393); returns the count."""
    raise NotImplementedError
