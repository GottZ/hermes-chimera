"""JIT helpers with the template's JavaScript number semantics (math.rs) -- the ONLY implementation.

JIT file: loaded under the fixed module name ``_chimera_nbmath`` by the worker's
numba loader, which sets ``CHIMERA_JIT`` to ``numba.njit(...)`` before executing this file. Imported as
part of the package, ``jit`` is the identity and every function runs as plain CPython (the reference).

Rules for this file: only ``math`` is imported; every function takes and returns float64/int64 scalars
(or tuples of them); no classes. ``jsmath`` re-exports these functions and adds the Rng/Drift classes.
u32 arithmetic is done in int64 and masked after every operation, so CPython and numba agree bit for bit.
"""

import math

jit = globals().get("CHIMERA_JIT", lambda f: f)

U32 = 0xFFFFFFFF
TWO32 = 4294967296.0
TWO52 = 4503599627370496.0
I64_MAX_F = 9.223372036854775807e18


@jit
def js_min(a, b):
    """Math.min: NaN if either is NaN; -0 is below +0 (math.rs:87-100)."""
    if a != a or b != b:
        return math.nan
    if a < b:
        return a
    if b < a:
        return b
    if math.copysign(1.0, a) < 0.0:
        return a
    return b


@jit
def js_max(a, b):
    """Math.max: NaN if either is NaN; +0 is above -0 (math.rs:101-116)."""
    if a != a or b != b:
        return math.nan
    if a > b:
        return a
    if b > a:
        return b
    if math.copysign(1.0, a) > 0.0:
        return a
    return b


@jit
def js_round(x):
    """Math.round: floor, halves toward +inf (math.rs:117-122). NaN/inf pass through."""
    if x != x or x == 0.0:
        return x  # NaN, +0.0 and -0.0 come back unchanged (floor keeps the sign of zero)
    if x >= TWO52 or x <= -TWO52:
        return x  # already integral (also covers +-inf)
    r = float(math.floor(x))
    if x - r >= 0.5:
        return r + 1.0
    return r


@jit
def js_sign(x):
    """Math.sign: +1, -1, or x itself (keeps -0.0 and NaN)."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return x


@jit
def js_trunc_i32(x):
    """``x | 0`` as math.rs: non-finite -> 0, else trunc -> i64 (saturating) -> i32 (wrapping)."""
    if x != x or x == math.inf or x == -math.inf:
        return 0
    if x >= I64_MAX_F:
        v = 9223372036854775807
    elif x <= -I64_MAX_F:
        v = -9223372036854775807 - 1
    else:
        v = int(x)
    v = v & U32
    if v >= 0x80000000:
        v -= 0x100000000
    return v


@jit
def clamp(x, lo, hi):
    """math.rs clamp: NaN passes through."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@jit
def fdiv(a, b):
    """IEEE-754 division (Rust ``a / b``): x/0 -> +-inf, 0/0 and NaN -> NaN; never raises."""
    if b == 0.0:
        if a != a or a == 0.0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


@jit
def smoothstep(e0, e1, x):
    """math.rs smoothstep: t = clamp((x-e0)/(e1-e0), 0, 1); t*t*(3-2t)."""
    t = clamp(fdiv(x - e0, e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@jit
def rng_next(state):
    """One mulberry32 step (math.rs:189-195). state: u32 in int64. Returns (new state, value in [0,1))."""
    s = (state + 0x6D2B79F5) & U32
    t = s
    t = ((t ^ (t >> 15)) * (t | 1)) & U32
    t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & U32)) & U32)) & U32
    return s, (t ^ (t >> 14)) / TWO32


@jit
def rng_gauss(state):
    """Four rng_next draws, (a+b+c+d-2.0)*1.7320508 summed left to right (math.rs gauss)."""
    state, a = rng_next(state)
    state, b = rng_next(state)
    state, c = rng_next(state)
    state, d = rng_next(state)
    return state, (a + b + c + d - 2.0) * 1.7320508
