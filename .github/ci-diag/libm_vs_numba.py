"""Temporary CI diagnostic: where do numba-compiled math calls differ from CPython's math module on this platform?
Runs numba in this process (a diagnostic, not the plugin). Prints mismatch counts and worst ulp distance."""

import math
import sys

import numba
import numpy as np


@numba.njit
def cos_sin(a):
    return math.cos(a), math.sin(a)


@numba.njit
def cos_only(a):
    return math.cos(a)


@numba.njit
def sin_only(a):
    return math.sin(a)


@numba.njit
def hypot(x, y):
    return math.hypot(x, y)


@numba.njit
def pow_(x, y):
    return math.pow(x, y)


@numba.njit
def atan2(y, x):
    return math.atan2(y, x)


@numba.njit
def acos(x):
    return math.acos(x)


@numba.njit
def mulsub(a, b, c, d):
    return a * b - c * d


def ulps(a, b):
    if a == b:
        return 0
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def report(name, pairs):
    bad = [(u, x) for u, x in ((ulps(p, q), x) for p, q, x in pairs) if u]
    worst = max(bad) if bad else (0, None)
    print(f"{name:10s} {len(bad):6d} of {len(pairs)} differ, worst {worst[0]} ulp at {worst[1]!r}")


rng = np.random.default_rng(20261109)
n = 100000
ang = list(rng.uniform(-4.0, 4.0, n)) + [math.radians(180.0 + (36.0 - 180.0) * i / 30) for i in range(31)]
pairs = []
for a in ang:
    c, s = cos_sin(a)
    pairs.append((c, math.cos(a), a))
    pairs.append((s, math.sin(a), a))
report("cos+sin", pairs)
report("cos", [(cos_only(a), math.cos(a), a) for a in ang])
report("sin", [(sin_only(a), math.sin(a), a) for a in ang])
xy = rng.uniform(-20.0, 20.0, (n, 2))
report("hypot", [(hypot(x, y), math.hypot(x, y), (x, y)) for x, y in xy])
pp = np.abs(rng.uniform(0.0, 5.0, (n, 2)))
report("pow", [(pow_(x, y), math.pow(x, y), (x, y)) for x, y in pp])
report("atan2", [(atan2(y, x), math.atan2(y, x), (y, x)) for x, y in xy])
report("acos", [(acos(x), math.acos(x), x) for x in rng.uniform(-1.0, 1.0, n)])
q = rng.uniform(-3.0, 3.0, (n, 4))
report("a*b-c*d", [(mulsub(a, b, c, d), a * b - c * d, (a, b, c, d)) for a, b, c, d in q])
print(sys.platform, numba.__version__, np.__version__, sys.version.split()[0])
