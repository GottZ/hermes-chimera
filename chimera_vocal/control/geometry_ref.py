"""Vocal-tract geometry, scalar reference form: HumanVocalTract::compute / ::resample (anatomy/human.rs:725-1370,
anatomy/geom.rs) in the numba subset.

JIT file: loaded by the worker's numba loader under the fixed module name
``_chimera_geometry_ref`` (entry points ``geometry_compute``, ``geometry_resample``; manifest in
``control/jit_manifest.py``). Imported as part of the package, ``jit`` is the identity and this module is the
CPython reference of the tests (``GEOMETRY_REF_CPYTHON`` in ``control/geometry.py``).

Rules: imports only ``math``, ``numpy`` and ``_chimera_nbmath``; every function is ``@jit``; functions take only
numpy arrays (float64, C-contiguous) and Python scalars; no classes. Because a JIT file cannot import the
package, the frame and tract layouts are defined HERE and re-exported by ``control/frame_layout.py``.

Frame (float64[NFRAME], written by geometry_compute, read by geometry_resample and the control model):
scalars at F_* (the first 16 in the order of the golden test's ``scalars``, golden_geometry.rs:37), the grid as
rows of NG columns starting at F_GRID (row count in F_N_GRID), six outlines as interleaved (x, y) points starting
at F_OUTLINES + o * 2 * MAX_PTS (point counts in F_N_OUTLINE + o), then private scratch space of the geometry.
Tract (float64[NTRACT], written by frame_layout.tract_constants(body) through ``tract_init``): everything
HumanVocalTract::new precomputes (human.rs:644-679). Its member offsets (T_*) are private to this module.

Port conventions. A Vec2 is a pair of float locals (functions return (x, y) tuples); a Vec<Vec2> / Polyline is
an interleaved (x, y) run in a float64 array, addressed as (array, base, point count). Every expression keeps
the template's operand order and grouping (Rust and Python both fold ``a + b + c`` and ``a * b * c`` from the
left), so CPython and numba compute the template's floating-point operations one for one; transcendental
functions are the platform libm's (``math.*``), the template's are the ``libm`` crate's (math.rs).
"""

import math

import numpy as np

try:
    import _chimera_nbmath as nbm
except ImportError:
    from ..core import nbmath as nbm

jit = globals().get("CHIMERA_JIT", lambda f: f)

# ---- frame layout -------------------------------------------------------------------------------------------
MAX_GRID = 64        # grid rows (template: 12 pharyngeal + 31 oral fan + 6 front rays = 49 at most)
MAX_PTS = 256        # points per outline (human.rs:568 POINTS)

# scalars; F_LENGTH .. F_CONTACT_BODY_NY are the golden test's 16 `scalars` in that order (golden_geometry.rs:37)
F_LENGTH = 0
F_PHARYNX_LENGTH = 1
F_TEETH_S = 2
F_LIP_S = 3
F_VELUM_PORT = 4
F_LIP_AREA = 5
F_TONGUE_RADIUS = 6
F_TIP_X = 7
F_TIP_Y = 8
F_SPLIT_INDEX = 9
F_CONTACT_TIP = 10
F_CONTACT_BODY = 11
F_CONTACT_TIP_NX = 12
F_CONTACT_TIP_NY = 13
F_CONTACT_BODY_NX = 14
F_CONTACT_BODY_NY = 15
F_GLOTTIS_X = 16
F_GLOTTIS_Y = 17
F_TONGUE_CX = 18
F_TONGUE_CY = 19
F_N_GRID = 20
F_N_OUTLINE = 21     # 21..26: point count of outline o (O_* order)
N_SCALAR_SLOTS = 32

# grid row r, column c at F_GRID + r * NG + c (GridSample, human.rs:453-468)
G_REGION = 0         # 0 epilarynx, 1 pharynx, 2 oral, 3 front
G_D = 1
G_AREA = 2
G_S = 3
G_MID_X = 4
G_MID_Y = 5
G_IN_X = 6
G_IN_Y = 7
G_OUT_X = 8
G_OUT_Y = 9
G_RAY = 10
NG = 11
F_GRID = N_SCALAR_SLOTS

# outlines (HumanFrame fields checked by the golden test, golden_geometry.rs:38-42)
O_OUTER = 0
O_INNER = 1
O_TONGUE = 2
O_VELUM = 3
O_UPPER_LIP = 4
O_LOWER_LIP = 5
N_OUTLINES = 6
F_OUTLINES = F_GRID + MAX_GRID * NG

F_SCRATCH = F_OUTLINES + N_OUTLINES * 2 * MAX_PTS
N_SCRATCH = 16384    # private working space of geometry_compute (8128 used, layout S_* below)
NFRAME = F_SCRATCH + N_SCRATCH

# scratch (the template's Scratch, human.rs:542-565). outer/inner are built in place in the O_OUTER/O_INNER
# outlines (HumanFrame.outer/.inner are copies of them, human.rs:1245-1246); the roof is read in place from
# the outer outline (it is a copy of outer[roof_start..=roof_end], human.rs:1078-1080).
MAX_CROSS = 64       # human.rs:519
PB = 2 * MAX_PTS     # floats per point buffer
S_TMP = F_SCRATCH
S_CROSS_T = S_TMP + MAX_CROSS
S_CROSS_EXIT = S_CROSS_T + MAX_CROSS      # 1.0 = exit, 0.0 = entry
S_CUM = S_CROSS_EXIT + MAX_CROSS          # MAX_PTS floats
S_VEL = S_CUM + MAX_PTS
S_VEL_ORAL = S_VEL + PB
S_VEL_NASAL = S_VEL_ORAL + PB
S_ULIP_IN = S_VEL_NASAL + PB
S_LLIP_IN = S_ULIP_IN + PB
S_LIP_OUT = S_LLIP_IN + PB
S_ROOT = S_LIP_OUT + PB
S_ARC = S_ROOT + PB
S_BLADE = S_ARC + PB
S_UNDER = S_BLADE + PB
S_FLOOR = S_UNDER + PB
S_LTEETH = S_FLOOR + PB
S_BACK = S_LTEETH + PB
S_TIP_PTS = S_BACK + PB
S_OUTLINE = S_TIP_PTS + PB
S_END = S_OUTLINE + PB
assert S_END <= F_SCRATCH + N_SCRATCH

# ---- tract layout (HumanVocalTract::new, human.rs:644-679) ---------------------------------------------------
NTRACT = 1024
T_K = 0              # body scale k
T_PHARYNX = 1        # body.pharynx
T_TONGUE_RADIUS = 2  # body.tongue_radius (before scale)
T_LIP_WIDTH = 3      # body.lip_width (before scale)
T_LAW_PH_A = 4       # AreaLaw (alpha, beta) per region: pharynx, oral, front, ahead
T_LAW_PH_B = 5
T_LAW_OR_A = 6
T_LAW_OR_B = 7
T_LAW_FR_A = 8
T_LAW_FR_B = 9
T_LAW_AH_A = 10
T_LAW_AH_B = 11
T_PNS_X = 12
T_PNS_Y = 13
T_CONDYLE_X = 14
T_CONDYLE_Y = 15
T_POLAR_C_X = 16
T_POLAR_C_Y = 17
T_POLAR_C2_X = 18
T_POLAR_C2_Y = 19
T_N_PHAR = 20        # point counts of the three polylines
T_N_PALATE = 21
T_N_NASAL = 22
T_POLY_CAP = 64      # points reserved per polyline
T_PHAR = 32          # interleaved (x, y): posterior pharyngeal wall (33 points)
T_PALATE = T_PHAR + 2 * T_POLY_CAP        # palate (34 points)
T_NASAL = T_PALATE + 2 * T_POLY_CAP       # nasal cavity (17 points)
assert T_NASAL + 2 * T_POLY_CAP <= NTRACT

# articulation vector (ARTIC_KEYS order, human.rs:83-105)
A_JAW = 0
A_TB_X = 1
A_TB_Y = 2
A_TT_X = 3
A_TT_Y = 4
A_TT_CURL = 5
A_ROOT = 6
A_LIP_AP = 7
A_LIP_PROT = 8
A_LIP_WIDTH = 9
A_LABIODENTAL = 10
A_VELUM = 11
A_LARYNX = 12
A_EPILARYNX = 13
A_VENTRICULAR = 14
A_LATERAL = 15
A_DORSUM = 16

# ---- constants (math.rs, human.rs:507-519) -------------------------------------------------------------------
PI = math.pi
DEG = PI / 180.0
LAB_HANDOVER = 0.5
LAB_GAP = 0.3
LAB_FRONT_OPENS = 0.7
LAB_FRONT_RAMP = 0.1
ARC_POINTS = 32
NEUTRAL_TB_X = 2.9   # NEUTRAL.tb_x / tb_y (human.rs:421-439)
NEUTRAL_TB_Y = 6.85
N_PH_RAYS = 12       # pharyngeal horizontals (human.rs:929)


# ---- scalar helpers (geom.rs, math.rs) --------------------------------------------------------------------

@jit
def lerp(a, b, t):
    """math.rs lerp: a + (b - a) * t."""
    return a + (b - a) * t


@jit
def dist(ax, ay, bx, by):
    """geom.rs dist: hypot(a.x - b.x, a.y - b.y)."""
    return math.hypot(ax - bx, ay - by)


@jit
def norm(x, y):
    """geom.rs norm (a zero or NaN length counts as 1)."""
    ln = math.hypot(x, y)
    if ln == 0.0 or ln != ln:
        ln = 1.0
    return x / ln, y / ln


@jit
def polar(cx, cy, r, ang):
    """geom.rs polar: c + r (cos ang, sin ang)."""
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


@jit
def rotate_about(px, py, cx, cy, ang):
    """geom.rs rotate_about."""
    s = math.sin(ang)
    co = math.cos(ang)
    dx = px - cx
    dy = py - cy
    return cx + dx * co - dy * s, cy + dx * s + dy * co


@jit
def tpl_p(k, ph, x, y):
    """HumanVocalTract::p (human.rs:685-691, 1379-1385): template point -> body coordinates."""
    if x > 0.5:
        xx = 0.5 + (x - 0.5) * 0.86
    else:
        xx = x
    if y < 7.0:
        yy = 7.0 - (7.0 - y) * ph
    else:
        yy = y
    return xx * k, yy * k


@jit
def jaw_pt(k, ph, cx, cy, phi, x, y):
    """The jaw-carried point j(x, y) of compute (human.rs:736): p(x, y) rotated about the condyle by phi."""
    px, py = tpl_p(k, ph, x, y)
    return rotate_about(px, py, cx, cy, phi)


@jit
def tangent_angle(cx, cy, r, px, py, upper):
    """geom.rs tangent_angle -> (found, angle); found False is the template's None."""
    d = dist(cx, cy, px, py)
    if d <= r * 1.0001:
        return False, 0.0
    base = math.atan2(py - cy, px - cx)
    off = math.acos(r / d)
    if upper:
        return True, base + off
    return True, base - off


# ---- point buffers (Vec<Vec2> / Polyline as interleaved runs) ------------------------------------------------

@jit
def put(buf, base, i, x, y):
    """Point i of the run at ``base``."""
    buf[base + 2 * i] = x
    buf[base + 2 * i + 1] = y


@jit
def pts_append(dst, dbase, nd, src, sbase, ns, skip_first):
    """Polyline::append (geom.rs:77-82): src[min(skip_first, ns)..] after dst's nd points; returns the new count."""
    start = 1 if skip_first else 0
    if start > ns:
        start = ns
    for i in range(start, ns):
        dst[dbase + 2 * nd] = src[sbase + 2 * i]
        dst[dbase + 2 * nd + 1] = src[sbase + 2 * i + 1]
        nd += 1
    return nd


@jit
def bezier3(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, count, out, base):
    """geom.rs bezier3 (count points, both ends included); returns count."""
    for i in range(count):
        t = i / (count - 1.0)
        u = 1.0 - t
        a = u * u * u
        b = 3.0 * u * u * t
        c = 3.0 * u * t * t
        d = t * t * t
        out[base + 2 * i] = a * p0x + b * p1x + c * p2x + d * p3x
        out[base + 2 * i + 1] = a * p0y + b * p1y + c * p2y + d * p3y
    return count


@jit
def bezier2(p0x, p0y, p1x, p1y, p2x, p2y, count, out, base):
    """geom.rs bezier2; returns count."""
    for i in range(count):
        t = i / (count - 1.0)
        u = 1.0 - t
        out[base + 2 * i] = u * u * p0x + 2.0 * u * t * p1x + t * t * p2x
        out[base + 2 * i + 1] = u * u * p0y + 2.0 * u * t * p1y + t * t * p2y
    return count


@jit
def arc(cx, cy, r, a0, a1, count, out, base):
    """geom.rs arc: count points from angle a0 to a1; returns count."""
    for i in range(count):
        x, y = polar(cx, cy, r, a0 + ((a1 - a0) * i) / (count - 1.0))
        out[base + 2 * i] = x
        out[base + 2 * i + 1] = y
    return count


@jit
def catmull_rom(src, sbase, n, per, dst, dbase):
    """geom.rs catmull_rom (uniform, ``per`` samples per segment, last point appended); returns the count."""
    if n < 3:
        for i in range(n):
            dst[dbase + 2 * i] = src[sbase + 2 * i]
            dst[dbase + 2 * i + 1] = src[sbase + 2 * i + 1]
        return n
    m = 0
    for i in range(n - 1):
        i0 = i - 1 if i > 0 else 0
        i3 = i + 2 if i + 2 < n - 1 else n - 1
        p0x = src[sbase + 2 * i0]
        p0y = src[sbase + 2 * i0 + 1]
        p1x = src[sbase + 2 * i]
        p1y = src[sbase + 2 * i + 1]
        p2x = src[sbase + 2 * i + 2]
        p2y = src[sbase + 2 * i + 3]
        p3x = src[sbase + 2 * i3]
        p3y = src[sbase + 2 * i3 + 1]
        for kk in range(per):
            t = kk / per
            t2 = t * t
            t3 = t2 * t
            dst[dbase + 2 * m] = 0.5 * (2.0 * p1x + (-p0x + p2x) * t + (2.0 * p0x - 5.0 * p1x + 4.0 * p2x - p3x) * t2
                                        + (-p0x + 3.0 * p1x - 3.0 * p2x + p3x) * t3)
            dst[dbase + 2 * m + 1] = 0.5 * (2.0 * p1y + (-p0y + p2y) * t + (2.0 * p0y - 5.0 * p1y + 4.0 * p2y - p3y)
                                            * t2 + (-p0y + 3.0 * p1y - 3.0 * p2y + p3y) * t3)
            m += 1
    dst[dbase + 2 * m] = src[sbase + 2 * (n - 1)]
    dst[dbase + 2 * m + 1] = src[sbase + 2 * (n - 1) + 1]
    return m + 1


# ---- rays (geom.rs:106-160) ---------------------------------------------------------------------------------

@jit
def crossing_t(ox, oy, dx, dy, ax, ay, bx, by):
    """geom.rs crossing_t: where segment A->B crosses the line o + t d, or NaN."""
    sa = dx * (ay - oy) - dy * (ax - ox) >= 0.0
    sb = dx * (by - oy) - dy * (bx - ox) >= 0.0
    if sa == sb:
        return math.nan
    ex = bx - ax
    ey = by - ay
    den = dx * ey - dy * ex
    if abs(den) < 1e-15:
        return math.nan
    return ((ax - ox) * ey - (ay - oy) * ex) / den


@jit
def ray_polyline(ox, oy, dx, dy, pl, base, n, t_min, t_max, out, obase, cap):
    """geom.rs ray_polyline: crossing parameters t in [t_min, t_max] into out[obase:obase+cap]; returns count."""
    m = 0
    for i in range(n - 1):
        j = base + 2 * i
        t = crossing_t(ox, oy, dx, dy, pl[j], pl[j + 1], pl[j + 2], pl[j + 3])
        if t >= t_min and t <= t_max and m < cap:
            out[obase + m] = t
            m += 1
    return m


@jit
def ray_crossings(ox, oy, dx, dy, pl, base, n, t_min, t_max, out, obase, ebase, cap):
    """geom.rs ray_crossings: like ray_polyline, exit flags (1.0 / 0.0) at out[ebase + i]; returns count."""
    m = 0
    for i in range(n - 1):
        j = base + 2 * i
        t = crossing_t(ox, oy, dx, dy, pl[j], pl[j + 1], pl[j + 2], pl[j + 3])
        if t >= t_min and t <= t_max and m < cap:
            out[obase + m] = t
            if dx * (pl[j + 3] - pl[j + 1]) - dy * (pl[j + 2] - pl[j]) < 0.0:
                out[ebase + m] = 1.0
            else:
                out[ebase + m] = 0.0
            m += 1
    return m


# ---- roof helpers (human.rs:1398-1489) ------------------------------------------------------------------------

@jit
def project_on(pl, base, n, cum, cbase, px, py):
    """human.rs project_on -> (s, dist, signed, x, y, seg)."""
    best = math.inf
    bs = 0.0
    bx = 0.0
    by = 0.0
    sign = 1.0
    bseg = 0
    for i in range(n - 1):
        j = base + 2 * i
        ax = pl[j]
        ay = pl[j + 1]
        ex = pl[j + 2] - ax
        ey = pl[j + 3] - ay
        l2 = ex * ex + ey * ey
        if l2 < 1e-12:
            continue
        t = ((px - ax) * ex + (py - ay) * ey) / l2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx = ax + ex * t
        qy = ay + ey * t
        d = math.hypot(px - qx, py - qy)
        if d < best:
            best = d
            bs = cum[cbase + i] + t * math.sqrt(l2)
            bx = qx
            by = qy
            cross = ex * (py - ay) - ey * (px - ax)
            sign = 1.0 if cross < 0.0 else -1.0
            bseg = i
    return bs, best, sign * best, bx, by, bseg


@jit
def point_along(pl, base, n, cum, cbase, s):
    """human.rs point_along: the point at arc length s (n = cum.len())."""
    i = 0
    while i + 2 < n and cum[cbase + i + 1] < s:
        i += 1
    ln = cum[cbase + i + 1] - cum[cbase + i]
    if ln > 1e-9:
        u = nbm.clamp((s - cum[cbase + i]) / ln, 0.0, 1.0)
    else:
        u = 0.0
    j = base + 2 * i
    return pl[j] + (pl[j + 2] - pl[j]) * u, pl[j + 1] + (pl[j + 3] - pl[j + 1]) * u


@jit
def signed_gap(pl, base, n, px, py):
    """human.rs signed_gap: distance to a closed polygon's boundary, negative inside (even-odd)."""
    best = math.inf
    inside = False
    for i in range(n - 1):
        j = base + 2 * i
        ax = pl[j]
        ay = pl[j + 1]
        bx = pl[j + 2]
        by = pl[j + 3]
        ex = bx - ax
        ey = by - ay
        l2 = ex * ex + ey * ey
        if l2 > 1e-12:
            t = ((px - ax) * ex + (py - ay) * ey) / l2
        else:
            t = 0.0
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = px - ax - ex * t
        dy = py - ay - ey * t
        d2 = dx * dx + dy * dy
        if d2 < best:
            best = d2
        if (ay > py) != (by > py) and px < ax + ((py - ay) * ex) / ey:
            inside = not inside
    d = math.sqrt(best)
    return -d if inside else d


@jit
def closest_approach(pl, base, n, cum, cbase, pts, pbase, npts):
    """human.rs closest_approach -> (signed gap, normal x, normal y)."""
    d = math.inf
    seg = 0
    for i in range(npts):
        pr = project_on(pl, base, n, cum, cbase, pts[pbase + 2 * i], pts[pbase + 2 * i + 1])
        if pr[2] < d:
            d = pr[2]
            seg = pr[5]
    j = base + 2 * seg
    ex = pl[j + 2] - pl[j]
    ey = pl[j + 3] - pl[j + 1]
    ln = math.hypot(ex, ey)
    if ln == 0.0 or ln != ln:
        ln = 1.0
    return d, ey / ln, -ex / ln


# ---- ray measurement (the closures of compute, human.rs:943-1016) -------------------------------------------------

@jit
def fan_inner(frame, ibase, n_in, ox, oy, dx, dy, t_out, k):
    """compute's ``fan_inner`` closure (human.rs:943-984): inner surface parameter of an oral fan ray."""
    n = ray_crossings(ox, oy, dx, dy, frame, ibase, n_in, 0.0, t_out + 1.5 * k, frame, S_CROSS_T, S_CROSS_EXIT,
                      MAX_CROSS)
    # sort the crossings along the ray (stable insertion sort, human.rs:946-957)
    for i in range(1, n):
        t = frame[S_CROSS_T + i]
        e = frame[S_CROSS_EXIT + i]
        jj = i - 1
        while jj >= 0 and frame[S_CROSS_T + jj] > t:
            frame[S_CROSS_T + jj + 1] = frame[S_CROSS_T + jj]
            frame[S_CROSS_EXIT + jj + 1] = frame[S_CROSS_EXIT + jj]
            jj -= 1
        frame[S_CROSS_T + jj + 1] = t
        frame[S_CROSS_EXIT + jj + 1] = e
    i = 0
    if n > 0 and frame[S_CROSS_EXIT] != 0.0:
        t_in = frame[S_CROSS_T]
        i = 1
    else:
        back = 2.5 * k
        n_back = ray_polyline(ox, oy, -dx, -dy, frame, ibase, n_in, 0.0, back, frame, S_TMP, MAX_CROSS)
        tb = back
        for q in range(n_back):
            tb = nbm.js_min(tb, frame[S_TMP + q])
        t_in = -tb
    tissue = 0.0
    while i < n and frame[S_CROSS_T + i] < t_out:
        if frame[S_CROSS_EXIT + i] != 0.0:
            i += 1
            continue
        if i + 1 < n:
            bb = frame[S_CROSS_T + i + 1]
        else:
            bb = t_out + 1.5 * k
        tissue += nbm.js_min(bb, t_out) - frame[S_CROSS_T + i]
        i += 2
    return t_in + tissue


@jit
def measure(frame, obase, n_out, ibase, n_in, ox, oy, dx, dy, max_len, fan, k):
    """compute's ``measure`` closure (human.rs:985-1015) -> (found, inner x, inner y, outer x, outer y, d)."""
    n_o = ray_polyline(ox, oy, dx, dy, frame, obase, n_out, 0.0, max_len, frame, S_TMP, MAX_CROSS)
    if n_o == 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0
    t_out = math.inf
    for i in range(n_o):
        t_out = nbm.js_min(t_out, frame[S_TMP + i])
    if fan:
        t_in = fan_inner(frame, ibase, n_in, ox, oy, dx, dy, t_out, k)
    else:
        n_i = ray_polyline(ox, oy, dx, dy, frame, ibase, n_in, 0.0, t_out + 1.5 * k, frame, S_TMP, MAX_CROSS)
        t_in = -math.inf
        for i in range(n_i):
            ti = frame[S_TMP + i]
            if ti > t_in and ti < t_out + 1.5 * k:
                t_in = ti
        if t_in == -math.inf:
            t_in = nbm.js_max(0.0, t_out - 3.5 * k)
    d = nbm.js_max(0.0, t_out - t_in)
    tm = nbm.js_min(t_in, t_out)
    return True, ox + dx * tm, oy + dy * tm, ox + dx * t_out, oy + dy * t_out, d


@jit
def push_sample(frame, row, region, ix, iy, ox, oy, d):
    """compute's ``sample`` closure (human.rs:1016): a GridSample with area 0, s 0, ray = d."""
    b = F_GRID + row * NG
    frame[b + G_REGION] = region
    frame[b + G_D] = d
    frame[b + G_AREA] = 0.0
    frame[b + G_S] = 0.0
    frame[b + G_MID_X] = ix + (ox - ix) * 0.5
    frame[b + G_MID_Y] = iy + (oy - iy) * 0.5
    frame[b + G_IN_X] = ix
    frame[b + G_IN_Y] = iy
    frame[b + G_OUT_X] = ox
    frame[b + G_OUT_Y] = oy
    frame[b + G_RAY] = d


@jit
def dorsum_dip(px, py, x0, x1, dorsum, k):
    """compute's ``dip`` closure (human.rs:856-864)."""
    u = (px - x0) / nbm.js_max(1e-6, x1 - x0)
    if u <= 0.0 or u >= 1.0:
        return px, py
    w = math.sin(PI * math.pow(u, 0.8))
    return px + 0.0, py + -dorsum * 0.9 * k * w * w


# ---- tract constants (HumanVocalTract::new) -------------------------------------------------------------------

@jit
def _cp(buf, i, k, ph, x, y):
    """Control point i = p(x, y); returns i + 1."""
    px, py = tpl_p(k, ph, x, y)
    buf[2 * i] = px
    buf[2 * i + 1] = py
    return i + 1


@jit
def tract_init(scale, pharynx, tongue_radius, lip_width, tract):
    """HumanVocalTract::new (human.rs:644-679) into ``tract`` (float64[NTRACT])."""
    k = scale
    ph = pharynx
    tract[T_K] = k
    tract[T_PHARYNX] = ph
    tract[T_TONGUE_RADIUS] = tongue_radius
    tract[T_LIP_WIDTH] = lip_width
    tract[T_LAW_PH_A] = 2.0
    tract[T_LAW_PH_B] = 1.4
    tract[T_LAW_OR_A] = 2.6
    tract[T_LAW_OR_B] = 1.4
    tract[T_LAW_FR_A] = 2.0
    tract[T_LAW_FR_B] = 1.4
    tract[T_LAW_AH_A] = 1.6
    tract[T_LAW_AH_B] = 1.4
    cp = np.empty(2 * 16)
    # posterior pharyngeal wall, bottom -> top
    n = _cp(cp, 0, k, ph, -1.0, -1.5)
    n = _cp(cp, n, k, ph, -1.1, 1.5)
    n = _cp(cp, n, k, ph, -1.2, 3.6)
    n = _cp(cp, n, k, ph, -1.3, 5.6)
    n = _cp(cp, n, k, ph, -1.38, 7.5)
    n = _cp(cp, n, k, ph, -1.33, 8.9)
    n = _cp(cp, n, k, ph, -1.05, 10.1)
    n = _cp(cp, n, k, ph, -0.45, 10.9)
    n = _cp(cp, n, k, ph, 0.4, 11.3)
    tract[T_N_PHAR] = catmull_rom(cp, 0, n, 4, tract, T_PHAR)
    pns_x, pns_y = tpl_p(k, ph, 1.0, 9.75)
    n = _cp(cp, 0, k, ph, 1.0, 9.6)
    n = _cp(cp, n, k, ph, 1.8, 9.92)
    n = _cp(cp, n, k, ph, 2.7, 10.18)
    n = _cp(cp, n, k, ph, 3.6, 10.3)
    n = _cp(cp, n, k, ph, 4.4, 10.25)
    n = _cp(cp, n, k, ph, 5.1, 10.05)
    n = _cp(cp, n, k, ph, 5.7, 9.72)
    n = _cp(cp, n, k, ph, 6.15, 9.36)
    n = _cp(cp, n, k, ph, 6.5, 9.05)
    n = _cp(cp, n, k, ph, 6.78, 8.7)
    n = _cp(cp, n, k, ph, 6.98, 8.3)
    n = _cp(cp, n, k, ph, 7.12, 7.95)
    tract[T_N_PALATE] = catmull_rom(cp, 0, n, 3, tract, T_PALATE)
    qx, qy = tpl_p(k, ph, -2.5, 10.5)
    tract[T_CONDYLE_X] = qx
    tract[T_CONDYLE_Y] = qy
    qx, qy = tpl_p(k, ph, 2.6, 6.95)
    tract[T_POLAR_C_X] = qx
    tract[T_POLAR_C_Y] = qy
    qx, qy = tpl_p(k, ph, 5.0, 6.85)
    tract[T_POLAR_C2_X] = qx
    tract[T_POLAR_C2_Y] = qy
    tract[T_PNS_X] = pns_x
    tract[T_PNS_Y] = pns_y
    cp[0] = pns_x
    cp[1] = pns_y
    n = _cp(cp, 1, k, ph, 2.5, 10.8)
    n = _cp(cp, n, k, ph, 5.0, 11.2)
    n = _cp(cp, n, k, ph, 7.3, 10.6)
    n = _cp(cp, n, k, ph, 8.0, 10.1)
    tract[T_N_NASAL] = catmull_rom(cp, 0, n, 4, tract, T_NASAL)


# ---- compute (HumanVocalTract::compute, human.rs:725-1263) -------------------------------------------------------

@jit
def walls(artic, tract, frame):
    """First part of HumanVocalTract::compute (human.rs:725-922): jaw, velum, outer wall, lips, tongue and the
    inner wall into the frame; returns the scalars the later parts need."""
    k = tract[T_K]
    ph = tract[T_PHARYNX]
    cdx = tract[T_CONDYLE_X]
    cdy = tract[T_CONDYLE_Y]
    n_palate = int(tract[T_N_PALATE])
    pal0x = tract[T_PALATE]
    pal0y = tract[T_PALATE + 1]
    inc_x = tract[T_PALATE + 2 * (n_palate - 1)]
    inc_y = tract[T_PALATE + 2 * (n_palate - 1) + 1]
    obase = F_OUTLINES + O_OUTER * 2 * MAX_PTS
    ibase = F_OUTLINES + O_INNER * 2 * MAX_PTS

    lar = artic[A_LARYNX] * k
    gx, gy = tpl_p(k, ph, 0.0, 0.0)
    glottis_x = gx + 0.0
    glottis_y = gy + lar

    # ---- jaw (rotation about the condyle)
    jrx, jry = tpl_p(k, ph, 6.95, 7.75)
    jaw_r = dist(cdx, cdy, jrx, jry)
    phi = -nbm.clamp(artic[A_JAW], -0.2, 3.0) * k / jaw_r

    # ---- velum flap from PNS; angle from port area
    pns_x = tract[T_PNS_X]
    pns_y = tract[T_PNS_Y]
    v_open = nbm.clamp(artic[A_VELUM] / 1.2, 0.0, 1.0)
    v_ang = (194.0 + 46.0 * v_open) * DEG
    v_len = 2.8 * k
    for i in range(11):
        s = i / 10.0
        sag = math.sin(s * PI) * k * (0.3 * v_open - 0.12)
        px, py = polar(pns_x, pns_y, v_len * s, v_ang)
        cx = px + 0.0
        cy = py + -sag
        nx = -math.sin(v_ang)
        ny = math.cos(v_ang)
        th = (0.48 - 0.2 * s) * k
        put(frame, S_VEL_ORAL, i, cx + nx * th, cy + ny * th)
        put(frame, S_VEL_NASAL, i, cx - nx * th, cy - ny * th)
        put(frame, S_VEL, i, cx, cy)
    for i in range(11):
        u = i / 10.0 / 0.35
        if u < 1.0:
            w = (1.0 - u) * (1.0 - u)
            b = S_VEL_ORAL + 2 * i
            frame[b] = frame[b] + (pal0x - frame[b]) * w
            frame[b + 1] = frame[b + 1] + (pal0y - frame[b + 1]) * w

    # ---- outer contour: pharyngeal wall up to just below the uvula, velum oral surface (tip -> PNS), palate,
    # upper lip
    n_out = 0
    uv_y = frame[S_VEL_ORAL + 2 * 10 + 1] - 0.3 * k
    n_phar = int(tract[T_N_PHAR])
    for i in range(n_phar):
        px = tract[T_PHAR + 2 * i]
        py = tract[T_PHAR + 2 * i + 1]
        if py < uv_y:
            put(frame, obase, n_out, px, py)
            n_out += 1
            continue
        if i > 0:
            ax = tract[T_PHAR + 2 * i - 2]
            ay = tract[T_PHAR + 2 * i - 1]
            t = nbm.clamp((uv_y - ay) / nbm.js_max(1e-9, py - ay), 0.0, 1.0)
            put(frame, obase, n_out, ax + (px - ax) * t, ay + (py - ay) * t)
            n_out += 1
        break
    roof_start = n_out
    for i in range(10, -1, -1):
        put(frame, obase, n_out, frame[S_VEL_ORAL + 2 * i], frame[S_VEL_ORAL + 2 * i + 1])
        n_out += 1
    n_out = pts_append(frame, obase, n_out, tract, T_PALATE, n_palate, True)
    roof_end = n_out - 1
    # upper lip
    prot = nbm.clamp(artic[A_LIP_PROT], -0.2, 1.6) * k
    spread_w = nbm.clamp(artic[A_LIP_WIDTH], 0.0, 1.0)
    y_mid = (7.85 - 0.3 * artic[A_JAW]) * k
    lab = nbm.clamp(artic[A_LABIODENTAL], 0.0, 1.0)
    ap = artic[A_LIP_AP] * k
    lfx, lfy = tpl_p(k, ph, 8.15 + 0.12 * (1.0 - spread_w), 0.0)
    lip_front_x = lfx + prot * 0.95
    ul_x = lip_front_x - 0.15 * k
    ul_y = y_mid + ap * 0.35 + lab * 0.35 * k
    ll_x = lip_front_x - 0.25 * k
    ll_y = y_mid - ap * 0.65
    if lab > 0.0:
        # lower lip tucks under the upper incisor edge for /f v/
        tgx = inc_x + 0.05 * k
        tgy = inc_y - nbm.js_max(artic[A_LIP_AP] * k, -0.1)
        ll_x = ll_x + (tgx - ll_x) * lab
        ll_y = ll_y + (tgy - ll_y) * lab
        ux = inc_x + 0.55 * k
        uy = inc_y + 0.45 * k
        ul_x = ul_x + (ux - ul_x) * (lab * 0.7)
        ul_y = ul_y + (uy - ul_y) * (lab * 0.7)
    ubx, uby = tpl_p(k, ph, 7.32, 8.35)
    bezier3(ubx, uby, ubx + (0.3 * k + prot * 0.4), uby + 0.05 * k, ul_x + -0.35 * k, ul_y + 0.12 * k, ul_x, ul_y,
            7, frame, S_ULIP_IN)
    n_out = pts_append(frame, obase, n_out, frame, S_ULIP_IN, 7, False)

    # ---- tongue
    r = tract[T_TONGUE_RADIUS] * k
    bx, by = tpl_p(k, ph, artic[A_TB_X], artic[A_TB_Y])
    b0x, b0y = tpl_p(k, ph, NEUTRAL_TB_X, NEUTRAL_TB_Y)
    vx, vy = tpl_p(k, ph, 0.75, 3.3)
    val_x = vx + (0.45 * (bx - b0x) + artic[A_ROOT] * k)
    val_y = vy + (0.3 * (by - b0y) + 0.55 * lar)
    ex, ey = tpl_p(k, ph, 0.3, 2.4)
    epi_base_x = ex + 0.0
    epi_base_y = ey + lar
    ex, ey = tpl_p(k, ph, 0.0, 4.5)
    epi_tip_x = ex + 0.35 * (val_x - 0.75 * k)
    epi_tip_y = ey + (lar * 0.8 + 0.15 * (by - b0y))
    th_b = 200.0 * DEG
    pbx, pby = polar(bx, by, r, th_b)
    tan_bx = math.sin(th_b)
    tan_by = -math.cos(th_b)
    h_r = dist(val_x, val_y, pbx, pby) / 3.0
    bezier3(val_x, val_y, val_x + 0.08 * k, val_y + h_r, pbx - tan_bx * h_r, pby - tan_by * h_r, pbx, pby, 9,
            frame, S_ROOT)
    tip_x, tip_y = tpl_p(k, ph, artic[A_TT_X], artic[A_TT_Y])
    # keep tip outside the body circle
    d_t = dist(tip_x, tip_y, bx, by)
    if d_t < r * 1.05:
        nx, ny = norm(tip_x - bx, tip_y - by)
        tip_x = bx + nx * (r * 1.05)
        tip_y = by + ny * (r * 1.05)
    found, th_f = tangent_angle(bx, by, r, tip_x, tip_y, True)
    if not found:
        th_f = math.atan2(tip_y - by, tip_x - bx)
    # arc runs from thB (200 deg) decreasing to thF
    while th_f > th_b:
        th_f -= 2.0 * PI
    while th_f < th_b - 2.0 * PI:
        th_f += 2.0 * PI
    arc(bx, by, r, th_b, th_f, ARC_POINTS, frame, S_ARC)
    pfx = frame[S_ARC + 2 * (ARC_POINTS - 1)]
    pfy = frame[S_ARC + 2 * (ARC_POINTS - 1) + 1]
    tan_fx = math.sin(th_f)
    tan_fy = -math.cos(th_f)
    # blade: quadratic from the tangent point to the tip; curl lifts the tip and turns it back
    bl = dist(pfx, pfy, tip_x, tip_y)
    curl = nbm.clamp(artic[A_TT_CURL], 0.0, 1.0)
    ctrl_x = pfx + tan_fx * (bl * 0.5) + 0.0
    ctrl_y = pfy + tan_fy * (bl * 0.5) + (0.18 + 0.5 * curl) * k
    bezier2(pfx, pfy, ctrl_x, ctrl_y, tip_x, tip_y, 7, frame, S_BLADE)
    dorsum = artic[A_DORSUM]
    if dorsum > 0.0:
        x0 = bx
        x1 = tip_x
        for i in range(ARC_POINTS):
            qx, qy = dorsum_dip(frame[S_ARC + 2 * i], frame[S_ARC + 2 * i + 1], x0, x1, dorsum, k)
            put(frame, S_ARC, i, qx, qy)
        for i in range(1, 7 - 1):
            qx, qy = dorsum_dip(frame[S_BLADE + 2 * i], frame[S_BLADE + 2 * i + 1], x0, x1, dorsum, k)
            put(frame, S_BLADE, i, qx, qy)
    # underside back to the frenulum / floor of mouth (jaw-carried)
    frx, fry = jaw_pt(k, ph, cdx, cdy, phi, 5.7, 6.15)
    bezier2(tip_x, tip_y, tip_x + (-0.3 * k - curl * 0.4 * k), tip_y + -0.6 * k, frx, fry, 6, frame, S_UNDER)
    put(frame, S_FLOOR, 0, frx, fry)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 6.2, 6.25)
    put(frame, S_FLOOR, 1, qx, qy)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 6.45, 6.55)
    put(frame, S_FLOOR, 2, qx, qy)
    # lower incisors (jaw)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 6.45, 6.55)
    put(frame, S_LTEETH, 0, qx, qy)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 6.72, 7.15)
    put(frame, S_LTEETH, 1, qx, qy)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 6.95, 7.75)
    put(frame, S_LTEETH, 2, qx, qy)
    qx, qy = jaw_pt(k, ph, cdx, cdy, phi, 7.12, 7.45)
    put(frame, S_LTEETH, 3, qx, qy)
    # lower lip: from labial base of lower incisor to the edge
    lbx, lby = jaw_pt(k, ph, cdx, cdy, phi, 7.12, 7.45)
    bezier3(lbx, lby, lbx + (0.25 * k + prot * 0.3), lby + -0.05 * k, ll_x + -0.3 * k, ll_y + -0.12 * k, ll_x, ll_y,
            7, frame, S_LLIP_IN)

    # acoustic inner wall: epiglottis, then the tongue root above the epiglottis tip
    n_in = 0
    put(frame, ibase, 0, epi_base_x, epi_base_y)
    put(frame, ibase, 1, epi_tip_x, epi_tip_y)
    n_in = 2
    y_root = epi_tip_y + 0.05 * k
    n_back = pts_append(frame, S_BACK, 0, frame, S_ROOT, 9, False)
    n_back = pts_append(frame, S_BACK, n_back, frame, S_ARC, ARC_POINTS, True)
    root_len = 9
    start = -1
    for q in range(n_back):
        if frame[S_BACK + 2 * q + 1] > y_root:
            start = q
            break
    if start < 0:
        start = n_back - 1
    elif start > 0:
        # start exactly at that height
        ax = frame[S_BACK + 2 * (start - 1)]
        ay = frame[S_BACK + 2 * (start - 1) + 1]
        bbx = frame[S_BACK + 2 * start]
        bby = frame[S_BACK + 2 * start + 1]
        t = nbm.clamp((y_root - ay) / nbm.js_max(1e-9, bby - ay), 0.0, 1.0)
        put(frame, ibase, n_in, ax + (bbx - ax) * t, ay + (bby - ay) * t)
        n_in += 1
    if start < root_len:
        for i in range(start, root_len):
            put(frame, ibase, n_in, frame[S_ROOT + 2 * i], frame[S_ROOT + 2 * i + 1])
            n_in += 1
        n_in = pts_append(frame, ibase, n_in, frame, S_ARC, ARC_POINTS, True)
    else:
        off = start - root_len + 1
        n_in = pts_append(frame, ibase, n_in, frame, S_ARC + 2 * off, ARC_POINTS - off, True)
    n_in = pts_append(frame, ibase, n_in, frame, S_BLADE, 7, True)
    n_in = pts_append(frame, ibase, n_in, frame, S_UNDER, 6, True)
    n_in = pts_append(frame, ibase, n_in, frame, S_FLOOR, 3, True)
    n_in = pts_append(frame, ibase, n_in, frame, S_LTEETH, 4, True)
    n_in = pts_append(frame, ibase, n_in, frame, S_LLIP_IN, 7, False)
    return (n_out, n_in, roof_start, roof_end, inc_x, ul_x, ul_y, ll_x, ll_y, lab, ap, spread_w, tip_x, tip_y, bx, by,
            r, phi, glottis_x, glottis_y)


@jit
def grid_rays(tract, frame, n_out, n_in, glottis_y, inc_x, ul_x, ll_x, ll_y, lab):
    """The measurement grid of compute (human.rs:924-1067): pharyngeal horizontals, oral fan, front verticals;
    returns (n_grid, split_index, teeth_idx)."""
    k = tract[T_K]
    ph = tract[T_PHARYNX]
    obase = F_OUTLINES + O_OUTER * 2 * MAX_PTS
    ibase = F_OUTLINES + O_INNER * 2 * MAX_PTS
    # ---- grid
    n_grid = 0
    y_epi_top = glottis_y + 2.4 * k * ph
    y_phar_top = tract[T_POLAR_C_Y]
    split_index = -1
    # pharyngeal horizontals (origin in front, looking back)
    for i in range(N_PH_RAYS):
        y = lerp(y_epi_top + 0.12 * k, y_phar_top, i / N_PH_RAYS)
        m = measure(frame, obase, n_out, ibase, n_in, 4.6 * k, y, -1.0, 0.0, 9.0 * k, False, k)
        if m[0]:
            push_sample(frame, n_grid, 1.0, m[1], m[2], m[3], m[4], m[5])
            n_grid += 1
    # oral family: rays from a centre gliding from mid-oral to behind the incisors
    pcx = tract[T_POLAR_C_X]
    pcy = tract[T_POLAR_C_Y]
    pc2x = tract[T_POLAR_C2_X]
    pc2y = tract[T_POLAR_C2_Y]
    n_rad = 30
    for i in range(n_rad + 1):
        t = i / n_rad
        ang = lerp(180.0, 36.0, t) * DEG
        if t < 0.45:
            w = 0.0
        else:
            w = (t - 0.45) / 0.55
        sw = w * w * (3.0 - 2.0 * w)
        cx = pcx + (pc2x - pcx) * sw
        cy = pcy + (pc2y - pcy) * sw
        m = measure(frame, obase, n_out, ibase, n_in, cx, cy, math.cos(ang), math.sin(ang), 8.0 * k, True, k)
        if not m[0]:
            continue
        if split_index < 0 and ang <= 150.0 * DEG:
            split_index = n_grid
        region = 1.0 if split_index < 0 else 2.0
        push_sample(frame, n_grid, region, m[1], m[2], m[3], m[4], m[5])
        n_grid += 1
    # front verticals (upper incisor edge -> lip opening), looking up
    x_start = inc_x + 0.03 * k
    x_end = nbm.js_max(x_start + 0.3 * k, nbm.js_min(ul_x, ll_x) - 0.04 * k)
    n_fr = 6
    teeth_idx = n_grid
    for i in range(n_fr):
        x = lerp(x_start, x_end, i / (n_fr - 1.0))
        m = measure(frame, obase, n_out, ibase, n_in, x, 4.0 * k, 0.0, 1.0, 8.0 * k, False, k)
        if m[0]:
            mix = m[1]
            miy = m[2]
            mox = m[3]
            moy = m[4]
            md = m[5]
            if x > ll_x:
                face = nbm.js_min(ll_y - (x - ll_x) * 4.0 / 3.0, moy)
                opn = nbm.smoothstep(LAB_FRONT_OPENS, 1.0, lab) * nbm.smoothstep(0.0, LAB_FRONT_RAMP * k, x - ll_x)
                y_in = lerp(face, moy - 3.5 * k, opn)
                mix = x
                miy = y_in
                md = moy - y_in
            push_sample(frame, n_grid, 3.0, mix, miy, mox, moy, md)
            n_grid += 1
    return n_grid, split_index, teeth_idx


@jit
def contact_model(frame, n_grid, roof_start, roof_end, k):
    """The soft-tissue contact model of compute (human.rs:1071-1151): roof contacts of tip and body, and the
    gap of every roof ray to the tongue surface; returns (tip gap, tip normal, body gap, body normal)."""
    obase = F_OUTLINES + O_OUTER * 2 * MAX_PTS
    # ---- soft-tissue contact model (human.rs:1071-1151)
    rbase = obase + 2 * roof_start
    n_roof = roof_end - roof_start + 1
    frame[S_CUM] = 0.0
    for i in range(1, n_roof):
        j = rbase + 2 * i
        frame[S_CUM + i] = frame[S_CUM + i - 1] + math.hypot(frame[j] - frame[j - 2], frame[j + 1] - frame[j - 1])
    n_tip = pts_append(frame, S_TIP_PTS, 0, frame, S_BLADE + 2 * 2, 7 - 2, False)
    n_tip = pts_append(frame, S_TIP_PTS, n_tip, frame, S_UNDER + 2 * 1, 2, False)
    tip_c = closest_approach(frame, rbase, n_roof, frame, S_CUM, frame, S_TIP_PTS, n_tip)
    body_c = closest_approach(frame, rbase, n_roof, frame, S_CUM, frame, S_ARC, ARC_POINTS)
    # the whole tongue outline, closed
    n_ol = pts_append(frame, S_OUTLINE, 0, frame, S_ROOT, 9, False)
    n_ol = pts_append(frame, S_OUTLINE, n_ol, frame, S_ARC, ARC_POINTS, True)
    n_ol = pts_append(frame, S_OUTLINE, n_ol, frame, S_BLADE, 7, True)
    n_ol = pts_append(frame, S_OUTLINE, n_ol, frame, S_UNDER, 6, True)
    n_ol = pts_append(frame, S_OUTLINE, n_ol, frame, S_FLOOR, 3, True)
    put(frame, S_OUTLINE, n_ol, frame[S_ROOT], frame[S_ROOT + 1])
    n_ol += 1
    press = 1.0 - nbm.smoothstep(-0.06 * k, 0.0, nbm.js_min(tip_c[0], body_c[0]))
    half = 0.1 * k * (1.0 + press)
    roof_len = frame[S_CUM + n_roof - 1]
    for row in range(n_grid):
        gb = F_GRID + row * NG
        if frame[gb + G_REGION] == 0.0:
            continue
        gox = frame[gb + G_OUT_X]
        goy = frame[gb + G_OUT_Y]
        pr = project_on(frame, rbase, n_roof, frame, S_CUM, gox, goy)
        on_roof = 1.0 - nbm.smoothstep(0.15 * k, 0.25 * k, pr[1])
        if on_roof <= 0.0:
            continue
        dc = signed_gap(frame, S_OUTLINE, n_ol, gox, goy)
        if dc - half - pr[1] <= 0.6 * k:
            for fr in (-1.0, -0.5, 0.5, 1.0):
                qx, qy = point_along(frame, rbase, n_roof, frame, S_CUM, nbm.clamp(pr[0] + fr * half, 0.0, roof_len))
                dc = nbm.js_min(dc, signed_gap(frame, S_OUTLINE, n_ol, qx, qy))
        w = (1.0 - nbm.smoothstep(0.15 * k, 0.6 * k, dc)) * on_roof
        cap = (1.0 - nbm.smoothstep(0.45 * k, 0.6 * k, dc)) * on_roof
        if w <= 0.0 and cap <= 0.0:
            continue
        dp = nbm.js_max(0.0, dc)
        gd = frame[gb + G_D]
        dm = gd + (nbm.js_min(gd, dp) - gd) * cap
        frame[gb + G_D] = dm + (dp - dm) * w
    return tip_c[0], tip_c[1], tip_c[2], body_c[0], body_c[1], body_c[2]


@jit
def grid_areas(artic, tract, frame, n_grid, split_index, teeth_idx, inc_x, ul_x, ll_x, lab, ap, spread_w,
               tip_x, tip_y):
    """Lip contact, area law, spike clamp, lateral channel and centreline of compute (human.rs:1153-1226)."""
    k = tract[T_K]
    ph = tract[T_PHARYNX]
    lip_w = (1.1 + (tract[T_LIP_WIDTH] - 1.1) * spread_w) * k
    # lip contact (human.rs:1153-1172)
    edge_x = nbm.js_min(ul_x, ll_x)
    rl = 0.6 * k
    lab_open = nbm.smoothstep(0.0, LAB_HANDOVER, lab)
    if lab_open >= 1.0:
        ap_eff = math.inf
    else:
        ap_eff = ap + LAB_GAP * k * lab_open / (1.0 - lab_open)
    for row in range(n_grid):
        gb = F_GRID + row * NG
        if frame[gb + G_REGION] != 3.0 or frame[gb + G_MID_X] < inc_x:
            continue
        dxl = nbm.js_max(0.0, edge_x - frame[gb + G_MID_X])
        frame[gb + G_D] = nbm.js_min(frame[gb + G_D], nbm.js_max(0.0, ap_eff + (dxl * dxl) / (2.0 * rl)))

    # ---- areas
    a_p = tract[T_LAW_PH_A]
    b_p = tract[T_LAW_PH_B]
    a_o = tract[T_LAW_OR_A]
    b_o = tract[T_LAW_OR_B]
    a_f = tract[T_LAW_FR_A]
    b_f = tract[T_LAW_FR_B]
    a_h = tract[T_LAW_AH_A]
    b_h = tract[T_LAW_AH_B]
    teeth_x = inc_x
    for row in range(n_grid):
        gb = F_GRID + row * NG
        reg = frame[gb + G_REGION]
        gd = frame[gb + G_D]
        if reg == 1.0:
            frame[gb + G_AREA] = a_p * math.pow(gd, b_p)
        elif reg == 2.0:
            ahead = nbm.smoothstep(0.0, 0.4 * k, frame[gb + G_MID_X] - tip_x)
            frame[gb + G_AREA] = lerp(a_o * math.pow(gd, b_o), a_h * math.pow(gd, b_h), ahead)
        else:
            u = nbm.clamp((frame[gb + G_MID_X] - teeth_x) / (0.45 * k), 0.0, 1.0)
            dental = a_f * math.pow(gd, b_f)
            lip = (PI / 4.0) * gd * lip_w
            frame[gb + G_AREA] = lerp(dental, lip, u)
    # suppress isolated spikes: clamp each open sample to 1.35x the larger of its neighbours
    for i in range(1, n_grid - 1):
        gb = F_GRID + i * NG
        lim = 1.35 * nbm.js_max(frame[gb - NG + G_AREA], frame[gb + NG + G_AREA])
        if frame[gb + G_AREA] > lim:
            frame[gb + G_AREA] = lim
    # lateral channel (for /l/)
    lateral = artic[A_LATERAL]
    if lateral > 0.0:
        for row in range(n_grid):
            gb = F_GRID + row * NG
            if frame[gb + G_REGION] >= 2.0 and dist(frame[gb + G_MID_X], frame[gb + G_MID_Y], tip_x, tip_y) < 1.4 * k:
                frame[gb + G_AREA] = nbm.js_max(frame[gb + G_AREA], lateral * 0.35 * k * k)

    # ---- centreline s
    epi_len = 2.4 * k * ph
    s = epi_len
    for i in range(n_grid):
        gb = F_GRID + i * NG
        if i > 0:
            s += dist(frame[gb + G_MID_X], frame[gb + G_MID_Y], frame[gb - NG + G_MID_X], frame[gb - NG + G_MID_Y])
        frame[gb + G_S] = s
    last = n_grid - 1
    if teeth_idx < n_grid:
        frame[F_TEETH_S] = frame[F_GRID + teeth_idx * NG + G_S]
    else:
        frame[F_TEETH_S] = frame[F_GRID + last * NG + G_S]
    lip_s = frame[F_GRID + last * NG + G_S]
    frame[F_LIP_S] = lip_s
    if split_index > 0:
        frame[F_PHARYNX_LENGTH] = frame[F_GRID + split_index * NG + G_S]
    else:
        frame[F_PHARYNX_LENGTH] = frame[F_GRID + N_PH_RAYS * NG + G_S]
    frame[F_LENGTH] = lip_s
    frame[F_LIP_AREA] = frame[F_GRID + last * NG + G_AREA]


@jit
def finish(artic, tract, frame, n_out, n_in, n_grid, split_index, ul_x, ul_y, ll_x, ll_y, phi, r, tip_x, tip_y,
           bx, by, glottis_x, glottis_y, tip_d, tip_nx, tip_ny, body_d, body_nx, body_ny):
    """Drawing outlines and frame scalars of compute (human.rs:1228-1262)."""
    k = tract[T_K]
    ph = tract[T_PHARYNX]
    cdx = tract[T_CONDYLE_X]
    cdy = tract[T_CONDYLE_Y]
    # ---- drawing parts (human.rs:1228-1262)
    tbase = F_OUTLINES + O_TONGUE * 2 * MAX_PTS
    n_t = pts_append(frame, tbase, 0, frame, S_ROOT, 9, False)
    n_t = pts_append(frame, tbase, n_t, frame, S_ARC + 2, ARC_POINTS - 1, False)
    n_t = pts_append(frame, tbase, n_t, frame, S_BLADE + 2, 7 - 1, False)
    n_t = pts_append(frame, tbase, n_t, frame, S_UNDER + 2, 6 - 1, False)
    n_t = pts_append(frame, tbase, n_t, frame, S_FLOOR + 2, 3 - 1, False)
    ulbase = F_OUTLINES + O_UPPER_LIP * 2 * MAX_PTS
    n_ul = pts_append(frame, ulbase, 0, frame, S_ULIP_IN, 7, False)
    p2x, p2y = tpl_p(k, ph, 8.3 + 0.0, 9.6)
    p3x, p3y = tpl_p(k, ph, 7.9, 10.4)
    bezier3(ul_x, ul_y, ul_x + 0.25 * k, ul_y + 0.35 * k, p2x, p2y, p3x, p3y, 6, frame, S_LIP_OUT)
    n_ul = pts_append(frame, ulbase, n_ul, frame, S_LIP_OUT + 2, 6 - 1, False)
    llbase = F_OUTLINES + O_LOWER_LIP * 2 * MAX_PTS
    n_ll = pts_append(frame, llbase, 0, frame, S_LLIP_IN, 7, False)
    p2x, p2y = jaw_pt(k, ph, cdx, cdy, phi, 8.1, 6.3)
    p3x, p3y = jaw_pt(k, ph, cdx, cdy, phi, 7.6, 5.6)
    bezier3(ll_x, ll_y, ll_x + 0.3 * k, ll_y + -0.4 * k, p2x, p2y, p3x, p3y, 6, frame, S_LIP_OUT)
    n_ll = pts_append(frame, llbase, n_ll, frame, S_LIP_OUT + 2, 6 - 1, False)
    vbase = F_OUTLINES + O_VELUM * 2 * MAX_PTS
    n_v = pts_append(frame, vbase, 0, frame, S_VEL_NASAL, 11, False)
    for i in range(10, -1, -1):
        put(frame, vbase, n_v, frame[S_VEL_ORAL + 2 * i], frame[S_VEL_ORAL + 2 * i + 1])
        n_v += 1

    # ---- scalars
    frame[F_VELUM_PORT] = nbm.js_max(0.0, artic[A_VELUM]) * k * k
    frame[F_TONGUE_RADIUS] = r
    frame[F_TIP_X] = tip_x
    frame[F_TIP_Y] = tip_y
    frame[F_SPLIT_INDEX] = float(split_index)
    frame[F_CONTACT_TIP] = tip_d
    frame[F_CONTACT_BODY] = body_d
    frame[F_CONTACT_TIP_NX] = tip_nx
    frame[F_CONTACT_TIP_NY] = tip_ny
    frame[F_CONTACT_BODY_NX] = body_nx
    frame[F_CONTACT_BODY_NY] = body_ny
    frame[F_GLOTTIS_X] = glottis_x
    frame[F_GLOTTIS_Y] = glottis_y
    frame[F_TONGUE_CX] = bx
    frame[F_TONGUE_CY] = by
    frame[F_N_GRID] = float(n_grid)
    frame[F_N_OUTLINE + O_OUTER] = float(n_out)
    frame[F_N_OUTLINE + O_INNER] = float(n_in)
    frame[F_N_OUTLINE + O_TONGUE] = float(n_t)
    frame[F_N_OUTLINE + O_VELUM] = float(n_v)
    frame[F_N_OUTLINE + O_UPPER_LIP] = float(n_ul)
    frame[F_N_OUTLINE + O_LOWER_LIP] = float(n_ll)


@jit
def geometry_compute(artic, tract, frame):
    """HumanVocalTract::compute (human.rs:725-1263) for ``artic`` (float64[17], ARTIC_KEYS order) on the body
    whose constants are ``tract`` (float64[NTRACT]); writes grid, scalars, contacts and outlines into ``frame``
    (float64[NFRAME]). Stateless apart from ``frame``: the contacts the template keeps on the tract are frame
    scalars here."""
    w = walls(artic, tract, frame)
    n_out, n_in, roof_start, roof_end, inc_x, ul_x, ul_y, ll_x, ll_y, lab, ap, spread_w = w[:12]
    tip_x, tip_y, bx, by, r, phi, glottis_x, glottis_y = w[12:]
    n_grid, split_index, teeth_idx = grid_rays(tract, frame, n_out, n_in, glottis_y, inc_x, ul_x, ll_x, ll_y, lab)
    c = contact_model(frame, n_grid, roof_start, roof_end, tract[T_K])
    grid_areas(artic, tract, frame, n_grid, split_index, teeth_idx, inc_x, ul_x, ll_x, lab, ap, spread_w, tip_x,
               tip_y)
    finish(artic, tract, frame, n_out, n_in, n_grid, split_index, ul_x, ul_y, ll_x, ll_y, phi, r, tip_x, tip_y, bx,
           by, glottis_x, glottis_y, c[0], c[1], c[2], c[3], c[4], c[5])


# ---- resample (human.rs:1265-1366) ------------------------------------------------------------------------------

@jit
def epi_profile(x, vent, top):
    """epilarynx_area's ``f`` closure: ventricle, ventricular folds, then the tube to its top."""
    if x < 0.3:
        return 1.3
    if x < 0.5:
        return vent
    return lerp(1.2, top, (x - 0.5) / 0.5)


@jit
def epilarynx_area(artic, k, s, epi_len):
    """HumanVocalTract::epilarynx_area (human.rs:1266-1282) at s in [0, epi_len]."""
    u = s / epi_len
    vent = 0.95 * (1.0 - 0.75 * nbm.clamp(artic[A_VENTRICULAR], 0.0, 1.0))
    top = lerp(1.9, 0.35, nbm.clamp(artic[A_EPILARYNX], 0.0, 1.0))
    r = 0.03
    area = epi_profile(u, vent, top)
    for u0 in (0.3, 0.5):
        if abs(u - u0) < r:
            area = lerp(epi_profile(u0 - r, vent, top), epi_profile(u0 + r, vent, top), (u - (u0 - r)) / (2.0 * r))
    return area * k * k


@jit
def area_at(artic, k, frame, n_grid, s, epi_len, flare):
    """resample's ``area_at`` closure (human.rs:1293-1319)."""
    if s <= epi_len - flare:
        return epilarynx_area(artic, k, s, epi_len)
    if s <= epi_len:
        return lerp(epilarynx_area(artic, k, s, epi_len), frame[F_GRID + G_AREA], (s - (epi_len - flare)) / flare)
    if s <= frame[F_GRID + G_S]:
        return frame[F_GRID + G_AREA]
    # binary search
    lo = 0
    hi = n_grid - 1
    if s >= frame[F_GRID + hi * NG + G_S]:
        return frame[F_GRID + hi * NG + G_AREA]
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if frame[F_GRID + m * NG + G_S] <= s:
            lo = m
        else:
            hi = m
    s_lo = frame[F_GRID + lo * NG + G_S]
    t = (s - s_lo) / nbm.js_max(1e-9, frame[F_GRID + hi * NG + G_S] - s_lo)
    return lerp(frame[F_GRID + lo * NG + G_AREA], frame[F_GRID + hi * NG + G_AREA], t)


@jit
def resample_fill(artic, k, frame, n_grid, s0, s1, out, mins, epi_len, flare):
    """resample's ``fill`` closure (human.rs:1320-1361): harmonic-mean areas of n uniform sections."""
    n = out.shape[0]
    d = nbm.fdiv(s1 - s0, float(n))
    sub = 8
    gi = 0
    for i in range(n):
        inv = 0.0
        closed = False
        mn = math.inf
        for jj in range(sub):
            area = area_at(artic, k, frame, n_grid, s0 + d * (i + (jj + 0.5) / sub), epi_len, flare)
            if area < mn:
                mn = area
            if area <= 1e-5:
                closed = True
                break
            inv += 1.0 / area
        # actual grid samples inside the section (a point closure must seal the section)
        a0 = s0 + d * i
        a1 = s0 + d * (i + 1.0) + (1e-6 if i == n - 1 else 0.0)
        while gi < n_grid and frame[F_GRID + gi * NG + G_S] < a0:
            gi += 1
        q = gi
        while q < n_grid and frame[F_GRID + q * NG + G_S] <= a1:
            aq = frame[F_GRID + q * NG + G_AREA]
            if aq < mn:
                mn = aq
            if aq <= 1e-5:
                closed = True
            q += 1
        out[i] = 0.0 if closed else nbm.fdiv(float(sub), inv)
        mins[i] = 0.0 if closed else nbm.js_min(mn, out[i])


@jit
def geometry_resample(artic, tract, frame, out_ph, out_or, min_ph, min_or, lens):
    """HumanVocalTract::resample (human.rs:1288-1366) of the frame last computed for ``artic``: harmonic-mean
    section areas into out_ph[n_ph] / out_or[n_or], sub-section minima into min_ph / min_or, and
    lens[0] = pharynx length, lens[1] = oral length (cm)."""
    k = tract[T_K]
    epi_len = 2.4 * k * tract[T_PHARYNX]
    flare = 0.1 * epi_len
    n_grid = int(frame[F_N_GRID])
    sp = frame[F_PHARYNX_LENGTH]
    lip_s = frame[F_LIP_S]
    resample_fill(artic, k, frame, n_grid, 0.0, sp, out_ph, min_ph, epi_len, flare)
    resample_fill(artic, k, frame, n_grid, sp, lip_s, out_or, min_or, epi_len, flare)
    lens[0] = sp
    lens[1] = lip_s - sp
