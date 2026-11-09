"""Vocal-tract geometry, numpy form: the same function as ``geometry_ref`` (HumanVocalTract::compute / ::resample,
anatomy/human.rs:725-1370) for processes without numba.

The scalar reference spends nearly all of its CPython time in three loops: rays against wall polylines
(crossing_t), roof projections (project_on) and tongue-surface distances (signed_gap). Here those run as
rays x segments and points x segments arrays, and the resampling evaluates all sub-section positions at once. The
wall construction, the lip/area/centreline pass and the drawing outlines are shared with ``geometry_ref`` (called as
plain Python), so both forms build the same polylines from the same code.

Every elementwise expression keeps the reference's operand order, ties resolve as in the reference (first minimum,
stable sort of ray crossings) and the sums the reference folds left to right are folded the same way here. The
only difference is ``hypot``: numpy calls the C library's, CPython's ``math.hypot`` has its own algorithm (they can
differ in the last bit; numba's reference uses the C library's too).
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry_ref as g
from .geometry_ref import (
    F_GRID,
    F_LIP_S,
    F_N_GRID,
    F_OUTLINES,
    F_PHARYNX_LENGTH,
    G_AREA,
    G_D,
    G_IN_X,
    G_IN_Y,
    G_MID_X,
    G_MID_Y,
    G_OUT_X,
    G_OUT_Y,
    G_RAY,
    G_REGION,
    G_S,
    LAB_FRONT_OPENS,
    LAB_FRONT_RAMP,
    MAX_PTS,
    N_PH_RAYS,
    NG,
    O_INNER,
    O_OUTER,
    S_ARC,
    S_BLADE,
    S_FLOOR,
    S_ROOT,
    S_UNDER,
    T_K,
    T_PHARYNX,
    T_POLAR_C2_X,
    T_POLAR_C2_Y,
    T_POLAR_C_X,
    T_POLAR_C_Y,
    A_EPILARYNX,
    A_VENTRICULAR,
    DEG,
)
from ..core import nbmath as nbm

_OBASE = F_OUTLINES + O_OUTER * 2 * MAX_PTS
_IBASE = F_OUTLINES + O_INNER * 2 * MAX_PTS
_N_RAD = 30
_N_FR = 6
_SUB = 8
_ERR = dict(divide="ignore", invalid="ignore", over="ignore")


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """math.rs smoothstep, elementwise (clamp passes NaN through, like nbmath.clamp)."""
    t = (x - e0) / (e1 - e0)
    t = np.where(t < 0.0, 0.0, np.where(t > 1.0, 1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _points(frame: np.ndarray, base: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    run = frame[base:base + 2 * n]
    return run[0::2].copy(), run[1::2].copy()


def _crossings(ox, oy, dx, dy, px, py):
    """crossing_t of every ray (rows) against every segment of the polyline (columns); NaN = no crossing."""
    ax, ay, bx, by = px[:-1], py[:-1], px[1:], py[1:]
    ox, oy, dx, dy = ox[:, None], oy[:, None], dx[:, None], dy[:, None]
    sa = dx * (ay - oy) - dy * (ax - ox) >= 0.0
    sb = dx * (by - oy) - dy * (bx - ox) >= 0.0
    ex = bx - ax
    ey = by - ay
    den = dx * ey - dy * ex
    with np.errstate(**_ERR):
        t = ((ax - ox) * ey - (ay - oy) * ex) / den
    t[(sa == sb) | (np.abs(den) < 1e-15)] = np.nan
    return t, ex, ey


def _rows_min(t, valid, empty):
    return np.min(np.where(valid, t, empty), axis=1)


def _fan_t_in(ti, exits, vi, t_out, lim, tb_min, found):
    """compute's fan_inner (human.rs:943-984) for every fan ray: crossings sorted stably along the ray (as the
    template's insertion sort), entry/exit walk, fallback to the surface behind the origin."""
    order = np.argsort(np.where(vi, ti, np.inf), axis=1, kind="stable")
    ts_all = np.take_along_axis(ti, order, axis=1).tolist()
    es_all = np.take_along_axis(exits, order, axis=1).tolist()
    counts = np.count_nonzero(vi, axis=1).tolist()
    found_l, t_out_l, lim_l, tb_l = found.tolist(), t_out.tolist(), lim.tolist(), tb_min.tolist()
    t_in = [0.0] * len(counts)
    for r, n in enumerate(counts):
        if not found_l[r]:
            continue
        ts, es, t_o = ts_all[r], es_all[r], t_out_l[r]
        i = 0
        if n > 0 and es[0]:
            tin = ts[0]
            i = 1
        else:
            tin = -tb_l[r]
        tissue = 0.0
        while i < n and ts[i] < t_o:
            if es[i]:
                i += 1
                continue
            bb = ts[i + 1] if i + 1 < n else lim_l[r]
            tissue += nbm.js_min(bb, t_o) - ts[i]
            i += 2
        t_in[r] = tin + tissue
    return np.array(t_in)


def _measure_all(ox, oy, dx, dy, max_len, n_ph, n_fan, opx, opy, ipx, ipy, k):
    """compute's ``measure`` (human.rs:985-1015) for all rays of the grid at once: rows [0, n_ph) and
    [n_ph + n_fan, ...) are plain rays, rows [n_ph, n_ph + n_fan) the oral fan.
    Returns (found, inner x, inner y, outer x, outer y, d)."""
    n = ox.shape[0]
    to, _, _ = _crossings(ox, oy, dx, dy, opx, opy)
    valid = (to >= 0.0) & (to <= max_len[:, None])
    found = valid.any(axis=1)
    t_out = _rows_min(to, valid, np.inf)
    lim = t_out + 1.5 * k
    fan = slice(n_ph, n_ph + n_fan)
    # inner wall: all rays forward, plus the fan rays backwards (the surface behind a fan origin)
    bo_x = np.concatenate((ox, ox[fan]))
    bo_y = np.concatenate((oy, oy[fan]))
    bd_x = np.concatenate((dx, -dx[fan]))
    bd_y = np.concatenate((dy, -dy[fan]))
    tall, iex, iey = _crossings(bo_x, bo_y, bd_x, bd_y, ipx, ipy)
    ti = tall[:n]
    in_range = (ti >= 0.0) & (ti <= lim[:, None])
    # plain rays: last inner crossing before the wall (strictly inside the 1.5 k margin), else 3.5 k below it
    t_in = np.max(np.where(in_range & (ti < lim[:, None]), ti, -np.inf), axis=1)
    none = t_in == -np.inf
    if none.any():
        fb = t_out - 3.5 * k
        t_in = np.where(none, np.where(fb > 0.0, fb, 0.0), t_in)
    # fan rays
    back = 2.5 * k
    tb = tall[n:]
    tb_min = _rows_min(tb, (tb >= 0.0) & (tb <= back), back)
    exits = dx[fan, None] * iey - dy[fan, None] * iex < 0.0
    t_in[fan] = _fan_t_in(ti[fan], exits, in_range[fan], t_out[fan], lim[fan], tb_min, found[fan])
    with np.errstate(**_ERR):
        diff = t_out - t_in
        d = np.where(np.isnan(diff), np.nan, np.where(diff > 0.0, diff, 0.0))
        tm = np.minimum(t_in, t_out)
        return found, ox + dx * tm, oy + dy * tm, ox + dx * t_out, oy + dy * t_out, d


def _push(frame, row, region, ix, iy, ox, oy, d):
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


def _grid_rays(tract, frame, n_out, n_in, glottis_y, inc_x, ul_x, ll_x, ll_y, lab):
    """geometry_ref.grid_rays with all rays measured as one batch."""
    k = tract[T_K]
    ph = tract[T_PHARYNX]
    opx, opy = _points(frame, _OBASE, n_out)
    ipx, ipy = _points(frame, _IBASE, n_in)
    ox, oy, dx, dy, ml, angs = [], [], [], [], [], []
    # pharyngeal horizontals (origin in front, looking back)
    y_epi_top = glottis_y + 2.4 * k * ph
    y_phar_top = tract[T_POLAR_C_Y]
    for i in range(N_PH_RAYS):
        ox.append(4.6 * k)
        oy.append(g.lerp(y_epi_top + 0.12 * k, y_phar_top, i / N_PH_RAYS))
        dx.append(-1.0)
        dy.append(0.0)
        ml.append(9.0 * k)
    # oral fan: centre gliding from behind the tongue to behind the incisors
    pcx, pcy = tract[T_POLAR_C_X], tract[T_POLAR_C_Y]
    pc2x, pc2y = tract[T_POLAR_C2_X], tract[T_POLAR_C2_Y]
    for i in range(_N_RAD + 1):
        t = i / _N_RAD
        ang = g.lerp(180.0, 36.0, t) * DEG
        w = 0.0 if t < 0.45 else (t - 0.45) / 0.55
        sw = w * w * (3.0 - 2.0 * w)
        ox.append(pcx + (pc2x - pcx) * sw)
        oy.append(pcy + (pc2y - pcy) * sw)
        dx.append(math.cos(ang))
        dy.append(math.sin(ang))
        ml.append(8.0 * k)
        angs.append(ang)
    # front verticals (upper incisor edge -> lip opening), looking up
    x_start = inc_x + 0.03 * k
    x_end = nbm.js_max(x_start + 0.3 * k, nbm.js_min(ul_x, ll_x) - 0.04 * k)
    xs = [g.lerp(x_start, x_end, i / (_N_FR - 1.0)) for i in range(_N_FR)]
    for x in xs:
        ox.append(x)
        oy.append(4.0 * k)
        dx.append(0.0)
        dy.append(1.0)
        ml.append(8.0 * k)
    found, ix, iy, oxs, oys, d = (a.tolist() for a in _measure_all(
        np.array(ox), np.array(oy), np.array(dx), np.array(dy), np.array(ml), N_PH_RAYS, _N_RAD + 1, opx, opy,
        ipx, ipy, k))
    n_grid = 0
    split_index = -1
    for r in range(N_PH_RAYS):
        if found[r]:
            _push(frame, n_grid, 1.0, ix[r], iy[r], oxs[r], oys[r], d[r])
            n_grid += 1
    for i in range(_N_RAD + 1):
        r = N_PH_RAYS + i
        if not found[r]:
            continue
        if split_index < 0 and angs[i] <= 150.0 * DEG:
            split_index = n_grid
        _push(frame, n_grid, 1.0 if split_index < 0 else 2.0, ix[r], iy[r], oxs[r], oys[r], d[r])
        n_grid += 1
    teeth_idx = n_grid
    for i, x in enumerate(xs):
        r = N_PH_RAYS + _N_RAD + 1 + i
        if not found[r]:
            continue
        mix, miy, mox, moy, md = ix[r], iy[r], oxs[r], oys[r], d[r]
        if x > ll_x:
            # in front of the tucked lower lip's edge the floor starts at the lip's outer face
            face = nbm.js_min(ll_y - (x - ll_x) * 4.0 / 3.0, moy)
            opn = nbm.smoothstep(LAB_FRONT_OPENS, 1.0, lab) * nbm.smoothstep(0.0, LAB_FRONT_RAMP * k, x - ll_x)
            y_in = g.lerp(face, moy - 3.5 * k, opn)
            mix = x
            miy = y_in
            md = moy - y_in
        _push(frame, n_grid, 3.0, mix, miy, mox, moy, md)
        n_grid += 1
    return n_grid, split_index, teeth_idx


def _project(rx, ry, cum, px, py):
    """project_on for a batch of points -> (s, dist, signed, seg) arrays (first minimum, as the strict `<`)."""
    ax, ay = rx[:-1], ry[:-1]
    ex = rx[1:] - ax
    ey = ry[1:] - ay
    l2 = ex * ex + ey * ey
    px, py = px[:, None], py[:, None]
    with np.errstate(**_ERR):
        t = ((px - ax) * ex + (py - ay) * ey) / l2
    t = np.where(t < 0.0, 0.0, np.where(t > 1.0, 1.0, t))
    qx = ax + ex * t
    qy = ay + ey * t
    d = np.hypot(px - qx, py - qy)
    d = np.where((l2 < 1e-12) | np.isnan(d), np.inf, d)
    seg = np.argmin(d, axis=1)
    rows = np.arange(d.shape[0])
    best = d[rows, seg]
    tt = t[rows, seg]
    s = cum[seg] + tt * np.sqrt(l2[seg])
    cross = ex[seg] * (py[:, 0] - ay[seg]) - ey[seg] * (px[:, 0] - ax[seg])
    sign = np.where(cross < 0.0, 1.0, -1.0)
    ok = np.isfinite(best)
    s = np.where(ok, s, 0.0)
    sign = np.where(ok, sign, 1.0)
    seg = np.where(ok, seg, 0)
    return s, best, sign * best, seg


def _signed_gap(ox, oy, px, py):
    """signed_gap of a batch of points against the closed outline (ox, oy)."""
    ax, ay, bx, by = ox[:-1], oy[:-1], ox[1:], oy[1:]
    ex = bx - ax
    ey = by - ay
    l2 = ex * ex + ey * ey
    px, py = px[:, None], py[:, None]
    with np.errstate(**_ERR):
        t = np.where(l2 > 1e-12, ((px - ax) * ex + (py - ay) * ey) / l2, 0.0)
        t = np.where(t < 0.0, 0.0, np.where(t > 1.0, 1.0, t))
        ddx = px - ax - ex * t
        ddy = py - ay - ey * t
        d2 = ddx * ddx + ddy * ddy
        best = np.min(np.where(np.isnan(d2), np.inf, d2), axis=1)
        straddle = (ay > py) != (by > py)
        xcross = ax + ((py - ay) * ex) / ey
        flips = straddle & (px < xcross)
    inside = (np.count_nonzero(flips, axis=1) & 1).astype(bool)
    d = np.sqrt(best)
    return np.where(inside, -d, d)


def _point_along(rx, ry, cum, s):
    """point_along for a batch of arc lengths."""
    n = cum.shape[0]
    i = np.minimum(np.searchsorted(cum[1:], s, side="left"), n - 2)
    ln = cum[i + 1] - cum[i]
    with np.errstate(**_ERR):
        u = (s - cum[i]) / ln
    u = np.where(u < 0.0, 0.0, np.where(u > 1.0, 1.0, u))
    u = np.where(ln > 1e-9, u, 0.0)
    return rx[i] + (rx[i + 1] - rx[i]) * u, ry[i] + (ry[i + 1] - ry[i]) * u


def _closest(signed, seg, rx, ry):
    """closest_approach from the projections of a point set -> (signed gap, normal x, normal y)."""
    d = math.inf
    sg = 0
    for i, v in enumerate(signed.tolist()):
        if v < d:
            d = v
            sg = int(seg[i])
    ex = rx[sg + 1] - rx[sg]
    ey = ry[sg + 1] - ry[sg]
    ln = math.hypot(ex, ey)
    if ln == 0.0 or ln != ln:
        ln = 1.0
    return d, ey / ln, -ex / ln


def _contact_model(frame, n_grid, roof_start, roof_end, k):
    """geometry_ref.contact_model with the projections and surface distances batched."""
    rx, ry = _points(frame, _OBASE + 2 * roof_start, roof_end - roof_start + 1)
    seg_len = np.hypot(rx[1:] - rx[:-1], ry[1:] - ry[:-1])
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    bx, by = _points(frame, S_BLADE, 7)
    ux, uy = _points(frame, S_UNDER, 6)
    arx, ary = _points(frame, S_ARC, g.ARC_POINTS)
    rows = frame[F_GRID:F_GRID + n_grid * NG].reshape(n_grid, NG)
    use = np.flatnonzero(rows[:, G_REGION] != 0.0)
    # one projection batch: tip points (blade from its third point, two of the underside), body arc, roof rays
    px = np.concatenate((bx[2:], ux[1:3], arx, rows[use, G_OUT_X]))
    py = np.concatenate((by[2:], uy[1:3], ary, rows[use, G_OUT_Y]))
    pr_s, pr_d, signed, seg = _project(rx, ry, cum, px, py)
    n_tip = 7
    n_body = g.ARC_POINTS
    tip_c = _closest(signed[:n_tip], seg[:n_tip], rx, ry)
    body_c = _closest(signed[n_tip:n_tip + n_body], seg[n_tip:n_tip + n_body], rx, ry)
    pr_s = pr_s[n_tip + n_body:]
    pr_d = pr_d[n_tip + n_body:]
    gox = px[n_tip + n_body:]
    goy = py[n_tip + n_body:]
    result = (tip_c[0], tip_c[1], tip_c[2], body_c[0], body_c[1], body_c[2])
    rtx, rty = _points(frame, S_ROOT, 9)
    flx, fly = _points(frame, S_FLOOR, 3)
    olx = np.concatenate((rtx, arx[1:], bx[1:], ux[1:], flx[1:], rtx[:1]))
    oly = np.concatenate((rty, ary[1:], by[1:], uy[1:], fly[1:], rty[:1]))
    press = 1.0 - nbm.smoothstep(-0.06 * k, 0.0, nbm.js_min(tip_c[0], body_c[0]))
    half = 0.1 * k * (1.0 + press)
    roof_len = float(cum[-1])
    on_roof = 1.0 - _smoothstep(0.15 * k, 0.25 * k, pr_d)
    keep = ~(on_roof <= 0.0)
    if not keep.any():
        return result
    use, gox, goy, pr_s, pr_d, on_roof = use[keep], gox[keep], goy[keep], pr_s[keep], pr_d[keep], on_roof[keep]
    dc = _signed_gap(olx, oly, gox, goy)
    near = np.flatnonzero(dc - half - pr_d <= 0.6 * k)
    if near.size:
        # the narrowest gap along each ray's own stretch of roof (+-half), in the template's order of the four probes
        sq = pr_s[near][None, :] + np.array([-1.0, -0.5, 0.5, 1.0])[:, None] * half
        sq = np.where(sq < 0.0, 0.0, np.where(sq > roof_len, roof_len, sq)).ravel()
        qx, qy = _point_along(rx, ry, cum, sq)
        gq = _signed_gap(olx, oly, qx, qy).reshape(4, near.size)
        m = dc[near]
        for j in range(4):
            m = np.minimum(m, gq[j])
        dc[near] = m
    w = (1.0 - _smoothstep(0.15 * k, 0.6 * k, dc)) * on_roof
    cap = (1.0 - _smoothstep(0.45 * k, 0.6 * k, dc)) * on_roof
    act = ~((w <= 0.0) & (cap <= 0.0))
    dp = np.where(np.isnan(dc), np.nan, np.where(dc > 0.0, dc, 0.0))
    gd = rows[use, G_D]
    dm = gd + (np.minimum(gd, dp) - gd) * cap
    newd = dm + (dp - dm) * w
    frame[F_GRID + use[act] * NG + G_D] = newd[act]
    return result


def compute(artic: np.ndarray, tract: np.ndarray, frame: np.ndarray) -> None:
    """HumanVocalTract::compute (human.rs:725-1263) into ``frame``; signature of GeometryImpl.compute."""
    w = g.walls(artic, tract, frame)
    n_out, n_in, roof_start, roof_end, inc_x, ul_x, ul_y, ll_x, ll_y, lab, ap, spread_w = w[:12]
    tip_x, tip_y, bx, by, r, phi, glottis_x, glottis_y = w[12:]
    n_grid, split_index, teeth_idx = _grid_rays(tract, frame, n_out, n_in, glottis_y, inc_x, ul_x, ll_x, ll_y, lab)
    c = _contact_model(frame, n_grid, roof_start, roof_end, tract[T_K])
    g.grid_areas(artic, tract, frame, n_grid, split_index, teeth_idx, inc_x, ul_x, ll_x, lab, ap, spread_w, tip_x,
                 tip_y)
    g.finish(artic, tract, frame, n_out, n_in, n_grid, split_index, ul_x, ul_y, ll_x, ll_y, phi, r, tip_x, tip_y, bx,
             by, glottis_x, glottis_y, c[0], c[1], c[2], c[3], c[4], c[5])


def _epilarynx(artic, k, s, epi_len):
    """epilarynx_area for an array of positions."""
    u = s / epi_len
    vent = 0.95 * (1.0 - 0.75 * nbm.clamp(artic[A_VENTRICULAR], 0.0, 1.0))
    top = g.lerp(1.9, 0.35, nbm.clamp(artic[A_EPILARYNX], 0.0, 1.0))

    def prof(x):
        return np.where(x < 0.3, 1.3, np.where(x < 0.5, vent, 1.2 + (top - 1.2) * ((x - 0.5) / 0.5)))

    area = prof(u)
    r = 0.03
    for u0 in (0.3, 0.5):
        lo = g.epi_profile(u0 - r, vent, top)
        hi = g.epi_profile(u0 + r, vent, top)
        ramp = lo + (hi - lo) * ((u - (u0 - r)) / (2.0 * r))
        area = np.where(np.abs(u - u0) < r, ramp, area)
    return area * k * k


def _fill2(artic, k, gs, ga, tubes, epi_len, flare):
    """resample's fill (human.rs:1320-1361) for the sections of both tubes at once.

    ``tubes`` = ((s0, s1, out, mins), ...): n uniform sections between s0 and s1 each; per section the harmonic mean
    of 8 sub-samples of area_at, and the smallest sub-sample or grid area inside it (a closure seals it)."""
    pos, a0s, a1s, sizes = [], [], [], []
    offs = np.array([(jj + 0.5) / _SUB for jj in range(_SUB)])
    for s0, s1, out, _ in tubes:
        n = out.shape[0]
        d = nbm.fdiv(s1 - s0, float(n))
        idx = np.arange(n, dtype=np.float64)
        pos.append(s0 + d * (idx[:, None] + offs[None, :]))
        a0s.append(s0 + d * idx)
        a1 = s0 + d * (idx + 1.0)
        a1[n - 1] = a1[n - 1] + 1e-6
        a1s.append(a1)
        sizes.append(n)
    s = np.concatenate(pos).ravel()
    a = np.empty_like(s)
    n_grid = gs.shape[0]
    e1 = s <= epi_len - flare
    e2 = ~e1 & (s <= epi_len)
    rest = ~(e1 | e2)
    if e1.any():
        a[e1] = _epilarynx(artic, k, s[e1], epi_len)
    if e2.any():
        se = s[e2]
        ea = _epilarynx(artic, k, se, epi_len)
        a[e2] = ea + (ga[0] - ea) * ((se - (epi_len - flare)) / flare)
    if rest.any():
        sr = s[rest]
        lo = np.clip(np.searchsorted(gs, sr, side="right") - 1, 0, max(n_grid - 2, 0))
        hi = np.minimum(lo + 1, n_grid - 1)
        s_lo = gs[lo]
        den = gs[hi] - s_lo
        den = np.where(den > 1e-9, den, 1e-9)
        with np.errstate(**_ERR):
            ar = ga[lo] + (ga[hi] - ga[lo]) * ((sr - s_lo) / den)
        ar = np.where(sr >= gs[n_grid - 1], ga[n_grid - 1], ar)
        ar = np.where(sr <= gs[0], ga[0], ar)
        a[rest] = ar
    a = a.reshape(-1, _SUB)
    closed = (a <= 1e-5).any(axis=1)
    mn = a.min(axis=1)
    inv = np.zeros(a.shape[0])
    with np.errstate(**_ERR):
        for jj in range(_SUB):
            inv = inv + 1.0 / a[:, jj]
        o = _SUB / inv
    a0 = np.concatenate(a0s)
    a1 = np.concatenate(a1s)
    q0 = np.searchsorted(gs, a0, side="left").tolist()
    q1 = np.searchsorted(gs, a1, side="right").tolist()
    for i in range(a.shape[0]):
        if q1[i] > q0[i]:
            seg = ga[q0[i]:q1[i]]
            m = seg.min()
            if m < mn[i]:
                mn[i] = m
            if (seg <= 1e-5).any():
                closed[i] = True
    out_all = np.where(closed, 0.0, o)
    min_all = np.where(closed, 0.0, np.minimum(mn, out_all))
    at = 0
    for (_, _, out, mins), n in zip(tubes, sizes):
        out[:] = out_all[at:at + n]
        mins[:] = min_all[at:at + n]
        at += n


def resample(artic: np.ndarray, tract: np.ndarray, frame: np.ndarray, out_ph: np.ndarray, out_or: np.ndarray,
             min_ph: np.ndarray, min_or: np.ndarray, lens: np.ndarray) -> None:
    """HumanVocalTract::resample (human.rs:1288-1366); signature of GeometryImpl.resample."""
    k = tract[T_K]
    epi_len = 2.4 * k * tract[T_PHARYNX]
    flare = 0.1 * epi_len
    n_grid = int(frame[F_N_GRID])
    rows = frame[F_GRID:F_GRID + n_grid * NG].reshape(n_grid, NG)
    gs = rows[:, G_S].copy()
    ga = rows[:, G_AREA].copy()
    sp = frame[F_PHARYNX_LENGTH]
    lip_s = frame[F_LIP_S]
    _fill2(artic, k, gs, ga, ((0.0, sp, out_ph, min_ph), (sp, lip_s, out_or, min_or)), epi_len, flare)
    lens[0] = sp
    lens[1] = lip_s - sp
