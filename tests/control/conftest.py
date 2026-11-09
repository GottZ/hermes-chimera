"""Shared helpers of the control tests (``chimera_vocal.control``), exposed as the session fixture ``c2``.

Test modules cannot import each other under ``--import-mode=importlib``; everything shared lives here.
No dsp code (``chimera_vocal.dsp``) is used: ``TractIndexV2`` is built from V2 ``tests/vectors/topology.json``.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Mapping

import numpy as np
import pytest

import support
from chimera_vocal.core.body import BODIES, HumanBody
from chimera_vocal.core.keys import ARTIC_KEYS
from chimera_vocal.core.types import Gesture

# articulators.rs:24-46 (min, max, hz), ARTIC_KEYS order -- the test's own copy of the template table
TEMPLATE_ARTIC_SPECS = (
    (-0.1, 2.6, 10.0), (1.0, 4.8, 12.0), (4.8, 9.2, 12.0), (3.5, 8.0, 13.0), (5.0, 10.8, 13.0), (0.0, 1.0, 9.0),
    (-1.0, 1.0, 9.0), (-0.4, 2.4, 11.0), (-0.2, 1.5, 11.0), (0.0, 1.0, 11.0), (0.0, 1.0, 10.0), (-0.25, 1.4, 12.0),
    (-1.6, 1.6, 8.0), (0.0, 1.0, 6.0), (0.0, 1.0, 6.0), (0.0, 1.0, 10.0), (0.0, 1.0, 9.0),
)
# human.rs:421-439
TEMPLATE_NEUTRAL = (0.6, 2.9, 6.85, 6.45, 7.3, 0.0, 0.0, 0.7, 0.2, 0.5, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0)

T2_TOL, T2_FLOOR = 1e-9, 1e-6

DT_48K = 128.0 / 96000.0      # ctrl_every / fs at 48 kHz, m = 2 (human_voice.rs:427)

# golden JSON outline key -> frame_layout outline name
GOLDEN_OUTLINES = (("outer", "outer"), ("inner", "inner"), ("tongue", "tongue"), ("velum", "velum"),
                   ("upperLip", "upper_lip"), ("lowerLip", "lower_lip"))


# ---- topology (V2) -> TractIndex stand-in ---------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TractIndexV2:
    """Duck type of ``dsp.network.TractIndex`` built from V2 (human_airway.rs:69-159)."""
    fs: float
    n_ph: int
    n_or: int
    n_nasal: int
    n_trachea: int
    n_nodes: int
    n_branches: int
    ph_nodes: np.ndarray
    or_nodes: np.ndarray
    glottis_port: int
    velum_port: int
    lungs: int
    sub_node: int
    supra_node: int
    tables: Any = dataclasses.field(default=None, compare=False, repr=False)

    def branch_of(self, tube: str, i: int) -> int:
        n = {"trachea": self.n_trachea, "pharynx": self.n_ph, "oral": self.n_or}[tube]
        return int(self.tables[tube][min(max(int(i), 0), n - 2)])


def v2_entry(label: str) -> dict:
    for b in support.vector("topology")["bodies"]:
        if b["label"] == label:
            return b
    raise KeyError(label)


def tract_index(label: str) -> TractIndexV2:
    e = v2_entry(label)
    return TractIndexV2(
        fs=float(e["fs_internal"]), n_ph=e["n_ph"], n_or=e["n_or"], n_nasal=e["n_nasal"], n_trachea=e["n_trachea"],
        n_nodes=e["n_nodes"], n_branches=e["n_branches"], ph_nodes=np.asarray(e["ph_nodes"], dtype=np.int64),
        or_nodes=np.asarray(e["or_nodes"], dtype=np.int64), glottis_port=e["glottis_port"],
        velum_port=e["velum_port"], lungs=e["lungs"], sub_node=e["sub_node"], supra_node=e["supra_node"],
        tables={k: list(v) for k, v in e["branch_of"].items()})


def body_of(label: str) -> HumanBody:
    """HumanBody of a V2/V3 body label (presets by name, mixes from the exported body numbers)."""
    if label in BODIES:
        return BODIES[label]
    p = v2_entry(label)["body_params"]
    return HumanBody(p["scale"], p["pharynx"], p["tongue_radius"], p["lip_width"], p["name"])


# ---- articulation ---------------------------------------------------------------------------------------------

def artic_of(d: Mapping[str, float]) -> np.ndarray:
    """A complete HumanArtic in serde form -> float64[17] (ARTIC_KEYS order)."""
    missing = [k for k in ARTIC_KEYS if k not in d]
    assert not missing, f"articulation without {missing}"
    return np.array([float(d[k]) for k in ARTIC_KEYS], dtype=np.float64)


def random_artic(rng: np.random.Generator) -> np.ndarray:
    """A posture uniformly inside the template ranges (articulators.rs:24-46)."""
    return np.array([rng.uniform(lo, hi) for lo, hi, _ in TEMPLATE_ARTIC_SPECS], dtype=np.float64)


# ---- geometry -------------------------------------------------------------------------------------------------

_TRACTS: dict = {}


def tract_for(body: HumanBody) -> np.ndarray:
    from chimera_vocal.control.frame_layout import tract_constants

    key = (body.scale, body.pharynx, body.tongue_radius, body.lip_width)
    if key not in _TRACTS:
        _TRACTS[key] = tract_constants(body)
    return _TRACTS[key]


def run_geometry(impl, body: HumanBody, artic: np.ndarray, n_ph: int = 18, n_or: int = 18) -> dict:
    """compute + resample with ``impl`` (a GeometryImpl); everything the frame readers expose, as a dict."""
    from chimera_vocal.control import frame_layout as fl

    tract = tract_for(body)
    frame = fl.alloc_frame()
    impl.compute(artic, tract, frame)
    out = {"scalars": fl.frame_scalars(frame), "grid": fl.frame_grid(frame),
           "outlines": {name: fl.frame_outline(frame, name) for name in fl.OUTLINES}}
    ph, or_, mph, mor, lens = (np.zeros(n_ph), np.zeros(n_or), np.zeros(n_ph), np.zeros(n_or), np.zeros(2))
    impl.resample(artic, tract, frame, ph, or_, mph, mor, lens)
    out["resample"] = {"ph": ph, "or": or_, "phMin": mph, "orMin": mor, "phLen": float(lens[0]),
                       "orLen": float(lens[1])}
    return out


def worst_rel(a, b, floor: float) -> float:
    return support.worst([float(x) for x in np.ravel(a)], [float(x) for x in np.ravel(b)], floor)


def check_golden_case(impl, case: dict) -> tuple[float, list[str]]:
    """One case of tests/golden/geometry.json the way golden_geometry.rs:20-64 checks it.

    Returns (worst relative difference with floor 1e-6, structural failures). Structural = grid size, region,
    outline lengths (exact in the template test)."""
    body = BODIES[case["body"]]
    got = run_geometry(impl, body, artic_of(case["artic"]))
    problems: list[str] = []
    w = 0.0
    grid = case["grid"]
    g = got["grid"]
    if len(grid) != len(g["d"]):
        return float("inf"), [f"grid size {len(g['d'])} != {len(grid)}"]
    regions = [int(r[0]) for r in grid]
    if list(g["region"]) != regions:
        problems.append(f"regions {list(g['region'])} != {regions}")
    for col, i in (("d", 1), ("area", 2), ("s", 3)):
        w = max(w, worst_rel(g[col], [r[i] for r in grid], T2_FLOOR))
    from chimera_vocal.control.frame_layout import GOLDEN_SCALARS

    w = max(w, worst_rel([got["scalars"][k] for k in GOLDEN_SCALARS], case["scalars"], T2_FLOOR))
    for key, name in GOLDEN_OUTLINES:
        if key in case:
            mine = got["outlines"][name]
            if len(mine) != len(case[key]):
                problems.append(f"{key}: {len(mine)} != {len(case[key])} coordinates")
                continue
            w = max(w, worst_rel(mine, case[key], T2_FLOOR))
    rs, mine = case["resample"], got["resample"]
    w = max(w, worst_rel([mine["phLen"], mine["orLen"]], [rs["phLen"], rs["orLen"]], T2_FLOOR))
    for key in ("ph", "or", "phMin", "orMin"):
        w = max(w, worst_rel(mine[key], rs[key], T2_FLOOR))
    return w, problems


def compare_geometry(a: dict, b: dict, tol: float, floor: float = T2_FLOOR) -> list[str]:
    """Field-by-field comparison of two run_geometry results; integers (region, sizes, split index) exact."""
    out: list[str] = []
    ga, gb = a["grid"], b["grid"]
    if len(ga["d"]) != len(gb["d"]):
        return [f"grid size {len(ga['d'])} != {len(gb['d'])}"]
    if not np.array_equal(ga["region"], gb["region"]):
        out.append("grid regions differ")
    for col in ga:
        if col != "region":
            w = worst_rel(ga[col], gb[col], floor)
            if w > tol:
                out.append(f"grid {col}: {w:.3e}")
    for k in a["scalars"]:
        if k in ("split_index", "n_grid"):
            if a["scalars"][k] != b["scalars"][k]:
                out.append(f"{k}: {a['scalars'][k]} != {b['scalars'][k]}")
        else:
            w = worst_rel([a["scalars"][k]], [b["scalars"][k]], floor)
            if w > tol:
                out.append(f"{k}: {a['scalars'][k]!r} vs {b['scalars'][k]!r} ({w:.3e})")
    for name in a["outlines"]:
        if len(a["outlines"][name]) != len(b["outlines"][name]):
            out.append(f"outline {name}: length {len(a['outlines'][name])} != {len(b['outlines'][name])}")
        else:
            w = worst_rel(a["outlines"][name], b["outlines"][name], floor)
            if w > tol:
                out.append(f"outline {name}: {w:.3e}")
    for k in a["resample"]:
        w = worst_rel(np.atleast_1d(a["resample"][k]), np.atleast_1d(b["resample"][k]), floor)
        if w > tol:
            out.append(f"resample {k}: {w:.3e}")
    return out


# ---- V3 control tracks ----------------------------------------------------------------------------------------

TRACK_SCENARIOS = ("sustain-aa-male", "hello-male", "stop-male", "song-female")
FLOAT_FIELDS = ("tube_area", "tube_min", "tube_len", "velum_area", "noise_par", "glot_par", "p_lung")


def _pad_noise(lst: list, fill, width: int = 3) -> list:
    return [list(e) for e in lst] + [[fill] * width for _ in range(3 - len(lst))]


def decode_track(name: str) -> types.SimpleNamespace:
    """tests/vectors/track-<name>.json (V3) as arrays in TrackChunk form (README: pad noise_idx with -1, noise_par
    with 0, noise_inherit with -1). ``floats[field]`` rows belong to ``float_blocks``."""
    t = support.vector(f"track-{name}")
    nb = t["n_blocks"]
    noise_n = np.zeros(nb, dtype=np.int64)
    noise_idx = np.full((nb, 3, 3), -1, dtype=np.int64)
    noise_inherit = np.full((nb, 3), -1, dtype=np.int64)
    runs = t["int_runs"]
    for r, (b0, lst, inh) in enumerate(runs):
        b1 = runs[r + 1][0] if r + 1 < len(runs) else nb
        noise_n[b0:b1] = len(lst)
        for j, e in enumerate(lst):
            noise_idx[b0:b1, j] = e
            noise_inherit[b0:b1, j] = inh[j]
    fl = t["float"]
    floats = {k: np.asarray(fl[k], dtype=np.float64) for k in
              ("tube_area", "tube_min", "tube_len", "velum_area", "glot_par", "p_lung", "clock")}
    floats["noise_par"] = np.asarray([_pad_noise(r, 0.0) for r in fl["noise_par"]], dtype=np.float64)
    ini = dict(t["initial"])
    initial = {k: np.asarray(ini[k], dtype=np.float64) for k in
               ("tube_area", "tube_min", "tube_len", "velum_area", "glot_par", "p_lung")}
    initial["noise_par"] = np.asarray(_pad_noise(ini["noise_par"], 0.0), dtype=np.float64)
    initial["noise_n"] = int(ini["noise_n"])
    initial["noise_idx"] = np.asarray(_pad_noise(ini["noise_idx"], -1), dtype=np.int64)
    initial["noise_inherit"] = np.asarray(list(ini["noise_inherit"]) + [-1] * (3 - len(ini["noise_inherit"])),
                                          dtype=np.int64)
    sc = t["scenario"]
    return types.SimpleNamespace(
        name=name, raw=t, scenario=sc, body_label=sc["body"], n_ph=t["n_ph"], n_or=t["n_or"], n_blocks=nb,
        noise_n=noise_n, noise_idx=noise_idx, noise_inherit=noise_inherit,
        float_blocks=np.asarray(t["float_blocks"], dtype=np.int64), floats=floats, initial=initial,
        list_a={int(k): [list(e) for e in v] for k, v in t["list_a"].items()},
        reapply_blocks=set(t["reapply_blocks"]),
        p1_critical=list(t["p1_critical_blocks"]),
        p1_wrong={int(k): list(v) for k, v in t["noise_inherit_p1_at_critical"].items()})


def apply_ops(model, ops) -> None:
    """Apply the scenario ops of a V3 head (all at output sample 0: before the first control step)."""
    for o in ops:
        assert o["at"] == 0, o
        op = o["op"]
        if op == "set_source":
            model.set_source(o["patch"])
        elif op == "set_articulation":
            model.set_articulation(o["patch"], o.get("tip_free"))
        elif op == "set_gate":
            model.set_gate(o["level"])
        elif op == "schedule":
            model.schedule([Gesture.from_json(g) for g in o["gestures"]])
        else:
            raise AssertionError(f"unknown op {op}")


def new_model(trk: types.SimpleNamespace, geometry, **kw):
    """ControlModel for a V3 scenario (body + V2 TractIndex), ops not yet applied."""
    from chimera_vocal.control.track import ControlModel

    label = trk.body_label
    idx = tract_index(label)
    assert (idx.n_ph, idx.n_or) == (trk.n_ph, trk.n_or)
    return ControlModel(body_of(label), idx, geometry=geometry, **kw)


def row_failures(chunk, row: int, exp: Mapping[str, Any], label: str, *, floats: bool = True,
                 ints: bool = True) -> list[str]:
    """Differences of TrackChunk row ``row`` against the exported row of the same block (T2 floats, T2i integers).

    Exported row b and TrackChunk row b both hold the state after the complete control step b (tests/vectors
    README, V3)."""
    out: list[str] = []
    if ints:
        n = int(chunk.noise_n[row])
        if n != int(exp["noise_n"]):
            out.append(f"{label}: noise_n {n} != {exp['noise_n']}")
        if not np.array_equal(chunk.noise_idx[row], exp["noise_idx"]):
            out.append(f"{label}: noise_idx {chunk.noise_idx[row].tolist()} != {np.asarray(exp['noise_idx']).tolist()}")
        if not np.array_equal(chunk.noise_inherit[row], exp["noise_inherit"]):
            out.append(f"{label}: noise_inherit {chunk.noise_inherit[row].tolist()} != "
                       f"{np.asarray(exp['noise_inherit']).tolist()}")
    if floats:
        for f in FLOAT_FIELDS:
            mine, want = np.atleast_1d(getattr(chunk, f)[row]), np.atleast_1d(np.asarray(exp[f], dtype=np.float64))
            w = worst_rel(mine, want, T2_FLOOR)
            if w >= T2_TOL:
                out.append(f"{label}: {f} worst relative difference {w:.3e}")
    return out


def check_chunk_shapes(chunk, n_blocks: int, n_tube: int) -> None:
    assert chunk.n_blocks == n_blocks
    for f, shape, dt in (("tube_area", (n_blocks, n_tube), np.float64), ("tube_min", (n_blocks, n_tube), np.float64),
                         ("tube_len", (n_blocks, 2), np.float64), ("velum_area", (n_blocks,), np.float64),
                         ("noise_n", (n_blocks,), np.int64), ("noise_idx", (n_blocks, 3, 3), np.int64),
                         ("noise_par", (n_blocks, 3, 3), np.float64), ("noise_inherit", (n_blocks, 3), np.int64),
                         ("glot_par", (n_blocks, 17), np.float64), ("p_lung", (n_blocks,), np.float64)):
        a = getattr(chunk, f)
        assert a.shape == shape and a.dtype == dt and a.flags.c_contiguous, (f, a.shape, a.dtype)


def chunk_equal(a, b) -> list[str]:
    """Bit-equality of two TrackChunks (T6)."""
    out = []
    if a.n_blocks != b.n_blocks:
        return [f"n_blocks {a.n_blocks} != {b.n_blocks}"]
    for f in a._fields[1:]:
        x, y = getattr(a, f), getattr(b, f)
        if x.shape != y.shape or x.tobytes() != y.tobytes():
            out.append(f)
    return out


def concat_chunks(chunks):
    from chimera_vocal.control.track import TrackChunk

    return TrackChunk(sum(c.n_blocks for c in chunks),
                      *[np.ascontiguousarray(np.concatenate([getattr(c, f) for c in chunks]))
                        for f in TrackChunk._fields[1:]])


def repeated_clock(k: int, dt: float = DT_48K) -> float:
    """The template clock after k control steps: 0.0 plus dt added k times (human_voice.rs:467)."""
    c = 0.0
    for _ in range(k):
        c += dt
    return c


@pytest.fixture(scope="session")
def c2() -> types.SimpleNamespace:
    names = ("TEMPLATE_ARTIC_SPECS", "TEMPLATE_NEUTRAL", "T2_TOL", "T2_FLOOR", "DT_48K", "GOLDEN_OUTLINES",
             "TractIndexV2", "v2_entry", "tract_index", "body_of", "artic_of", "random_artic", "tract_for",
             "run_geometry", "worst_rel", "check_golden_case", "compare_geometry", "TRACK_SCENARIOS",
             "FLOAT_FIELDS", "decode_track", "apply_ops", "new_model", "row_failures", "check_chunk_shapes",
             "chunk_equal", "concat_chunks", "repeated_clock")
    g = globals()
    return types.SimpleNamespace(**{n: g[n] for n in names})
