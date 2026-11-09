#!/usr/bin/env python3
"""Assemble the repository vectors (tests/vectors/*.json) and the manifest fragment from the raw
exports of `exporter` (template crate, public API) and `track-exporter` (patched crate copy).

Usage: python3 -I assemble.py <work-dir> <lab-refs-dir> <vectors-dir> <upstream-root>
  work-dir      directory holding out/ (raw exports) and manifest-extra.json (optional); manifest-rust.json is
                written there
  lab-refs-dir  $CHIMERA_LAB_REFS (full tracks are read from tracks/, audio from audio/, references from r1/,
                ref-audio/)
  vectors-dir   tests/vectors of the plugin repository (or a copy)
  upstream-root checkout of jamesondh/chimera-vocal-synth at 092b16ce12b9
Only stdlib. Writes compact JSON (separators (',', ':')), floats as Python repr (shortest
round-trip decimal), integers as JSON integers. Fails (non-zero exit) on any cross-check failure.
"""
import hashlib
import json
import math
import os
import struct
import subprocess
import sys

LAB, REFS, REPO, UPSTREAM = sys.argv[1:5]
OUT = os.path.join(LAB, "out")
UPSTREAM_HEAD = "092b16ce12b927d473c884ca2dfae53ce1b995df"
KB = 1000
BUDGET = {"rng.json": 70 * KB, "glottis.json": 70 * KB, "topology.json": 80 * KB, "decimator.json": 5 * KB, "track": 520 * KB}
SCENARIOS = ["sustain-aa-male", "hello-male", "stop-male", "song-female"]
FAIL = []
CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "ok": bool(ok), **({"detail": detail} if detail is not None else {})})
    if not ok:
        FAIL.append(name)


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


STATUS = {}


def numbers(obj):
    """The JSON structure with every string blanked and floats as bit patterns: equal = same numbers."""
    if isinstance(obj, dict):
        return {k: numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numbers(v) for v in obj]
    if isinstance(obj, str):
        return None
    if isinstance(obj, float):
        return ("f64", struct.pack("<d", obj))
    return obj


def is_rng_trim(cur, exp):
    """cur keeps a prefix of every array of exp (the repository trim of rng.json)."""
    try:
        for a, b in zip(cur["rng"], exp["rng"], strict=True):
            if a["seed"] != b["seed"] or any(numbers(a[k]) != numbers(b[k][:len(a[k])]) for k in a if k != "seed"):
                return False
        for a, b in zip(cur["drift"], exp["drift"], strict=True):
            if any(numbers(a[k]) != numbers(b[k]) for k in ("control_rate", "base_hz", "seed")) or numbers(a["steps"]) != numbers(b["steps"][:len(a["steps"])]):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def write(name, obj):
    """Track files are always written. Any other file is written when absent or when its numbers equal the
    export (only text fields may change). rng.json, trimmed in the repository after export (README "Repository
    trims"), keeps its trim and its `note`: only its `source` text is replaced in place. The
    manifest records the repository file and the export."""
    path = os.path.join(REPO, name)
    data = dumps(obj).encode("utf-8")
    st = {"export_bytes": len(data), "export_sha256": hashlib.sha256(data).hexdigest()}
    if name.startswith("track-") or not os.path.exists(path):
        new, st["repo_state"] = data, "written"
    else:
        with open(path, "rb") as f:
            raw = f.read()
        cur = json.loads(raw)
        if raw == data:
            new, st["repo_state"] = data, "identical to export"
        elif numbers(cur) == numbers(obj):
            new, st["repo_state"] = data, "written (numbers identical to the previous file, text fields updated)"
        elif name == "rng.json" and is_rng_trim(cur, obj):
            cur["source"] = obj["source"]
            new, st["repo_state"] = dumps(cur).encode("utf-8"), "source text replaced in place (repository copy trimmed after export; trim, note and numbers kept)"
        else:
            new, st["repo_state"] = None, "kept: numbers differ from the export, not overwritten"
            check(f"{name}: repository numbers differ from the export", False)
    if new is not None:
        with open(path, "wb") as f:
            f.write(new)
    STATUS[name] = st
    return path, os.path.getsize(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bits(x):
    return struct.pack("<d", x)


def same_f64(a, b):
    """bitwise equality of nested lists of numbers"""
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_f64(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return bits(float(a)) == bits(float(b))


SOURCE = ("exported from jamesondh/chimera-vocal-synth@092b16ce12b9 (MIT, Copyright (c) 2026 Jameson Hodge), crates/chimera, "
          "by tools/vector_export; see tests/vectors/README.md")

# ---------------------------------------------------------------- V1 rng.json
rp, rs = load("rng_public.json"), load("rng_states.json")
rng = []
for a, b in zip(rp["rng"], rs["rng"]):
    check(f"rng seed {a['seed']}: next public == next copy", a["seed"] == b["seed"] and same_f64(a["next"], b["next"]))
    s0 = 1 if a["seed"] == 0 else a["seed"]
    check(f"rng seed {a['seed']}: initial state", b["initial_state"] == s0, b["initial_state"])
    want = [(s0 + (i + 1) * 0x6D2B79F5) & 0xFFFFFFFF for i in range(1000)]
    check(f"rng seed {a['seed']}: states = s0 + n*0x6d2b79f5 mod 2^32", b["states"] == want)
    rng.append({"seed": a["seed"], "next": a["next"], "states": b["states"], "gauss": a["gauss"], "bipolar": a["bipolar"]})
rng_doc = {"source": SOURCE + "; math.rs:175-260 (Rng = mulberry32, Rng::new maps seed 0 to 1; Drift)", "rng": rng, "drift": rp["drift"]}
files = {}
files["rng.json"] = write("rng.json", rng_doc)
_rf = os.path.join(REFS, "tracks", "rng.full.json")
if os.path.exists(_rf):
    with open(_rf, encoding="utf-8") as _f:
        check("rng: numbers of lab tracks/rng.full.json == export", numbers(json.load(_f)) == numbers(rng_doc))

# ---------------------------------------------------------------- V2d decimator.json
dp_, dq = load("decimator_public.json"), load("decimator_private.json")
check("decimator m=2: design_lowpass(49, 0.23, 8.5) == Decimator::new(2).h", same_f64(dp_["m2"]["coeffs"], dq["m2"]["coeffs"]))
check("decimator m=2: 49 taps", len(dp_["m2"]["coeffs"]) == 49)
check("decimator m=1: h == [1.0]", same_f64(dq["m1"]["coeffs"], [1.0]))
h = dp_["m2"]["coeffs"]
check("decimator m=2: impulse output k == h[2k+1]", same_f64(dp_["m2"]["impulse_out"], [h[2 * k + 1] for k in range(8)]))
dec_doc = {
    "source": SOURCE + "; airway/decimator.rs:50-81",
    "m2": {"factor": 2, "taps": 49, "design": "design_lowpass(taps=2*24+1=49, cutoff=0.46/2, beta=8.5), normalised to sum 1",
           "coeffs": dp_["m2"]["coeffs"], "initial_pos": dq["m2"]["pos"][1], "initial_count": dq["m2"]["pos"][2], "buf_len": dq["m2"]["pos"][0],
           "impulse_in": "push(1.0) at push index 0, then push(0.0)",
           "impulse_push_index": dp_["m2"]["impulse_push_index"], "impulse_out": dp_["m2"]["impulse_out"]},
    "m1": {"factor": 1, "taps": 1, "coeffs": dq["m1"]["coeffs"], "impulse_push_index": dp_["m1"]["impulse_push_index"], "impulse_out": dp_["m1"]["impulse_out"]},
}
files["decimator.json"] = write("decimator.json", dec_doc)

# ---------------------------------------------------------------- V5 glottis.json
gp, gq = load("glottis_public.json"), load("glottis_private.json")
field_order = gp["field_order"]
params = []
written0 = None
for a, b in zip(gp["params"], gq["params"]):
    check(f"glottis params {a['name']}: public == copy", a["name"] == b["name"] and same_f64(a["glottis"], b["glottis"]))
    params.append({"name": a["name"], "body": a["body"], "body_mix": a["body_mix"], "body_scale": a["body_scale"], "source": a["source"], "glottis": a["glottis"], "written": b["written"]})
    written0 = written0 or b["written"]
    check(f"glottis params {a['name']}: written mask constant", b["written"] == written0)
expect_written = [f not in ("f0", "closure", "chaos") for f in field_order]
check("glottis written mask = all but f0/closure/chaos (human_voice.rs:308-341)", written0 == expect_written, written0)
check("glottis default == ParametricGlottis::new().params", same_f64(gp["default_glottis"], gp["new_glottis_params"]))
nexts_full = []
for a, b in zip(gp["next"], gq["next"]):
    check(f"glottis next {a['name']}: out public == out copy", same_f64(a["out"], b["out"]))
    check(f"glottis next {a['name']}: area telemetry == return value", same_f64(a["out"], a["area"]))
    nexts_full.append((a, b))


def glottis_doc(head, stride):
    idx = sorted(set(list(range(min(head, 2000))) + list(range(head, 2000, stride)) + [1999]))
    nx = []
    for a, b in nexts_full:
        nx.append({"name": a["name"], "from": a["from"], "overrides": a["overrides"], "glottis": a["glottis"], "dp_kind": a["dp_kind"],
                   "threshold": a["threshold"], "dp": [a["dp"][i] for i in idx], "area": [a["out"][i] for i in idx],
                   "f0_now": [a["f0_now"][i] for i in idx], "voicing": [a["voicing"][i] for i in idx]})
    return {"source": SOURCE + "; sources/glottis.rs, voice/human_voice.rs:203-341", "field_order": field_order,
            "default_glottis": gp["default_glottis"], "params": params,
            "next_fs": gp["next_fs"], "next_n": gp["next_n"], "next_dp_kinds": gp["dp_kinds"],
            "next_rule": "g = ParametricGlottis::new(next_fs); g.params = glottis (17, field_order); for i in 0..next_n: area_i = g.next(dp_i); record (g.area == area_i, g.f0_now, g.voicing) after the call",
            "next_index_rule": f"recorded sample indices: 0..{head - 1}, then every {stride}th from {head}, plus 1999 (listed in next_index)",
            "next_index": idx, "next": nx}


gdoc = None
for stride in range(2, 400):
    for head in (256, 128, 64):
        cand = glottis_doc(head, stride)
        if len(dumps(cand).encode()) <= BUDGET["glottis.json"]:
            gdoc = cand
            break
    if gdoc:
        break
files["glottis.json"] = write("glottis.json", gdoc)
with open(os.path.join(REFS, "tracks", "glottis-next.full.json"), "w", encoding="utf-8") as f:
    f.write(dumps({"source": SOURCE, "next_fs": gp["next_fs"], "next_n": gp["next_n"], "next_dp_kinds": gp["dp_kinds"],
                   "next": [{**{k: a[k] for k in ("name", "from", "overrides", "glottis", "dp_kind", "threshold", "dp", "f0_now", "voicing")},
                             "area": a["out"], "cycle_phase_end": a["cycle_phase_end"], "state_initial": b["state_initial"],
                             "state_after_1000": b["state_after_1000"], "state_after_2000": b["state_after_2000"]} for a, b in nexts_full]}))

# ---------------------------------------------------------------- V2 topology.json
tp, tq = load("topology_public.json"), load("topology_private.json")
bodies = []
ZERO_NET = ["p", "u", "psrc", "wall_y", "wall_u", "rad_u", "ext_flow"]
ZERO_PRIV = ["area_step", "amin_step", "dx_step", "half_l", "half_r", "sqrt_a", "inv_amin", "inv_lsum", "spill", "vol_src1",
             "vol_src2", "wall_v", "mom", "l_prev", "rad_ul", "rad_l_prev"]
for a, q in zip(tp["bodies"], tq["bodies"]):
    lab = a["label"]
    n, v = q["network"], q["voice"]
    st = a["state"]
    le = a["linear_elements"]
    check(f"topology {lab}: label match", lab == q["label"])
    check(f"topology {lab}: linear_elements a/b == b_a/b_b", le["a"] == n["b_a"] and le["b"] == n["b_b"])
    check(f"topology {lab}: fixed/rad_nodes", le["fixed"] == n["fixed"] and le["rad_nodes"] == n["rad_nodes"])
    check(f"topology {lab}: wall params public == private", all(same_f64(le[k], n[k]) for k in ("wall_m", "wall_r", "wall_k", "wall_rad")))
    check(f"topology {lab}: targets == current after snap", same_f64(n["area_target"], st["area"]) and same_f64(n["amin_target"], st["amin"]) and same_f64(n["dx_target"], st["dx"]))
    zero_ok = all(all(x == 0 for x in st[k]) if isinstance(st[k], list) else st[k] == 0 for k in ZERO_NET) and all(all(x == 0 for x in n[k]) for k in ZERO_PRIV) and n["rad_prev_total"] == 0
    check(f"topology {lab}: zero-initialised state", zero_ok)
    check(f"topology {lab}: glottis params public == private", same_f64(st["glottis_params"], v["glottis_params"]))
    check(f"topology {lab}: noise list public == private", [x[:4] for x in st["noise"]] == [[s["src"], s["fwd"], s["bwd"], s["gain"]] for s in n["noise"]])
    tubes = []
    for t, s in zip(a["tubes"], n["tube_specs"]):
        check(f"topology {lab}: tube {t['name']} layout", t["name"] == s["name"] and t["n"] == s["n"])
        tubes.append({"name": t["name"], "index": t["index"], "first": t["first"], "n": t["n"], "branch_base": t["branch_base"],
                      "length": s["length"], "area": s["area"], "wall": s["wall"], "perimeter_factor": s["perimeter_factor"],
                      "viscous_factor": s["viscous_factor"], "wall_radiation": s["wall_radiation"]})
    ph, orl = a["pharynx"], a["oral"]
    ports = [{k: p[k] for k in ("branch", "spec_area", "length", "k_entrance", "k_exit", "slit_width", "half_sections", "extra_resistance", "port_area", "port_area_target")} for p in n["ports"]]
    port_branches = [p["branch"] for p in n["ports"]]
    check(f"topology {lab}: b_kind 1 exactly at ports", [j for j, k in enumerate(n["b_kind"]) if k == 1] == port_branches)
    check(f"topology {lab}: port_area public == private", all(same_f64(st["port_area"][p["branch"]], p["port_area"]) for p in n["ports"]))
    gs = dict(v["glottis_state"])
    mna = a["constants"]["min_node_area"]
    check(f"topology {lab}: vol == max(area, min_node_area) * dx", same_f64(n["vol"], [max(x, mna) * d for x, d in zip(st["area"], st["dx"])]))
    check(f"topology {lab}: inv_a == 1 / max(area, 1e-6)", same_f64(n["inv_a"], [1.0 / max(x, 1e-6) for x in st["area"]]))
    bodies.append({
        "label": lab, "body": a["body"], "body_mix": a["body_mix"], "body_params": a["body_params"], "fs_internal": a["fs_internal"],
        "n_ph": a["n_ph"], "n_or": a["n_or"], "n_trachea": a["n_trachea"], "n_nasal": a["n_nasal"], "n_fossa": a["n_fossa"],
        "n_nodes": a["n_nodes"], "n_branches": a["n_branches"], "n_rad": len(n["rad_nodes"]),
        "ph_nodes": list(range(ph[0], ph[0] + ph[1])), "or_nodes": list(range(orl[0], orl[0] + orl[1])),
        "glottis_port": a["glottis_port"], "velum_port": a["velum_port"], "lungs": a["lungs"],
        "sub_node": a["sub_node"], "supra_node": a["supra_node"], "rad_lips": a["rad_lips"], "rad_nose": a["rad_nose"],
        "tubes": tubes, "branch_of": a["branch_of"],
        "branches": {"a": n["b_a"], "b": n["b_b"], "kind": n["b_kind"]}, "ports": ports, "radiators": n["rad_nodes"],
        "nodes": {"fixed": [k for k, x in enumerate(n["fixed"]) if x], "junction": [k for k, x in enumerate(n["is_junction"]) if x],
                  "has_wall": [k for k, x in enumerate(n["has_wall"]) if x], "node_tube": n["node_tube"], "rad_of": n["rad_of"],
                  "nbr_start": n["nbr_start"], "nbr": n["nbr_list"], "wall_m": n["wall_m"], "wall_r": n["wall_r"], "wall_k": n["wall_k"],
                  "wall_perim": n["wall_perim"], "wall_rad": n["wall_rad"], "visc_f": n["visc_f"]},
        "constants": {**a["constants"], "inv_t": n["inv_t"], "vol_lp_coef": n["vol_lp_coef"]},
        "initial": {
            "area": st["area"], "amin": st["amin"], "dx": st["dx"], "tube_length": a["tube_length"],
            "ph_areas": st["ph_areas"], "or_areas": st["or_areas"], "ph_min": st["ph_min"], "or_min": st["or_min"],
            "ph_len": st["ph_len"], "or_len": st["or_len"], "frame_velum_port": st["frame_velum_port"], "frame_teeth_s": st["frame_teeth_s"],
            "zero_fields": ZERO_NET + ZERO_PRIV + ["rad_prev_total"],
            "noise": [[s["src"], s["fwd"], s["bwd"], s["gain"], s["lp_coef"], s["hp_coef"]] for s in n["noise"]],
            "net_rng_state": n["rng_state"],
            "glottis_params": v["glottis_params"], "glottis_state": gs,
            "voice": {k: v[k] for k in ("ctrl_every", "ctrl_counter", "output_gain", "dc_coef", "lim_hold_n", "lim_release", "max_penetration",
                                       "p_lung", "p_lung_now", "p_lung_step", "f0_glide", "base_abduction", "base_closure", "gate", "gate_target", "clock")},
            "decimator": {"factor": v["decimator"]["factor"], "pos": v["decimator"]["pos"], "count": v["decimator"]["count"], "buf_len": v["decimator"]["buf_len"]},
            "artic_x": v["artic_x"], "artic_target": v["artic_target"], "source": st["source"],
        },
    })
topo = {"source": SOURCE + "; voice/human_airway.rs:69-154, airway/network.rs:426-617, voice/human_voice.rs:203-269",
        "initial_rule": "state right after HumanVoice::new(48000, body, 2): airway.apply(artic.x, 0) + net.snap_to_targets() + apply_source(); targets (area_target, amin_target, dx_target, port_area_target) equal the current values; every field listed in zero_fields is 0; derived and not stored: vol[k] = js_max(area[k], min_node_area) * dx[k], inv_a[k] = 1 / js_max(area[k], 1e-6) (network.rs:679-680, checked bitwise at export)",
        "node_order": "tubes in creation order (trachea, pharynx, oral, nasal, piriform, lungs, sinus); branch order = creation order (tube-internal series branches, then connect/add_port calls)",
        "bodies": bodies}
files["topology.json"] = write("topology.json", topo)
with open(os.path.join(REFS, "tracks", "topology.full.json"), "w", encoding="utf-8") as f:
    f.write(dumps({"source": SOURCE, "public": tp, "private": tq}))

# ---------------------------------------------------------------- V3 track-*.json
sc_check = load("selfcheck.json")["checks"]
FLOAT_KEYS = ["tube_area", "tube_min", "tube_len", "velum_area", "noise_par", "glot_par", "p_lung", "clock"]
tracks_full = {}
for s in SCENARIOS:
    with open(os.path.join(REFS, "tracks", f"track-{s}.full.json"), encoding="utf-8") as f:
        tracks_full[s] = json.load(f)
    pub = load(f"scenario_{s}.json")
    full_sc = tracks_full[s]["scenario"]
    Bf = tracks_full[s]["blocks"]
    check(f"track {s}: velum_target == max(0, velum_area) in every block", all(same_f64(max(0.0, a), t) for a, t in zip(Bf["velum_area"], Bf["velum_target"])))
    check(f"track {s}: n_blocks == ceil(n_samples * oversample / ctrl_every)", tracks_full[s]["n_blocks"] == -(-pub["n_samples"] * 2 // tracks_full[s]["ctrl_every"]))
    pressure = 8.0
    for op in pub["ops"]:
        if op["op"] == "set_source" and "pressure" in op["patch"]:
            pressure = op["patch"]["pressure"]
    cm = 980.665
    gidx_eff, gidx_f0 = 7, 0
    ok_pl = all(same_f64(pl, (pressure * cm) * g) for pl, g in zip(Bf["p_lung"], Bf["gate"]))
    ok_st = all(same_f64(st_, (pl - pn) / 128.0) for st_, pl, pn in zip(Bf["p_lung_step"], Bf["p_lung"], Bf["p_lung_now"]))
    ok_ef = all(same_f64(gp_[gidx_eff], min(1.0, max(0.0, (abs(pressure * g) - 3.0) / 10.0))) for gp_, g in zip(Bf["glot_par"], Bf["gate"]))
    ok_f0 = all(same_f64(gp_[gidx_f0], f) for gp_, f in zip(Bf["glot_par"], Bf["f0_glide"]))
    check(f"track {s}: row b is after the breath/pitch part of step b (p_lung = pressure*CM_H2O*gate, p_lung_step = (p_lung-p_lung_now)/128, effort(gate), glot f0 = f0_glide)", ok_pl and ok_st and ok_ef and ok_f0, [ok_pl, ok_st, ok_ef, ok_f0])
    check(f"track {s}: scenario (ops, gestures) public == copy", dumps(pub["ops"]) == dumps(full_sc["ops"]) and pub["n_samples"] == full_sc["n_samples"])


def int_runs(B, n):
    runs, prev = [], None
    for b in range(n):
        key = (B["noise_idx"][b], B["noise_inherit"][b])
        if key != prev:
            runs.append([b, B["noise_idx"][b], B["noise_inherit"][b]])
            prev = key
    return runs


def lean_scenario(sc):
    info = dict(sc["info"])
    for k in ("phrases", "plan_segments", "syllables"):
        info.pop(k, None)
    out = {k: v for k, v in sc.items() if k != "info"}
    out["info"] = info
    return out


def track_doc(s, float_blocks, selection):
    T = tracks_full[s]
    B, N = T["blocks"], T["n_blocks"]
    ini = T["initial"]
    crit = [b for b in range(N) if B["noise_inherit"][b] != B["noise_inherit_p1"][b]]
    a_ne_b = {}
    for b, la in T["list_a"].items():
        if [x[:2] for x in la["noise_idx"]] != [x[:2] for x in B["noise_idx"][int(b)]]:
            a_ne_b[b] = la["noise_idx"]
    return {
        "source": SOURCE + "; track-exporter on the crate copy with the control-step recorder (tools/vector_export/crate-chimera.patch), voice/human_voice.rs:426-509",
        "scenario": lean_scenario(T["scenario"]),
        "n_ph": T["n_ph"], "n_or": T["n_or"], "ctrl_every": T["ctrl_every"], "block_rule": T["block_rule"], "n_blocks": N,
        "initial": {**{k: ini[k] for k in FLOAT_KEYS}, "noise_n": ini["noise_n"], "noise_idx": ini["noise_idx"], "noise_inherit": [-1] * ini["noise_n"]},
        "int_runs": int_runs(B, N),
        "reapply_blocks": [b for b in range(N) if B["reapply"][b]],
        "list_a": a_ne_b,
        "p1_critical_blocks": crit,
        "noise_inherit_p1_at_critical": {str(b): B["noise_inherit_p1"][b] for b in crit},
        "float_selection": selection,
        "float_blocks": float_blocks,
        "float": {k: [B[k][b] for b in float_blocks] for k in FLOAT_KEYS},
    }


mandatory = {}
for s in SCENARIOS:
    N = tracks_full[s]["n_blocks"]
    B = tracks_full[s]["blocks"]
    m = set()
    if s in ("sustain-aa-male", "hello-male"):
        m |= set(range(0, min(80, N)))
    if s in ("hello-male", "stop-male"):
        for b in range(N):
            if B["noise_inherit"][b] != B["noise_inherit_p1"][b]:
                m |= {x for x in (b - 1, b, b + 1) if 0 <= x < N}
    mandatory[s] = m


def size_of(s, fb, sel):
    return len(dumps(track_doc(s, fb, sel)).encode())


def selection(s, stride):
    return {"head_0_79": s in ("sustain-aa-male", "hello-male"), "p1_neighbours": s in ("hello-male", "stop-male"), "stride": stride}


base = {s: size_of(s, sorted(mandatory[s]), selection(s, None)) for s in SCENARIOS}
remaining = BUDGET["track"] - sum(base.values())
check("track budget: mandatory parts fit", remaining >= 0, {"base": base, "remaining": remaining})
chosen = {}
for s in ("sustain-aa-male", "hello-male"):
    chosen[s] = (sorted(mandatory[s]), None)
half = remaining // 2
for s in ("stop-male", "song-female"):
    N = tracks_full[s]["n_blocks"]
    best = (sorted(mandatory[s]), None)
    for stride in range(max(1, N // 400), N + 1):
        fb = sorted(mandatory[s] | set(range(0, N, stride)))
        if size_of(s, fb, selection(s, stride)) - base[s] <= half:
            best = (fb, stride)
            break
    chosen[s] = best
track_sizes = {}
for s in SCENARIOS:
    fb, stride = chosen[s]
    doc = track_doc(s, fb, selection(s, stride))
    files[f"track-{s}.json"] = write(f"track-{s}.json", doc)
    track_sizes[s] = files[f"track-{s}.json"][1]
check("track budget: all track-*.json <= 520 KB", sum(track_sizes.values()) <= BUDGET["track"], track_sizes)

# ---------------------------------------------------------------- self-check summary + envelope of the counter-check


def read_f32(path):
    with open(path, "rb") as f:
        b = f.read()
    return struct.unpack(f"<{len(b) // 4}f", b)


def env_db(x):
    out = []
    for i in range(0, len(x), 480):
        c = x[i:i + 480]
        e = sum(float(v) * float(v) for v in c) / len(c)
        out.append(10.0 * math.log10(max(e, 1e-30)))
    return out


selfcheck = []
for c in sc_check:
    s = c["scenario"]
    entry = {k: c[k] for k in ("scenario", "n_samples", "n_blocks", "reapply_blocks", "a_ne_b_blocks", "p1_critical_blocks", "inherit_check_failures", "coef_check_failures", "emulation_rows_same_inherit")}
    for k in ("copy_plain_vs_upstream", "copy_record_vs_upstream", "copy_emulate_inherit_vs_upstream", "copy_p1_rebase_vs_upstream"):
        entry[k] = c[k]
    a = read_f32(os.path.join(REFS, "audio", f"track_{s}.f32"))
    b = read_f32(os.path.join(OUT, f"p1rebase_{s}.f32"))
    ea, eb = env_db(a), env_db(b)
    top = max(ea)
    entry["p1_rebase_envelope_max_db"] = max((abs(x - y) for x, y in zip(ea, eb) if x > top - 45.0), default=0.0)
    selfcheck.append(entry)
    check(f"selfcheck {s}: emulation bit-identical to upstream", c["copy_emulate_inherit_vs_upstream"]["first_bit_difference"] is None and c["copy_emulate_inherit_vs_upstream"]["len_got"] == c["copy_emulate_inherit_vs_upstream"]["len_want"])
    check(f"selfcheck {s}: record mode bit-identical to upstream", c["copy_record_vs_upstream"]["first_bit_difference"] is None)
    check(f"selfcheck {s}: copy without recorder bit-identical to upstream", c["copy_plain_vs_upstream"]["first_bit_difference"] is None)
    check(f"selfcheck {s}: recorded inherit predicts every filter state", c["inherit_check_failures"] == 0)
    check(f"selfcheck {s}: noise coefficients from lowpass/highpass Hz", c["coef_check_failures"] == 0)
hello = [e for e in selfcheck if e["scenario"] == "hello-male"][0]["copy_p1_rebase_vs_upstream"]
check("counter-check hello-male: P1 emulation diverges at sample 40066", hello["first_over_1e-6_peak"] == 40066, hello["first_over_1e-6_peak"])
check("counter-check hello-male: max |d| ~ 1.3e-2 x peak", 1.2e-2 <= hello["max_abs_diff_over_peak"] <= 1.35e-2, hello["max_abs_diff_over_peak"])

# independent references for the upstream renders


def wav_f32(path):
    with open(path, "rb") as f:
        d = f.read()
    i = 12
    while i < len(d):
        cid, sz = d[i:i + 4], struct.unpack("<I", d[i + 4:i + 8])[0]
        if cid == b"data":
            return d[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)


def raw(path):
    with open(path, "rb") as f:
        return f.read()


AUD = os.path.join(REFS, "audio")
refs = [
    ("sustain-aa-male", os.path.join(REFS, "r1", "voice_male_110hz_aa-back_1s.f32"), raw),
    ("hello-male", os.path.join(REFS, "r1", "speech_male_hello_world.f32"), raw),
    ("song-female", os.path.join(REFS, "ref-audio", "song-female", "twinkle-twinkle-female.wav"), wav_f32),
]
indep = []
for s, p, rd in refs:
    if not os.path.exists(p):
        indep.append({"scenario": s, "reference": p, "bit_identical": None, "note": "reference missing, skipped"})
        continue
    ok = rd(p) == raw(os.path.join(AUD, f"track_{s}.f32"))
    indep.append({"scenario": s, "reference": p, "bit_identical": ok})
    check(f"upstream render {s} == independent reference {os.path.basename(p)}", ok)
check("stop_male_text.f32 == track_stop-male.f32", raw(os.path.join(AUD, "stop_male_text.f32")) == raw(os.path.join(AUD, "track_stop-male.f32")))

# ---------------------------------------------------------------- manifest fragment


def git(*a):
    r = subprocess.run(["git", "-C", LAB, *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


_top = git("rev-parse", "--show-toplevel")
# exporter commit only when the work dir itself is the top of a git repository (not a copy inside another repo)
head = git("rev-parse", "HEAD") if _top and os.path.realpath(_top) == os.path.realpath(LAB) else None
dirty = [l for l in (git("status", "--porcelain") or "").splitlines() if not l.endswith("manifest-rust.json")]
up_status = subprocess.run(["git", "-C", UPSTREAM, "status", "--short"], capture_output=True, text=True, check=True).stdout
up_head = subprocess.run(["git", "-C", UPSTREAM, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
check("upstream HEAD", up_head == UPSTREAM_HEAD, up_head)
check("upstream worktree clean", up_status == "", up_status)
readme = os.path.join(REPO, "README.md")
budget_of = lambda n: BUDGET["track"] if n.startswith("track-") else BUDGET.get(n)
lab_files = []
for d in ("tracks", "audio"):
    for name in sorted(os.listdir(os.path.join(REFS, d))):
        p = os.path.join(REFS, d, name)
        if os.path.isfile(p):
            lab_files.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)})
full_of = {
    "rng.json": [os.path.join(REFS, "tracks", "rng.full.json")], "decimator.json": [],
    "glottis.json": [os.path.join(REFS, "tracks", "glottis-next.full.json")],
    "topology.json": [os.path.join(REFS, "tracks", "topology.full.json")],
}
for s in SCENARIOS:
    full_of[f"track-{s}.json"] = [os.path.join(REFS, "tracks", f"track-{s}.full.json"), os.path.join(REFS, "audio", f"track_{s}.f32")]
producer = {
    "rng.json": "exporter (next, gauss, bipolar, drift: public API) + track-exporter (states: Rng::chimera_state accessor)",
    "decimator.json": "exporter (design_lowpass, Decimator push/value: public API) + track-exporter (Decimator.h/pos accessor, cross-check)",
    "glottis.json": "exporter (params, next: public API) + track-exporter (written mask: apply_source on sentinel params)",
    "topology.json": "exporter (public fields, linear_elements) + track-exporter (private network/voice/glottis state accessors)",
}
manifest = {
    "fragment": "rust vectors",
    "template_commit": UPSTREAM_HEAD,
    "exporter_commit": head,
    "exporter_repo": LAB,
    "exporter_worktree_clean": None if head is None else not dirty,
    "toolchain": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
    "build_profile": "release: opt-level 3, lto fat, codegen-units 1, panic abort (same as upstream Cargo.toml:9-15)",
    "files": [],
    "lab_files": lab_files,
    "selfcheck": selfcheck,
    "independent_references": indep,
    "glottis_next_selection": {"head": gdoc["next_index_rule"], "n_indices": len(gdoc["next_index"])},
    "track_float_selection": {s: {"float_blocks": len(chosen[s][0]), "stride": chosen[s][1], "n_blocks": tracks_full[s]["n_blocks"]} for s in SCENARIOS},
    "checks": CHECKS,
    "checks_failed": FAIL,
}
for name, (path, size) in files.items():
    manifest["files"].append({"path": "tests/vectors/" + name, "bytes": size, "sha256": sha256(path), "budget_bytes": budget_of(name),
                              "within_budget": size <= budget_of(name) if not name.startswith("track-") else sum(track_sizes.values()) <= BUDGET["track"],
                              "producer": producer.get(name, "track-exporter (crate copy, Mode::Record) + exporter (scenario, gestures: public API; cross-checked)"),
                              "full_versions": full_of[name], **STATUS[name]})
if os.path.exists(readme):
    manifest["files"].append({"path": "tests/vectors/README.md", "bytes": os.path.getsize(readme), "sha256": sha256(readme)})
extra = os.path.join(LAB, "manifest-extra.json")
if os.path.exists(extra):
    with open(extra, encoding="utf-8") as f:
        manifest.update(json.load(f))
with open(os.path.join(LAB, "manifest-rust.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1, ensure_ascii=False, allow_nan=False)
    f.write("\n")
for n, (p, sz) in files.items():
    print(f"{sz:>8}  {p}")
print("tracks total", sum(track_sizes.values()), "selection", {s: (len(chosen[s][0]), chosen[s][1]) for s in SCENARIOS})
print("glottis", gdoc["next_index_rule"])
print("checks", len(CHECKS), "failed", FAIL)
sys.exit(1 if FAIL else 0)
