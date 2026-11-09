#!/usr/bin/env python3
"""Re-read the repository vectors the way a consumer would (README rules) and compare them with the
lab full versions. Usage: python3 -I verify.py <repo-vectors-dir> <lab-refs-dir>"""
import json
import os
import struct
import sys

REPO, REFS = sys.argv[1:3]
bad = []


def eq(a, b):
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return struct.pack("<d", float(a)) == struct.pack("<d", float(b))
    return a == b


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


for s in ["sustain-aa-male", "hello-male", "stop-male", "song-female"]:
    t = load(os.path.join(REPO, f"track-{s}.json"))
    full = load(os.path.join(REFS, "tracks", f"track-{s}.full.json"))
    B, N = full["blocks"], full["n_blocks"]
    assert t["n_blocks"] == N
    # decode int_runs: a run holds from its start block until the next run's start
    runs = t["int_runs"]
    idx, inh = [None] * N, [None] * N
    for r, (b0, li, lh) in enumerate(runs):
        b1 = runs[r + 1][0] if r + 1 < len(runs) else N
        for b in range(b0, b1):
            idx[b], inh[b] = li, lh
    n_ok = all(len(idx[b]) == B["noise_n"][b] for b in range(N))
    ok = n_ok and idx == B["noise_idx"] and inh == B["noise_inherit"]
    ok = ok and t["reapply_blocks"] == [b for b in range(N) if B["reapply"][b]]
    fb = t["float_blocks"]
    for k, v in t["float"].items():
        ok = ok and eq(v, [B[k][b] for b in fb])
    # the scenario must reproduce the n_blocks rule
    n_samples = t["scenario"]["n_samples"]
    ok = ok and N == -(-n_samples * t["scenario"]["oversample"] // t["ctrl_every"])
    print(f"{s}: blocks {N}, runs {len(runs)}, float rows {len(fb)}, consistent with full track: {ok}")
    if not ok:
        bad.append(s)
r = load(os.path.join(REPO, "rng.json"))
print("rng seeds", [x["seed"] for x in r["rng"]], "counts", {k: len(r["rng"][0][k]) for k in ("next", "states", "gauss", "bipolar")}, "drift", [(d["control_rate"], d["base_hz"], d["seed"], len(d["steps"])) for d in r["drift"]])
print("seed 0 first state", r["rng"][4]["states"][0], "= 1 + 0x6d2b79f5 =", (1 + 0x6D2B79F5) & 0xFFFFFFFF)
g = load(os.path.join(REPO, "glottis.json"))
print("glottis params", len(g["params"]), "bodies", sorted({p["body"] for p in g["params"]}), "fields", len(g["field_order"]), "written", g["params"][0]["written"])
print("FAIL" if bad else "OK", bad)
sys.exit(1 if bad else 0)
