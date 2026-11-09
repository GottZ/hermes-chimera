#!/usr/bin/env python3
"""S1-V3fix evidence: compare the corrected full tracks (row b = after the complete control step b) with the
pre-fix ones (row taken after the contact re-apply, before the breath/pitch part of the step).
Expected: every field identical in the same row, except p_lung, gate, f0_glide, glot_par[f0], glot_par[effort],
which equal the pre-fix row b+1 (b < n-1). Usage: python3 -I compare_shift.py <old-dir> <new-dir>"""
import json
import os
import struct
import sys

OLD, NEW = sys.argv[1:3]
LAGGED = {"p_lung", "gate", "f0_glide"}
GLAG = (0, 7)  # glot_par f0, effort


def eq(a, b):
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return struct.pack("<d", float(a)) == struct.pack("<d", float(b))
    return a == b


bad = 0
for s in ["sustain-aa-male", "hello-male", "stop-male", "song-female"]:
    with open(os.path.join(OLD, f"track-{s}.full.json"), encoding="utf-8") as f:
        o = json.load(f)
    with open(os.path.join(NEW, f"track-{s}.full.json"), encoding="utf-8") as f:
        n = json.load(f)
    N = n["n_blocks"]
    assert o["n_blocks"] == N
    same_keys = [k for k in o["blocks"] if k not in LAGGED and k != "glot_par"]
    diff_same = {k: sum(1 for b in range(N) if not eq(o["blocks"][k][b], n["blocks"][k][b])) for k in same_keys}
    glot_other = sum(1 for b in range(N) if not eq([x for i, x in enumerate(o["blocks"]["glot_par"][b]) if i not in GLAG],
                                                 [x for i, x in enumerate(n["blocks"]["glot_par"][b]) if i not in GLAG]))
    shifted = {k: sum(1 for b in range(N - 1) if not eq(n["blocks"][k][b], o["blocks"][k][b + 1])) for k in LAGGED}
    gshift = sum(1 for b in range(N - 1) for i in GLAG if not eq(n["blocks"]["glot_par"][b][i], o["blocks"]["glot_par"][b + 1][i]))
    changed_same_row = {k: sum(1 for b in range(N) if not eq(n["blocks"][k][b], o["blocks"][k][b])) for k in LAGGED}
    gchanged = sum(1 for b in range(N) for i in GLAG if not eq(n["blocks"]["glot_par"][b][i], o["blocks"]["glot_par"][b][i]))
    ini_same = eq(o["initial"], {k: v for k, v in n["initial"].items() if k in o["initial"]})
    new_keys = sorted(set(n["blocks"]) - set(o["blocks"]))
    errs = sum(diff_same.values()) + glot_other + sum(shifted.values()) + gshift + (0 if ini_same else 1)
    bad += errs
    print(f"{s}: blocks {N}; unchanged fields differing in the same row: {sum(diff_same.values())} (+glot other {glot_other}); "
          f"new row b != old row b+1 (lagged fields): {sum(shifted.values()) + gshift}; lagged values changed in the same row: "
          f"{ {**changed_same_row, 'glot f0+effort': gchanged} }; initial unchanged: {ini_same}; new keys {new_keys}; "
          f"last block p_lung/f0/effort new: {n['blocks']['p_lung'][N - 1]!r} {n['blocks']['glot_par'][N - 1][0]!r} {n['blocks']['glot_par'][N - 1][7]!r}")
print("OK" if bad == 0 else f"FAIL {bad}")
sys.exit(1 if bad else 0)
