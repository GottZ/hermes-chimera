"""K-2.6: ControlModel with the gestures of the V3 sources -> TrackChunk == V3 (tests/vectors/track-*.json) per block.

Float fields (tube_area, tube_min, tube_len, velum_area, noise_par, glot_par, p_lung) T2 at the exported float blocks;
the voice clock T2 there; integer fields noise_n / noise_idx / noise_inherit T2i (exact) over ALL blocks. The
P1-critical blocks (hello-male 626, stop-male 1857, song-female 17 blocks) are where `noise_inherit` differs from
"final list against the previous block" -- the red target of REVIEW-1 P1-01.

The lab variant compares every field of every block (plus articulator state, contacts, gate, glide f0) against the
full exports in $CHIMERA_LAB_REFS/tracks.
"""

import json

import numpy as np
import pytest

import support

REF, VEC = "GEOMETRY_REF_CPYTHON", "GEOMETRY_VEC"
CASES = [
    pytest.param("sustain-aa-male", VEC, id="sustain-aa-male-vec"),
    pytest.param("hello-male", VEC, id="hello-male-vec"),
    pytest.param("stop-male", VEC, id="stop-male-vec"),
    pytest.param("song-female", VEC, id="song-female-vec", marks=pytest.mark.slow),
    pytest.param("sustain-aa-male", REF, id="sustain-aa-male-ref"),
    pytest.param("hello-male", REF, id="hello-male-ref", marks=pytest.mark.slow),
    pytest.param("stop-male", REF, id="stop-male-ref", marks=pytest.mark.slow),
    pytest.param("song-female", REF, id="song-female-ref", marks=pytest.mark.slow),
]


def _geometry(name):
    from chimera_vocal.control import geometry

    return getattr(geometry, name)


def _int_mismatches(track, trk):
    bad = (track.noise_n != trk.noise_n) | (track.noise_idx != trk.noise_idx).any(axis=(1, 2)) | (
        track.noise_inherit != trk.noise_inherit).any(axis=1)
    return [int(b) for b in np.flatnonzero(bad)]


def run_scenario(c2, trk, impl):
    """initial row, then the ops, then all blocks; advance() is cut at every float block to read the clock there."""
    model = c2.new_model(trk, impl)
    ini = model.initial()
    c2.check_chunk_shapes(ini, 1, trk.n_ph + trk.n_or)
    c2.apply_ops(model, trk.scenario["ops"])
    chunks, done, clocks = [], 0, {}
    for b in trk.float_blocks.tolist():
        if b + 1 > done:
            chunks.append(model.advance(b + 1 - done))
            done = b + 1
        clocks[b] = model.clock
    if trk.n_blocks > done:
        chunks.append(model.advance(trk.n_blocks - done))
    track = c2.concat_chunks(chunks)
    c2.check_chunk_shapes(track, trk.n_blocks, trk.n_ph + trk.n_or)
    return ini, track, clocks


@pytest.mark.parametrize("scenario,impl_name", CASES)
def test_track_vectors(c2, scenario, impl_name):
    trk = c2.decode_track(scenario)
    ini, track, clocks = run_scenario(c2, trk, _geometry(impl_name))
    failures = c2.row_failures(ini, 0, trk.initial, "initial")
    bad = _int_mismatches(track, trk)
    for b in bad[:8]:
        failures.append(f"block {b}: noise_n {track.noise_n[b]} vs {trk.noise_n[b]}, idx {track.noise_idx[b].tolist()} "
                        f"vs {trk.noise_idx[b].tolist()}, inherit {track.noise_inherit[b].tolist()} vs "
                        f"{trk.noise_inherit[b].tolist()}")
    crit = [b for b in trk.p1_critical if b in bad]
    if crit:
        failures.append(f"P1-critical blocks wrong: {crit}")
    for i, b in enumerate(trk.float_blocks.tolist()):
        exp = {f: trk.floats[f][i] for f in c2.FLOAT_FIELDS}
        failures += c2.row_failures(track, b, exp, f"block {b}", ints=False)
        w = c2.worst_rel([clocks[b]], [trk.floats["clock"][i]], c2.T2_FLOOR)
        if w >= c2.T2_TOL:
            failures.append(f"block {b}: clock {clocks[b]!r} vs {trk.floats['clock'][i]!r}")
    assert not failures, f"{scenario} ({impl_name}): {len(bad)} blocks with integer differences; " + "\n".join(
        failures[:25])


def test_p1_targets_are_in_the_vectors(c2):
    """The red target exists: the P1 rule really differs from noise_inherit at the critical blocks of the files."""
    from_files = {}
    for s in c2.TRACK_SCENARIOS:
        trk = c2.decode_track(s)
        for b, wrong in trk.p1_wrong.items():
            assert list(trk.noise_inherit[b][: trk.noise_n[b]]) != wrong, (s, b)
        from_files[s] = trk.p1_critical
    assert from_files["hello-male"] == [626] and from_files["stop-male"] == [1857]
    assert len(from_files["song-female"]) == 17 and from_files["sustain-aa-male"] == []


LAB_CASES = [pytest.param("sustain-aa-male"), pytest.param("hello-male"),
             pytest.param("stop-male", marks=pytest.mark.slow), pytest.param("song-female", marks=pytest.mark.slow)]


@pytest.mark.lab
@pytest.mark.parametrize("scenario", LAB_CASES)
def test_track_full_lab(c2, lab_refs, scenario):
    """Every field of every block against $CHIMERA_LAB_REFS/tracks/track-<scenario>.full.json, block by block
    (advance(1)), including articulator x/v, contact depths, gate and glide f0 after each step."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC

    full = json.loads((lab_refs / "tracks" / f"track-{scenario}.full.json").read_text())
    trk = c2.decode_track(scenario)
    blk = full["blocks"]
    model = c2.new_model(trk, GEOMETRY_VEC)
    model.initial()
    c2.apply_ops(model, trk.scenario["ops"])
    failures = []
    for b in range(full["n_blocks"]):
        ch = model.advance(1)
        exp = {f: blk[f][b] for f in ("tube_area", "tube_min", "tube_len", "velum_area", "glot_par", "p_lung")}
        exp["noise_par"] = [list(e) for e in blk["noise_par"][b]] + [[0.0] * 3] * (3 - blk["noise_n"][b])
        exp["noise_n"] = blk["noise_n"][b]
        exp["noise_idx"] = [list(e) for e in blk["noise_idx"][b]] + [[-1] * 3] * (3 - blk["noise_n"][b])
        exp["noise_inherit"] = list(blk["noise_inherit"][b]) + [-1] * (3 - blk["noise_n"][b])
        f = c2.row_failures(ch, 0, exp, f"block {b}")
        state = {"clock": model.clock, "gate": model.gate, "f0_glide": model.f0}
        ct, _, cb, _ = model.airway.contacts()
        state.update(contact_tip=ct, contact_body=cb)
        for k, v in state.items():
            if c2.worst_rel([v], [blk[k][b]], c2.T2_FLOOR) >= c2.T2_TOL:
                f.append(f"block {b}: {k} {v!r} vs {blk[k][b]!r}")
        for k, arr in (("artic_x", model.artic.x), ("artic_v", model.artic.v)):
            if c2.worst_rel(arr, blk[k][b], c2.T2_FLOOR) >= c2.T2_TOL:
                f.append(f"block {b}: {k}")
        failures += f
        if len(failures) > 20:
            break
    assert not failures, "\n".join(failures[:25])
