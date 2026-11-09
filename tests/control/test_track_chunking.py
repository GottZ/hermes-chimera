"""K-2.7 (chunking, clock) and K-2.13 (settle_pitch, set_clock_block) of the ControlModel.

T6 inside one implementation: advance(1) x N == advance(N) == advance(77) + ... bit for bit; the clock grows by
repeated `+= dt` (human_voice.rs:467) -- equal to the template's exported clock and to the reference fold, never
k*dt. A voice started at block k (set_clock_block) reproduces the continuous voice's rows bit for bit when nothing
happened before block k (the rule that lets speech render each phrase window as a voice of its own).
"""

import numpy as np
import pytest

from chimera_vocal.core.glottis_params import G_F0
from chimera_vocal.core.types import Gesture

N_BLOCKS = 300   # hello-male: the first contact re-applications start at block 140


def _hello(c2):
    from chimera_vocal.control.geometry import GEOMETRY_VEC

    trk = c2.decode_track("hello-male")
    m = c2.new_model(trk, GEOMETRY_VEC)
    c2.apply_ops(m, trk.scenario["ops"])
    return trk, m


def test_track_chunking(c2):
    """K-2.7: three chunkings of the same 300 blocks are bit-identical; clocks follow the repeated sum."""
    trk, a = _hello(c2)
    ones, clocks = [], []
    for _ in range(N_BLOCKS):
        ones.append(a.advance(1))
        clocks.append(a.clock)
    _, b = _hello(c2)
    whole = b.advance(N_BLOCKS)
    _, c = _hello(c2)
    parts, left = [], N_BLOCKS
    while left:
        parts.append(c.advance(min(77, left)))
        left -= parts[-1].n_blocks
    c2.check_chunk_shapes(whole, N_BLOCKS, trk.n_ph + trk.n_or)
    assert c2.chunk_equal(c2.concat_chunks(ones), whole) == []
    assert c2.chunk_equal(c2.concat_chunks(parts), whole) == []
    assert a.clock == b.clock == c.clock
    ref = 0.0
    for k, clk in enumerate(clocks, start=1):
        ref += c2.DT_48K
        assert clk == ref, (k, clk, ref)
    for i, blk in enumerate(trk.float_blocks.tolist()):
        if blk < N_BLOCKS:
            assert clocks[blk] == trk.floats["clock"][i], (blk, clocks[blk], trk.floats["clock"][i])


def test_clock_is_not_k_times_dt(c2):
    """Guard: inside the tested range the repeated sum and k*dt really differ (else the clock checks prove nothing)."""
    diff = [k for k in range(1, N_BLOCKS + 1) if c2.repeated_clock(k) != k * c2.DT_48K]
    assert len(diff) > 10, diff


def test_advance_zero_blocks(c2):
    trk, m = _hello(c2)
    m.advance(5)
    clk, f0 = m.clock, m.f0
    empty = m.advance(0)
    c2.check_chunk_shapes(empty, 0, trk.n_ph + trk.n_or)
    assert (m.clock, m.f0) == (clk, f0)
    _, ref = _hello(c2)
    ref.advance(5)
    assert c2.chunk_equal(m.advance(20), ref.advance(20)) == []


def test_settle_pitch_model(c2):
    """K-2.13: settle_pitch sets the glide state to source.f0, so the first row already carries it; without it the
    glide starts at 110 Hz (human_voice.rs:225, :499-501)."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel
    from chimera_vocal.core.body import BODY_FEMALE

    idx = c2.tract_index("female")
    m = ControlModel(BODY_FEMALE, idx, geometry=GEOMETRY_VEC)
    m.set_source({"f0": 220.0})
    assert m.f0 == 110.0
    m.settle_pitch()
    assert m.f0 == 220.0 and m.source.f0 == 220.0
    rows = m.advance(10)
    assert (rows.glot_par[:, G_F0] == 220.0).all(), rows.glot_par[:, G_F0]
    n = ControlModel(BODY_FEMALE, idx, geometry=GEOMETRY_VEC)
    n.set_source({"f0": 220.0})
    first = n.advance(1).glot_par[0, G_F0]
    assert 110.0 < first < 220.0


def test_set_source_names(c2):
    """set_source takes SourceParams names and the template's serde names; voiced is a bool."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel
    from chimera_vocal.core.body import BODY_MALE

    m = ControlModel(BODY_MALE, c2.tract_index("male"), geometry=GEOMETRY_VEC)
    m.set_source({"vibratoDepth": 30.0, "pressure": 9.0})
    assert (m.source.vibrato_depth, m.source.pressure) == (30.0, 9.0)
    m.set_source({"vibrato_depth": 20.0, "voiced": False})
    assert (m.source.vibrato_depth, m.source.voiced) == (20.0, False)


@pytest.mark.parametrize("k", [0, 1, 7, 300, 1234])
def test_set_clock_block(c2, k):
    """K-2.13: clock = 0.0 plus dt added k times, bit for bit."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    from chimera_vocal.control.track import ControlModel
    from chimera_vocal.core.body import BODY_MALE

    m = ControlModel(BODY_MALE, c2.tract_index("male"), geometry=GEOMETRY_VEC)
    m.set_clock_block(k)
    assert m.clock == c2.repeated_clock(k)


def test_clock_block_start_matches_continuous_voice(c2):
    """K-2.13: hello-male shifted by 0.5 s (silent for the first 300 blocks). Voice A runs 300 + 200 blocks; voice B
    starts at block 300 (set_clock_block) and runs 200: B's rows == A's last 200 rows, bit for bit."""
    from chimera_vocal.control.geometry import GEOMETRY_VEC

    k = 300
    assert c2.repeated_clock(k) != k * c2.DT_48K
    trk = c2.decode_track("hello-male")
    shifted = []
    for o in trk.scenario["ops"]:
        if o["op"] == "schedule":
            for g in o["gestures"]:
                d = dict(g)
                d["t0"], d["t1"] = g["t0"] + 0.5, g["t1"] + 0.5
                shifted.append(Gesture.from_json(d))
    assert min(g.t0 - g.ramp_in for g in shifted) > k * c2.DT_48K + 0.01

    def voice():
        m = c2.new_model(trk, GEOMETRY_VEC)
        m.set_source({"f0": 112.0})
        m.settle_pitch()
        m.schedule(shifted)
        return m

    a = voice()
    a.advance(k)
    tail = a.advance(200)
    b = voice()
    b.set_clock_block(k)
    head = b.advance(200)
    assert c2.chunk_equal(tail, head) == []
    assert a.clock == b.clock
