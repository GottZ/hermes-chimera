"""K-2.12: filter-state inheritance of the noise sources (network.rs:725-745 `find` = FIRST match on (src, fwd);
human_voice.rs:468-474 two set_noise_sources calls in a block with contact re-application; REVIEW-1 P1-01).

Constructed cases (exact) plus a replay over every block of the four V3 tracks: for each block, the composition of
its calls (list A from `list_a` where exported, then the final list) against the previous block's final list must
give the exported `noise_inherit`.
"""

G = (70, 13, 11)            # glottal aspiration source of the male tract (V3 hello-male)
C1 = (45, 45, 44)           # a constriction source
C2 = (42, 44, 40)           # another one
T1 = (45, 50, 44)           # a teeth source sharing src with C1


def test_one_call_no_reapply():
    from chimera_vocal.control.airway import block_inherit, inherit_from

    assert inherit_from([G], [G, C1]) == [0, -1]
    assert block_inherit([G], [[G, C1]]) == [0, -1]
    assert block_inherit([G, C1], [[G, C1]]) == [0, 1]
    assert block_inherit([G, C1], [[C1, G]]) == [1, 0]           # order of the new list decides, not position
    assert block_inherit([G, C1], [[G]]) == [0]
    assert block_inherit([], [[G]]) == [-1]


def test_bwd_and_parameters_do_not_matter():
    """Only (src, fwd) identifies a source (network.rs:729): bwd and gains/bands are free to change."""
    from chimera_vocal.control.airway import inherit_from

    assert inherit_from([(45, 45, 44, 1e-6, 2600.0, 350.0)], [(45, 45, 30, 2e-6, 5000.0, 1200.0)]) == [0]
    assert inherit_from([(45, 45, 44)], [(45, 46, 44)]) == [-1]
    assert inherit_from([(45, 45, 44)], [(46, 45, 44)]) == [-1]


def test_reapply_with_identical_lists():
    from chimera_vocal.control.airway import block_inherit

    for prev, a in (([G], [G, C1]), ([G, C1], [G, C1]), ([G, C2], [G, C1, T1])):
        assert block_inherit(prev, [a, a]) == block_inherit(prev, [a])


def test_pair_in_previous_and_final_but_not_in_a_resets():
    """(src, fwd) in P ∩ B but not in A: the state was dropped by the first call -> -1 (the P1-01 case;
    'final list against the previous block' would wrongly give 1)."""
    from chimera_vocal.control.airway import block_inherit, inherit_from

    prev, a, b = [G, C1], [G], [G, C1]
    assert block_inherit(prev, [a, b]) == [0, -1]
    assert inherit_from(prev, b) == [0, 1]                       # the wrong rule, for contrast
    prev, a, b = [G, C1, C2], [G, C2], [G, C1, C2]
    assert block_inherit(prev, [a, b]) == [0, -1, 2]


def test_first_match_on_duplicate_pairs():
    """A list can hold two sources with the same (src, fwd) (constriction and teeth source when the teeth branch
    equals the forward branch). `find` takes the first one."""
    from chimera_vocal.control.airway import block_inherit, compose_inherit, inherit_from

    dup_a, dup_b = (45, 50, 44), (45, 50, 40)
    assert inherit_from([G, dup_a, dup_b], [dup_b]) == [1]
    assert inherit_from([dup_a, G, dup_b], [G, dup_a]) == [1, 0]
    assert block_inherit([G, dup_a, dup_b], [[dup_b, G]]) == [1, 0]
    # duplicates in the intermediate list A: the final list takes A's first match, which itself came from P
    assert block_inherit([dup_a, G], [[G, dup_a, dup_b], [dup_b]]) == [0]
    assert compose_inherit([3, -1, 0], [2, 1, 0, -1]) == [0, -1, 3, -1]


def test_track_replay_all_blocks(c2):
    """Every block of the four V3 tracks: block_inherit(final list of b-1, calls of b) == noise_inherit[b]; calls =
    [A, B] where the export lists an intermediate list A that differs from B, else [B]. Block 0 inherits from the
    list of `initial`. Also: the P1 rule reproduces the exported wrong values at the critical blocks."""
    from chimera_vocal.control.airway import block_inherit, inherit_from

    checked = 0
    for s in c2.TRACK_SCENARIOS:
        trk = c2.decode_track(s)
        prev = [tuple(e) for e in trk.initial["noise_idx"][: trk.initial["noise_n"]]]
        for b in range(trk.n_blocks):
            final = [tuple(e) for e in trk.noise_idx[b][: trk.noise_n[b]]]
            calls = [trk.list_a[b], final] if b in trk.list_a else [final]
            want = list(trk.noise_inherit[b][: trk.noise_n[b]])
            assert block_inherit(prev, calls) == want, (s, b, prev, calls, want)
            if b in trk.p1_wrong:
                assert inherit_from(prev, final) == trk.p1_wrong[b], (s, b)
            prev = final
            checked += 1
    assert checked == 750 + 904 + 2909 + 7725


def test_compose_maps_resets_through():
    """compose_inherit: a reset (-1) in the second call stays a reset; valid entries index the first mapping."""
    from chimera_vocal.control.airway import compose_inherit

    assert compose_inherit([0, 1], []) == []
    assert compose_inherit([0, 1], [-1, -1]) == [-1, -1]
    assert compose_inherit([-1, 4], [1, 0, -1]) == [4, -1, -1]
