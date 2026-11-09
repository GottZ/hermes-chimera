"""K-2.5: tests/vectors/geometry9.json (3 bodies x /a ɑ i/, exported from the template: HumanVocalTract::compute_new
+ resample(18, 18); HumanAirway::new(96000).apply(artic, 0)) -- T2.

- frame scalars of each case (the array members of `frame` were trimmed from the repository copy; grids/outlines are
  covered by K-2.1),
- resample18 (areas, minima, lengths) through both CPython-side geometry forms,
- airway96k: the airway's section counts and slew-free section areas/lengths at 96 kHz internal (Airway.apply with
  dt = 0, section counts from the V2 TractIndex of the body).
`calibratedFormants` (frequency oracle of the template) is not part of the port and not checked.
"""

import pytest

import support

FRAME_KEYS = (("length", "length"), ("lipArea", "lip_area"), ("lipS", "lip_s"), ("pharynxLength", "pharynx_length"),
              ("splitIndex", "split_index"), ("teethS", "teeth_s"), ("tongueRadius", "tongue_radius"),
              ("velumPort", "velum_port"))
POINT_KEYS = (("glottis", "glottis_x", "glottis_y"), ("tip", "tip_x", "tip_y"), ("tongueCenter", "tongue_cx", "tongue_cy"))


def _cases():
    cases = support.vector("geometry9")["cases"]
    assert len(cases) == 9 and {c["body"] for c in cases} == {"male", "female", "child"}
    return cases


@pytest.mark.parametrize("impl_name", ["GEOMETRY_REF_CPYTHON", "GEOMETRY_VEC"])
def test_vectors_geometry9(c2, impl_name):
    from chimera_vocal.control import geometry

    impl = getattr(geometry, impl_name)
    failures = []
    for c in _cases():
        label = f"{c['body']} {c['ipa']}"
        a = c2.artic_of(c["artic"])
        got = c2.run_geometry(impl, c2.body_of(c["body"]), a, 18, 18)
        s, f = got["scalars"], c["frame"]
        mine = [s[k] for _, k in FRAME_KEYS] + [s[k] for _, kx, ky in POINT_KEYS for k in (kx, ky)]
        want = [f[k] for k, _ in FRAME_KEYS] + [f[k][xy] for k, _, _ in POINT_KEYS for xy in ("x", "y")]
        w = c2.worst_rel(mine, want, c2.T2_FLOOR)
        rs, r = c["resample18"], got["resample"]
        w = max(w, c2.worst_rel([r["phLen"], r["orLen"]], [rs["phLen"], rs["orLen"]], c2.T2_FLOOR))
        for key in ("ph", "or", "phMin", "orMin"):
            w = max(w, c2.worst_rel(r[key], rs[key], c2.T2_FLOOR))
        if not w < c2.T2_TOL:
            failures.append(f"{label}: worst {w:.3e}")
    assert not failures, failures


def test_vectors_airway96k(c2):
    from chimera_vocal.control.airway import Airway
    from chimera_vocal.control.geometry import GEOMETRY_REF_CPYTHON

    failures = []
    for c in _cases():
        label = f"{c['body']} {c['ipa']}"
        aw96 = c["airway96k"]
        assert aw96["fs"] == 96000.0
        idx = c2.tract_index(c["body"])
        airway = Airway(c2.body_of(c["body"]), idx, GEOMETRY_REF_CPYTHON)
        if (airway.n_ph, airway.n_or) != (aw96["nPh"], aw96["nOr"]):
            failures.append(f"{label}: sections {(airway.n_ph, airway.n_or)} != {(aw96['nPh'], aw96['nOr'])}")
            continue
        airway.apply(c2.artic_of(c["artic"]), 0.0)
        w = c2.worst_rel([airway.ph_len, airway.or_len], [aw96["phLen"], aw96["orLen"]], c2.T2_FLOOR)
        w = max(w, c2.worst_rel(airway.ph_areas, aw96["phAreas"], c2.T2_FLOOR))
        w = max(w, c2.worst_rel(airway.or_areas, aw96["orAreas"], c2.T2_FLOOR))
        if not w < c2.T2_TOL:
            failures.append(f"{label}: worst {w:.3e}")
    assert not failures, failures


def test_v2_section_counts_match_geometry9(c2):
    """The V2 TractIndex the tests use carries the template's section counts (floor rule, human_airway.rs:73-78)."""
    for c in _cases():
        idx = c2.tract_index(c["body"])
        assert (idx.n_ph, idx.n_or) == (c["airway96k"]["nPh"], c["airway96k"]["nOr"]), c["body"]
