"""K-0.8: apply_source_params = HumanVoice::apply_source (human_voice.rs:308-341), V5 glottis.json "params"."""

import numpy as np

from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.glottis_params import (DEFAULT_GLOTTIS, G_CHAOS, G_CLOSURE, G_EFFORT, G_F0, GLOTTIS_FIELDS,
                                               WRITTEN, apply_source_params)
from chimera_vocal.core.types import SourceParams
from support import vector


def test_field_order_and_default():
    v = vector("glottis")
    assert tuple(v["field_order"]) == GLOTTIS_FIELDS
    assert tuple(v["default_glottis"]) == DEFAULT_GLOTTIS


def test_params_match_template():
    cases = vector("glottis")["params"]
    assert len(cases) >= 12
    for c in cases:
        src = SourceParams.from_json(c["source"])
        assert src.to_json() == c["source"], c["name"]
        scale = BodyMix.parse(c["body_mix"]).body().scale
        assert scale == c["body_scale"], (c["name"], scale, c["body_scale"])
        values, mask, base_abd = apply_source_params(src, c["body_scale"])
        assert mask.tolist() == c["written"], c["name"]
        want = np.asarray(c["glottis"], dtype=np.float64)
        assert np.array_equal(values[mask], want[mask]), (c["name"], values[mask] - want[mask])
        assert np.isnan(values[~mask]).all()
        assert base_abd == c["glottis"][2]


def test_mask_is_exactly_apply_source():
    _, mask, _ = apply_source_params(SourceParams(), 1.0)
    assert sorted(np.flatnonzero(mask).tolist()) == sorted(WRITTEN)
    assert not mask[G_F0] and not mask[G_CLOSURE] and not mask[G_CHAOS]


def test_effort_uses_absolute_pressure():
    """Rot-Beleg target: ingressive pressure -6 must give the same effort as +6."""
    a, _, _ = apply_source_params(SourceParams(pressure=-6.0), 1.0)
    b, _, _ = apply_source_params(SourceParams(pressure=6.0), 1.0)
    assert a[G_EFFORT] == b[G_EFFORT] == 0.3
