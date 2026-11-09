"""K-0.6: bodies, body mixes, default pitch (anatomy/human.rs:262-415, engine.rs:24-35)."""

import math

import pytest

from chimera_vocal.core.body import (BODY_CHILD, BODY_FEMALE, BODY_MALE, BodyMix, HumanBody, default_f0,
                                     parse_body_name)


def test_presets_exact():
    assert (BODY_MALE.scale, BODY_MALE.pharynx, BODY_MALE.tongue_radius, BODY_MALE.lip_width) == (1.0, 1.0, 2.25, 4.6)
    assert (BODY_FEMALE.scale, BODY_FEMALE.pharynx, BODY_FEMALE.tongue_radius, BODY_FEMALE.lip_width) == (
        0.9, 0.88, 2.2, 4.4)
    assert (BODY_CHILD.scale, BODY_CHILD.pharynx, BODY_CHILD.tongue_radius, BODY_CHILD.lip_width) == (
        0.74, 0.8, 2.15, 4.0)
    for name, body in (("male", BODY_MALE), ("female", BODY_FEMALE), ("child", BODY_CHILD)):
        mix = BodyMix.parse(name)
        assert mix.preset() == name
        assert mix.body() is body
    assert default_f0("male") == 110.0 and default_f0("female") == 205.0 and default_f0("child") == 260.0


@pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("c", [0.0, 0.3, 0.5, 1.0])
def test_weights_sum_to_one(f, c):
    w = BodyMix(f, c).weights()
    assert abs(w[0] + w[1] + w[2] - 1.0) <= 1e-12
    assert w == ((1.0 - c) * (1.0 - f), (1.0 - c) * f, c)


def test_preset_is_exact_compare():
    assert BodyMix(1.0, 0.0).preset() == "female"
    assert BodyMix(0.999999, 0.0).preset() is None
    assert BodyMix(0.3, 1.0).preset() == "child"          # weights (0, 0, 1)
    assert BodyMix(0.0, 0.0).preset() == "male"
    assert BodyMix(0.5, 0.0).nearest() == "male"          # tie: first wins
    assert BodyMix(0.6, 0.0).nearest() == "female"


def test_default_f0_geometric_blend():
    """default_f0({female .5}) = exp(.5 ln 110 + .5 ln 205) = 150.1666 (not the linear 157.5)."""
    got = default_f0(BodyMix(0.5, 0.0))
    assert abs(got - math.sqrt(110.0 * 205.0)) <= 1e-12 * got
    assert round(got, 3) == 150.167
    teen = default_f0({"female": 0.5, "child": 0.5})
    want = math.exp(0.25 * math.log(110.0) + 0.25 * math.log(205.0) + 0.5 * math.log(260.0))
    assert abs(teen - want) <= 1e-12 * want and round(teen, 2) == 197.59


def test_blend_body_fold_order():
    mix = BodyMix(0.5, 0.5)
    wm, wf, wc = mix.weights()
    b = mix.body()
    assert b.scale == ((0.0 + wm * 1.0) + wf * 0.9) + wc * 0.74
    assert b.lip_width == ((0.0 + wm * 4.6) + wf * 4.4) + wc * 4.0
    assert b.name == "Child"                      # weights (0.25, 0.25, 0.5): nearest preset is the child
    assert isinstance(b, HumanBody)


def test_parse_forms():
    assert parse_body_name("female") == "female" and parse_body_name("robot") == "male"
    assert BodyMix.parse("robot") == BodyMix(0.0, 0.0)
    assert BodyMix.parse({"female": 2.0}) == BodyMix(1.0, 0.0)
    assert BodyMix.parse({"child": float("nan"), "female": -1}) == BodyMix(0.0, 0.0)
    m = BodyMix(0.2, 0.1)
    assert BodyMix.parse(m) is m
    with pytest.raises(TypeError):
        BodyMix.parse(3)
