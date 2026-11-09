"""HumanVoice (INTERFACES §6.5): K-3.7 test_blocksize_independence, K-3.12 test_rates, K-3.14 test_settle_pitch,
K-3.15 test_start_block, and the performer-command contract. Control comes from test models
(tests/dsp/recorded_control.py) through voice.CONTROL_FACTORY; the "real" variants need C2 (S3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import dsp_assert
import dsp_cases
import recorded_control as rc
from chimera_vocal.core import glottis_params as gp
from chimera_vocal.core.body import BODY_FEMALE, BODY_MALE, BodyMix
from chimera_vocal.core.types import Seeds, SourceParams
from chimera_vocal.dsp import layout as L
from chimera_vocal.dsp import network, voice
from chimera_vocal.dsp.kernel_numpy import KERNEL_NUMPY
from support import vector

FEMALE_POSTURE = ("song-female", 369)     # a held sung vowel of the female track (in-repo float row)
N_HEAD = 4800


@pytest.fixture
def factory(monkeypatch):
    """factory(f) installs f as voice.CONTROL_FACTORY for this test."""

    def install(f):
        monkeypatch.setattr(voice, "CONTROL_FACTORY", f)
        return f

    return install


# ------------------------------------------------------------------------------------------- K-3.7

@pytest.mark.parametrize("scenario", ["sustain-aa-male", "hello-male"])
def test_blocksize_independence(kernel_name, scenario, run):
    """K-3.7 (T6): render in host blocks of 1 / 77 / 128 / 4096 samples, bit-identical per kernel."""
    ref = run(kernel_name, "voice_track", scenario=scenario, n=N_HEAD, block=128)
    for block in (1, 77, 4096):
        got = run(kernel_name, "voice_track", scenario=scenario, n=N_HEAD, block=block)
        dsp_assert.assert_bits(ref["audio"], got["audio"], f"{scenario} block {block} ({kernel_name})")
        dsp_assert.assert_state_equal(ref, got, f"{scenario} block {block} ({kernel_name})")


def test_control_steps_follow_the_counter():
    st, idx = network.human_net(BODY_MALE, 96000.0, Seeds())
    network.configure_output(st, 48000.0, 2)
    cases = {0: {1: 1, 2: 1, 128: 1, 129: 2, 256: 2, 257: 3},
             5: {1: 0, 5: 0, 6: 1, 133: 1, 134: 2},
             128: {128: 0, 129: 1}}
    for counter, want in cases.items():
        st.si[L.SI_CTRL_COUNTER] = counter
        for n, k in want.items():
            assert voice.control_steps(st, n) == k, (counter, n)


# ------------------------------------------------------------------------------------------- K-3.12

@pytest.mark.parametrize("fs_out, m", [(48000.0, 2), (96000.0, 1)])
def test_rates_allowed(fs_out, m, factory):
    factory(rc.steady_factory())
    v = voice.HumanVoice(fs_out, BodyMix(), m)
    t = vector("topology")["bodies"][0]
    assert v.index.fs == 96000.0
    assert (v.index.n_ph, v.index.n_or, v.index.n_nodes) == (t["n_ph"], t["n_or"], t["n_nodes"])
    assert v.state.si[L.SI_M] == m and v.state.si[L.SI_CTRL_EVERY] == 64 * m
    assert v.state.dec_c.shape == ((49,) if m == 2 else (1,))
    out = v.render(640)
    assert out.dtype == np.float32 and out.shape == (640,) and np.isfinite(out).all()


@pytest.mark.parametrize("fs_out, m", [(44100.0, 2), (48000.0, 1), (96000.0, 2), (24000.0, 4), (22050.0, 2),
                                       (32000.0, 3), (48000.0, 0)])
def test_rates_rejected(fs_out, m, factory):
    def boom(*a, **k):
        raise AssertionError("control model built for a rejected rate")

    factory(boom)
    with pytest.raises(ValueError):
        voice.HumanVoice(fs_out, BodyMix(), m)


# ------------------------------------------------------------------------------------------- K-3.14

def _female_posture():
    return rc.track(FEMALE_POSTURE[0]).posture(FEMALE_POSTURE[1])


def _first_voicing_cents(res, f0, window_s=0.1, block=64, fs_out=48000.0):
    v = res["voicing"]
    on = int(np.argmax(v >= 0.5))
    assert v[on] >= 0.5, "never voiced"
    n = int(round(window_s * fs_out / block))
    c = np.abs(rc.cents(res["f0_now"][on:on + n], f0))
    return float(np.percentile(c, 90))


@pytest.mark.parametrize("control_kind", ["steady", "real"])
def test_settle_pitch(kernel_name, control_kind, run):
    """K-3.14: female 205 Hz — f0 of the first 100 ms of voicing p90 ≤ 20 ct with settle_pitch(), > 60 ct without."""
    if control_kind == "real" and not dsp_cases.control_available():
        pytest.skip("S3: needs C2's ControlModel")
    kw = dict(body="female", f0=205.0, n=24000, control=control_kind, posture=FEMALE_POSTURE)
    with_settle = run(kernel_name, "voice_steady", settle=True, **kw)
    without = run(kernel_name, "voice_steady", settle=False, **kw)
    p_with = _first_voicing_cents(with_settle, 205.0)
    p_without = _first_voicing_cents(without, 205.0)
    assert p_with <= 20.0, f"with settle_pitch: p90 {p_with:.1f} ct"
    assert p_without > 60.0, f"without settle_pitch: p90 {p_without:.1f} ct (test would not detect a missing settle)"
    if control_kind == "steady":
        assert ["settle_pitch"] in json.loads(str(with_settle["calls"]))


def test_settle_pitch_writes_glottis_f0(factory):
    factory(rc.steady_factory(_female_posture()))
    v = voice.HumanVoice(48000.0, BodyMix.parse("female"), 2)
    v.set_source({"f0": 205.0})
    v.settle_pitch()
    assert v.state.glot[L.G_F0] == 205.0
    assert v.control.f0 == 205.0 and ("settle_pitch",) in v.control.calls


# ------------------------------------------------------------------------------------------- K-3.15

KS = (1, 7, 750, 12345)


def test_start_block_ks_discriminate():
    """The chosen k include values where k·dt differs from k additions of dt (so 'clock = k·dt' is caught)."""
    dt = 128.0 / 96000.0
    sums = []
    for k in KS:
        c = 0.0
        for _ in range(k):
            c += dt
        sums.append(c != k * dt)
    assert any(sums)


@pytest.mark.parametrize("k", KS)
def test_start_block(k, factory):
    """K-3.15 (T6): HumanVoice(start_block=k).clock is bit-identical to a continuous voice after k blocks, and stays
    so while both render on."""
    factory(rc.steady_factory())
    cont = voice.HumanVoice(48000.0, BodyMix(), 2)
    late = voice.HumanVoice(48000.0, BodyMix(), 2, start_block=k)
    if k <= 750:
        cont.render(64 * k)
    else:
        cont.control.set_clock_block(k)    # the steady model's clock is the same repeated sum (cheaper than rendering)
    assert late.clock == cont.clock, (late.clock, cont.clock)
    assert ("set_clock_block", k) in late.control.calls
    for _ in range(3):
        cont.render(64)
        late.render(64)
        assert late.clock == cont.clock


def test_start_block_zero_is_default(factory):
    factory(rc.steady_factory())
    v = voice.HumanVoice(48000.0, BodyMix(), 2, start_block=0)
    assert v.clock == 0.0
    assert not any(c[0] == "set_clock_block" and c[1] != 0 for c in v.control.calls)


# ------------------------------------------------------------------------------------------- commands

def test_set_source_takes_effect_at_once(factory):
    """set_source writes the masked apply_source values into the glottis immediately (human_voice.rs:283-286),
    mid-block; f0, closure and chaos stay (set per control step)."""
    factory(rc.steady_factory())
    v = voice.HumanVoice(48000.0, BodyMix(), 2)
    v.set_gate(1.0)
    v.render(100)                       # mid-block: next control step is not due
    before = v.state.glot.copy()
    v.set_source({"breathiness": 0.6, "roughness": 0.8, "tension": 0.9})
    src = SourceParams().patched(breathiness=0.6, roughness=0.8, tension=0.9)
    values, mask, _ = gp.apply_source_params(src, BODY_MALE.scale)
    g = v.state.glot
    for i in range(gp.N_GLOTTIS):
        if mask[i]:
            assert g[i] == values[i], gp.GLOTTIS_FIELDS[i]
        else:
            assert g[i] == before[i], gp.GLOTTIS_FIELDS[i]
    assert ("set_source", {"breathiness": 0.6, "roughness": 0.8, "tension": 0.9}) in v.control.calls
    assert v.control.source == src


def test_commands_reach_the_control_model(factory):
    from chimera_vocal.core.types import Gesture
    factory(rc.steady_factory())
    v = voice.HumanVoice(48000.0, BodyMix(), 2)
    g = Gesture(0.1, 0.2, 0.05, 0.05, 1.0, {"jaw": 0.5})
    v.set_gate(0.7)
    v.set_articulation({"jaw": 1.0}, True)
    v.schedule([g])
    v.release(3.0)
    v.cancel_score()
    names = [c[0] for c in v.control.calls]
    for want in ("set_gate", "set_articulation", "schedule", "release", "cancel_score"):
        assert want in names, want
    assert ("set_articulation", {"jaw": 1.0}, True) in v.control.calls


def test_render_into_given_buffer(factory):
    factory(rc.steady_factory())
    v = voice.HumanVoice(48000.0, BodyMix(), 2)
    v.set_gate(1.0)
    buf = np.full(300, np.nan, dtype=np.float32)
    got = v.render(300, buf)
    assert got is buf and np.isfinite(buf).all()
    assert v.render(0).shape == (0,)


def test_default_kernel_and_body(factory):
    factory(rc.steady_factory())
    v = voice.HumanVoice()
    assert v.index.n_ph == vector("topology")["bodies"][0]["n_ph"]
    assert voice.HumanVoice.__init__.__kwdefaults__["kernel"] is KERNEL_NUMPY
    f = voice.HumanVoice(48000.0, BODY_FEMALE, 2)
    assert f.index.n_ph == vector("topology")["bodies"][1]["n_ph"]
