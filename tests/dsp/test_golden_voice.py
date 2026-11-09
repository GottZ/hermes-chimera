"""K-3.9 test_golden_voice_head (S3), K-3.10 test_golden_voice_full (lab; S2 with the recorded V3 tracks, S3 with
the real control model) and K-3.10r — the red proof that T3 over whole files catches the noise-filter inheritance
error P1-01 that a 10-ms-envelope tolerance would let through (TESTPLAN §1, REVIEW-1 P1-02)."""

from __future__ import annotations

import numpy as np
import pytest

import dsp_assert
import dsp_cases
import recorded_control as rc
import support

GOLDEN = [s["name"] for s in support.golden("voice")]
NEEDS_CONTROL = pytest.mark.skipif(not dsp_cases.control_available(), reason="S3: needs C2's ControlModel")

# K-3.10r expectations [ausgeführt S0-V, tests/vectors/README.md "Self-check"]: template emulation of P1-01 on
# hello-male: first bit difference 40 064, first > 1e-6 x peak at 40 066, max 1.2745e-2 x peak, envelope <= 0.0006 dB
P1_HELLO = {"first_from": 40064, "first_to": 40200, "max_err_min": 1e-3, "env_db_max": 0.05}
P1_SONG = {"first_from": 67000, "first_to": 68000, "max_err_min": 1e-3}


# ------------------------------------------------------------------------------------------- K-3.9 (S3)

@NEEDS_CONTROL
@pytest.mark.parametrize("name", GOLDEN)
def test_golden_voice_head(kernel_name, name, run):
    """K-3.9: the template's golden scenarios with the real control model, first 4 800 samples, T3."""
    head = support.golden(f"voice-{name}.head")
    ref = support.f32(head["head"])
    res = run(kernel_name, "golden", name=name, n=head["head_samples"])
    dsp_assert.t3(ref, res["audio"], f"golden {name} head ({kernel_name})")


@NEEDS_CONTROL
@pytest.mark.parametrize("name", GOLDEN)
def test_golden_voice_blocks(kernel_name, name, run):
    """golden_voice.rs: host blocks 1 / 77 / 4096 give the same samples as 128 (first 4 800, real control)."""
    ref = run(kernel_name, "golden", name=name, n=4800, block=128)
    for block in (1, 77, 4096):
        got = run(kernel_name, "golden", name=name, n=4800, block=block)
        dsp_assert.assert_bits(ref["audio"], got["audio"], f"golden {name} block {block}")


# ------------------------------------------------------------------------------------------- K-3.10 (lab)

@pytest.mark.lab
@pytest.mark.parametrize("scenario", rc.SCENARIOS)
def test_track_full(kernel_name, scenario, run):
    """K-3.10 (S2): the full V3 track through HumanVoice (recorded control) = the template's audio of that track,
    T3 over the whole file, length exact. sustain = r1/voice_male_110hz_aa-back_1s, hello = r1/speech_male_hello_world
    (57 845), stop = audio/stop_male_text, song-female = ref-audio twinkle (byte-identical, S0-V)."""
    trk = rc.track(scenario, full=True)
    ref = trk.lab_audio()
    assert ref.shape == (trk.n_samples,)
    res = run(kernel_name, "voice_track", scenario=scenario, n=trk.n_samples, block=128, full=True)
    d = dsp_assert.t3(ref, res["audio"], f"{scenario} full ({kernel_name})")
    print(f"K-3.10 {scenario} {kernel_name}: n {trk.n_samples} peak {d.peak:.4g} max_err {d.max_err:.3e} "
          f"env {d.env_db:.5f} dB")


@pytest.mark.lab
def test_lab_track_audio_is_template_render(lab_refs):
    """The lab audio of hello-male is the R1 render of "Hello world." and stop-male equals stop_male_text."""
    a = np.fromfile(lab_refs / "audio" / "track_hello-male.f32", dtype="<f4")
    b = np.fromfile(lab_refs / "r1" / "speech_male_hello_world.f32", dtype="<f4")
    assert a.tobytes() == b.tobytes() and a.shape == (57845,)
    c = np.fromfile(lab_refs / "audio" / "track_stop-male.f32", dtype="<f4")
    e = np.fromfile(lab_refs / "audio" / "stop_male_text.f32", dtype="<f4")
    assert c.tobytes() == e.tobytes()


@pytest.mark.lab
@NEEDS_CONTROL
@pytest.mark.parametrize("name", GOLDEN)
def test_golden_voice_full(kernel_name, name, run, lab_refs):
    """K-3.10 (S3): the six golden scenarios in full length with the real control model, T3 whole file."""
    ref = np.fromfile(lab_refs / "golden" / f"voice-{name}.f32", dtype="<f4")
    res = run(kernel_name, "golden", name=name)
    dsp_assert.t3(ref, res["audio"], f"golden {name} full ({kernel_name})")


@pytest.mark.lab
@NEEDS_CONTROL
@pytest.mark.parametrize("scenario", ["hello-male", "stop-male"])
def test_text_full_real_control(kernel_name, scenario, run):
    """K-3.10 (S3): speech_male_hello_world and stop-male from their recorded inputs (set_source + the planned
    gestures, track scenario ops) with the real control model, T3 whole file."""
    trk = rc.track(scenario, full=True)
    res = run(kernel_name, "voice_ops_real", scenario=scenario)
    dsp_assert.t3(trk.lab_audio(), res["audio"], f"{scenario} real control ({kernel_name})")


# ------------------------------------------------------------------------------------------- K-3.10r (lab)

@pytest.mark.lab
def test_p1_red_proof_hello(kernel_name, run):
    """K-3.10r: replaying hello-male with "final list against the previous block" inheritance is RED under T3
    (first > 1e-6 × peak near sample 40 066, max ≈ 1.3e-2 × peak), while a ±0.05 dB 10-ms envelope would pass."""
    trk = rc.track("hello-male", full=True)
    assert (trk.p1 != trk.noise_inherit).any(axis=1).nonzero()[0].tolist() == [626]
    res = run(kernel_name, "voice_track", scenario="hello-male", n=trk.n_samples, block=128, full=True, inherit="p1")
    d = support.compare_audio(trk.lab_audio(), res["audio"], 1e-6)
    print(f"K-3.10r hello-male {kernel_name}: first > 1e-6 x peak at {d.first_diverge}, max {d.max_err:.4e} x peak, "
          f"envelope {d.env_db:.5f} dB")
    assert d.first_diverge is not None, "P1 emulation not detected by T3"
    assert P1_HELLO["first_from"] <= d.first_diverge <= P1_HELLO["first_to"], d.first_diverge
    assert d.max_err >= P1_HELLO["max_err_min"], d.max_err
    assert d.env_db <= P1_HELLO["env_db_max"], f"envelope {d.env_db} dB: a 0.05 dB envelope check would be red too"


@pytest.mark.lab
@pytest.mark.slow
def test_p1_red_proof_song(kernel_name, run):
    """K-3.10r on song-female (17 P1-critical blocks): T3 red from ≈ 67 523, max ≈ 8.4e-3 × peak."""
    trk = rc.track("song-female", full=True)
    res = run(kernel_name, "voice_track", scenario="song-female", n=trk.n_samples, block=128, full=True,
              inherit="p1")
    d = support.compare_audio(trk.lab_audio(), res["audio"], 1e-6)
    print(f"K-3.10r song-female {kernel_name}: first {d.first_diverge}, max {d.max_err:.4e}, env {d.env_db:.5f} dB")
    assert d.first_diverge is not None
    assert P1_SONG["first_from"] <= d.first_diverge <= P1_SONG["first_to"], d.first_diverge
    assert d.max_err >= P1_SONG["max_err_min"]
