"""K-3.4 test_kernel_on_recorded_track, K-3.5 test_kernel_differential, K-3.6 test_backend_swap.

The template's control track (V3) drives the kernel directly (no control model); the audio is compared with the
template's audio heads (V7, T3 over the first 4 800 samples). numba runs in a subprocess (dsp_numba)."""

from __future__ import annotations

import numpy as np
import pytest

import dsp_assert
import dsp_cases
import recorded_control as rc
import support

HEADS = {"sustain-aa-male": "voice_male_110hz_aa-back", "hello-male": "speech_male_hello_world"}
N_HEAD = 4800


@pytest.mark.parametrize("scenario", list(HEADS))
def test_kernel_on_recorded_track(kernel_name, scenario, run):
    """K-3.4: V3 track -> kernel -> output chain = template audio head (T3)."""
    ref = support.f32(support.audio_head(HEADS[scenario]))
    assert ref.shape == (N_HEAD,)
    res = run(kernel_name, "track_audio", scenario=scenario, n_out=N_HEAD)
    dsp_assert.t3(ref, res["audio"], f"{scenario} ({kernel_name})")


@pytest.mark.parametrize("scenario", list(HEADS))
def test_voice_on_recorded_track(kernel_name, scenario, run):
    """K-3.4 through HumanVoice: recorded control model, the scenario's ops, host blocks of 128 (the exporter's)."""
    ref = support.f32(support.audio_head(HEADS[scenario]))
    res = run(kernel_name, "voice_track", scenario=scenario, n=N_HEAD, block=128)
    dsp_assert.t3(ref, res["audio"], f"{scenario} via HumanVoice ({kernel_name})")


def test_recorded_track_covers_heads():
    """The in-repo float rows cover the control steps of the 4 800-sample heads (75 blocks)."""
    for s in HEADS:
        assert rc.track(s).float_head() >= N_HEAD // 64


@pytest.mark.numba
@pytest.mark.parametrize("n_internal", [1, 128, 9600])
def test_kernel_differential_state(n_internal, run):
    """K-3.5 (T5): numba vs numpy state after 1 / 128 / 9 600 internal samples of the /ɑ/ track."""
    a = run("numpy", "state_after", scenario="sustain-aa-male", n_internal=n_internal)
    b = run("numba", "state_after", scenario="sustain-aa-male", n_internal=n_internal)
    dsp_assert.assert_state_t5(a, b, f"state after {n_internal} samples")
    rr = support.compare_audio(a["raw"].astype(np.float32), b["raw"].astype(np.float32))
    assert rr.max_err <= 1e-9, f"raw radiated values differ by {rr.max_err:.2e} of peak"


@pytest.mark.numba
@pytest.mark.parametrize("scenario", list(HEADS))
def test_kernel_differential_heads(scenario, run):
    """K-3.5 (T3): numba vs numpy audio over the in-repo track heads (hello: /h/ aspiration noise included)."""
    a = run("numpy", "track_audio", scenario=scenario, n_out=N_HEAD)
    b = run("numba", "track_audio", scenario=scenario, n_out=N_HEAD)
    dsp_assert.t3(a["audio"], b["audio"], f"{scenario} numba vs numpy")
    dsp_assert.assert_state_t5(a, b, f"{scenario} final state")


@pytest.mark.numba
@pytest.mark.lab
@pytest.mark.parametrize("scenario", rc.SCENARIOS)
def test_kernel_differential_full(scenario, run):
    """K-3.5 (T3): numba vs numpy over the full V3 tracks (recorded control)."""
    trk = rc.track(scenario, full=True)
    a = run("numpy", "track_audio", scenario=scenario, n_out=trk.n_samples, full=True)
    b = run("numba", "track_audio", scenario=scenario, n_out=trk.n_samples, full=True)
    dsp_assert.t3(a["audio"], b["audio"], f"{scenario} numba vs numpy (full)")


@pytest.mark.numba
@pytest.mark.skipif(not dsp_cases.control_available(), reason="needs the control package")
@pytest.mark.parametrize("name", [s["name"] for s in support.golden("voice")])
def test_kernel_differential_golden(name, run):
    """K-3.5 (T3): numba vs numpy over the template's golden voice scenarios in full length, real control."""
    a = run("numpy", "golden", name=name)
    b = run("numba", "golden", name=name)
    dsp_assert.t3(a["audio"], b["audio"], f"golden {name} numba vs numpy")


@pytest.mark.numba
@pytest.mark.parametrize("scenario", list(HEADS))
def test_backend_swap(scenario, run, numba_cache, tmp_path):
    """K-3.6 (T5): kernel alternating numpy/numba per control block = pure numba run (state and audio)."""
    import dsp_numba

    swap = dsp_numba.run_subprocess("swap", {"kernel_names": ["numpy", "numba"], "scenario": scenario,
                                             "n_out": N_HEAD}, cache_dir=numba_cache, tmp_path=tmp_path)
    pure = dsp_numba.run_subprocess("swap", {"kernel_names": ["numba"], "scenario": scenario, "n_out": N_HEAD},
                                    cache_dir=numba_cache, tmp_path=tmp_path)
    dsp_assert.assert_state_t5(swap, pure, f"{scenario} swapped vs numba")
    dsp_assert.t3(pure["audio"], swap["audio"], f"{scenario} swapped vs numba")
