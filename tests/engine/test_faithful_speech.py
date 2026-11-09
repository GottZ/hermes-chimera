"""The public faithful switch renders text sample for sample like the template (``lab``, T3).

``Engine.speak(text, body, faithful=True)`` against the native ``chimera render text`` renders of HEAD 092b16ce12b9
(``$CHIMERA_LAB_REFS/ref-audio/text-{male,female,child}.wav``, float WAV): max |Δ| <= 1e-6 × peak over the whole
file, length exact, for both kernels. ``speak_to_wav(..., faithful=True)`` writes the same audio as 16-bit PCM.
The default (production) mode is measured against the same files to show the switch matters (measured
correlation < 0.9 for text-male/child). Red proof: 9a616be has no switch; ``speak`` without it is not sample-equal."""

from __future__ import annotations

import numpy as np
import pytest

from chimera_vocal import Engine, EngineConfig
from chimera_vocal.core.wavio import read_wav, wav_bytes
from support import assert_t3, compare_audio
from support.fakes import real_engine_available, wait_until

pytestmark = [pytest.mark.lab,
              pytest.mark.skipif(not real_engine_available(), reason="needs the real text, plan, control and dsp code")]
TEXT = "Hello there, my name is Chimera. Can you hear me?"     # $CHIMERA_LAB_REFS/ref-audio/manifest.md


@pytest.fixture(scope="module")
def numba_cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("numba-cache"))


@pytest.fixture
def real_engine(kernel_name, numba_cache, monkeypatch):
    for k in ("CHIMERA_WORKER_VOICE_FACTORY", "CHIMERA_TESTING", "CHIMERA_BACKEND", "CHIMERA_FAKE"):
        monkeypatch.delenv(k, raising=False)
    from chimera_vocal.music.compile import set_voice_factory
    from chimera_vocal.music.planning import set_planner_api
    set_planner_api(None)
    set_voice_factory(None)
    eng = Engine(EngineConfig(backend=kernel_name, workers=2, cache_dir=numba_cache))
    if kernel_name == "numba":
        eng.warmup()
        assert wait_until(lambda: eng.status().numba_state in ("ready", "failed"), 900, 0.5) is not None
        assert eng.status().numba_state == "ready", eng.status()
    yield eng
    eng.shutdown()


@pytest.mark.parametrize("body", ["male", "female", "child"])
def test_faithful_speak_is_render_text(lab_refs, real_engine, kernel_name, body, tmp_path):
    sr, ref = read_wav(lab_refs / "ref-audio" / f"text-{body}.wav")
    assert sr == 48000
    audio, kernel = real_engine.speak(TEXT, body, faithful=True)
    assert kernel == kernel_name
    d = assert_t3(ref, audio.samples, f"text-{body} faithful ({kernel_name})")
    print(f"text-{body} {kernel_name}: max |delta| {d.max_err:.3e} x peak over {len(ref)} samples")
    p = tmp_path / "f.wav"
    res = real_engine.speak_to_wav(TEXT, str(p), body, faithful=True)
    assert res.kernel == kernel_name and p.read_bytes() == wav_bytes(audio.samples, 48000)
    if body != "female":   # the production mode differs from the template (phrase windows, settled pitch)
        default, _ = real_engine.speak(TEXT, body)
        assert len(default.samples) == len(ref)
        assert compare_audio(ref, default.samples).first_diverge is not None
        assert float(np.corrcoef(ref, default.samples.astype(np.float64))[0, 1]) < 0.9
