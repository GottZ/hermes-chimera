"""TTS provider (INTERFACES §10; TESTPLAN K-6.2, K-6.3, K-6.4, K-6.13)."""

from __future__ import annotations

import threading

import pytest

import c5fakes


@pytest.fixture
def provider(registered):
    return registered.tts_providers[0]


# K-6.2: ".mp3" -> "<stem>.wav" (RIFF, 16 bit, mono, 48 kHz); ".ogg" -> WAV bytes exactly there
# (rot: WAV bytes under ".mp3")
def test_synthesize_paths(provider, plugin, tmp_path):
    out = tmp_path / "tts_20261109_141502.mp3"
    got = provider.synthesize("Hello world.", str(out), format="mp3")
    assert got == str(tmp_path / "tts_20261109_141502.wav")
    assert not out.exists(), "WAV bytes must never be written under .mp3"
    info = c5fakes.wav_info(got)
    assert info["magic"] == b"RIFF" and info["wave"] == b"WAVE"
    assert (info["channels"], info["bits"], info["rate"]) == (1, 16, 48000)
    assert info["frames"] == len(c5fakes.FakeEngine.expected_samples("Hello world."))

    for name in ("reply.ogg", "REPLY2.OGG"):
        ogg = tmp_path / name
        got = provider.synthesize("Hi there.", str(ogg), format="mp3")
        assert got == str(ogg)
        assert c5fakes.wav_info(ogg)["magic"] == b"RIFF"

    got = provider.synthesize("No suffix.", str(tmp_path / "plain"), format="wav")
    assert got == str(tmp_path / "plain.wav")
    got = provider.synthesize("Already wav.", str(tmp_path / "x.wav"), format="opus")
    assert got == str(tmp_path / "x.wav")
    calls = plugin.engine.calls_of("speak")
    assert [c["text"] for c in calls] == ["Hello world.", "Hi there.", "Hi there.", "No suffix.", "Already wav."]


# §10 error mapping: TextError and any engine failure -> RuntimeError, never ValueError (Hermes would report a
# configuration error, tools/tts_tool.py:456-461)
@pytest.mark.parametrize("exc_name", ["TextError", "ScoreError", "BackendError", "ValueError", "OSError", "KeyError"])
def test_synthesize_errors_are_runtime_errors(provider, plugin, tmp_path, exc_name):
    exc = getattr(plugin.vocal, exc_name, None) or {"ValueError": ValueError, "OSError": OSError,
                                                     "KeyError": KeyError}[exc_name]
    plugin.engine.fail["speak"] = exc("boom")
    with pytest.raises(RuntimeError) as ei:
        provider.synthesize("Hello.", str(tmp_path / "a.mp3"))
    assert not isinstance(ei.value, ValueError)
    assert not isinstance(ei.value, FileNotFoundError)
    assert not (tmp_path / "a.mp3").exists()


# K-6.3: voice_compatible from HERMES_SESSION_PLATFORM (rot: statically True)
@pytest.mark.parametrize("platform,expected", [
    ("telegram", True), ("Telegram", True), ("discord", True), ("matrix", True), ("feishu", True),
    ("whatsapp", True), ("signal", True), ("", False), ("cli", False), ("desktop", False), ("slack", False),
    (None, False)])
def test_voice_compatible(provider, plugin, platform, expected):
    if platform is not None:
        plugin.hermes.session["HERMES_SESSION_PLATFORM"] = platform
    assert provider.voice_compatible is expected


def _collect(provider, text, **kw):
    return b"".join(provider.stream(text, format="pcm", **kw))


# K-6.4: int16 LE @ 48 000; 3 parallel calls = serial results; closing the generator frees the slot and cancels;
# format "opus" -> NotImplementedError (rot: a buffer shared between calls)
def test_stream_pcm(provider, plugin):
    eng = plugin.engine
    assert provider.streams_pcm is True
    assert provider.stream_sample_rate == 48000
    texts = ["First sentence here.", "A second, longer sentence follows now.", "Third!"]
    serial = [_collect(provider, t) for t in texts]
    for t, pcm in zip(texts, serial):
        assert pcm == c5fakes.FakeEngine.expected_pcm(t)
        assert len(pcm) % 2 == 0

    eng.chunk_delay = 0.002
    eng.chunk_samples = 480
    results: list = [None] * 3
    barrier = threading.Barrier(3)
    errors: list = []

    def run(i):
        try:
            barrier.wait(5)
            results[i] = _collect(provider, texts[i])
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert eng.max_active_streams >= 2, "streams did not overlap; the test would prove nothing"
    assert results == serial

    gen = provider.stream("Close me early, please, this is long enough.", format="pcm")
    first = next(iter(gen))
    assert first and eng.active_streams == 1
    gen.close()
    assert eng.active_streams == 0, "closing the generator must close the engine stream (slot freed)"
    ev = eng.calls_of("speak_pcm16")[-1]["cancel"]
    assert isinstance(ev, threading.Event) and ev.is_set(), "closing the generator must set the cancel event"

    for fmt in ("opus", "mp3", "wav"):
        with pytest.raises(NotImplementedError):
            next(iter(provider.stream("x", format=fmt)))


def test_stream_passes_resolved_voice_and_speed(provider, plugin, registered):
    registered.settings["voice"] = "narrator"
    _collect(provider, "Hello.", voice="female", speed=1.25)
    call = plugin.engine.calls_of("speak_pcm16")[-1]
    assert c5fakes.FakeEngine.voice_id(call["voice"]) == "narrator"
    assert call["speed"] == pytest.approx(1.25 * 0.92)


# K-6.13: plugins.isolation host -> streams_pcm False, synthesize works (rot: class attribute True)
def test_plugin_host_mode(provider, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    assert provider.streams_pcm is False
    got = provider.synthesize("Host mode works.", str(tmp_path / "h.mp3"))
    assert c5fakes.wav_info(got)["magic"] == b"RIFF"
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "0")
    assert provider.streams_pcm is True
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS")
    assert provider.streams_pcm is True
    owner = next(c for c in type(provider).__mro__ if "streams_pcm" in c.__dict__)
    assert isinstance(owner.__dict__["streams_pcm"], property), "streams_pcm must be a property (read per reply)"


# §10: voices, default voice, warm/release, availability
def test_provider_catalog_and_lease(provider, plugin, registered):
    ids = [v["id"] for v in provider.list_voices()]
    assert ids == list(c5fakes.PRESET_TABLE)
    assert provider.default_voice() == "male"
    plugin.hermes.config["tts"] = {"voice": "child"}
    assert provider.default_voice() == "child"
    registered.settings["voice"] = "elder"
    assert provider.default_voice() == "elder"
    assert provider.is_available() is True
    provider.warm()
    assert plugin.engine.hold_warm_calls == [True] and plugin.engine.calls_of("warmup")
    provider.release()
    assert plugin.engine.hold_warm_calls == [True, False]
    plugin.engine.fail["warmup"] = RuntimeError("worker died")
    provider.warm()  # best effort: never raises
