"""Fakes for the C5 (Hermes binding) tests: FakeCtx, FakeEngine, fake engine namespace, plugin loader.

No module-level numpy import: the K-6.11 subprocess imports this module with numpy blocked.

* ``load_plugin()`` imports the plugin root the way Hermes does (``hermes_plugins.<slug>``, a synthetic package with
  ``submodule_search_locations``; hermes_cli/plugins_loader.py:817-848), so the plugin's relative imports work.
* ``FakeCtx`` records every registration (signatures of hermes_cli/plugins_context.py:270-307, 442-480, 866;
  plugins_content.py:31-67) and keeps plugin settings in a dict (``get_config``/``set_config``).
* ``make_fake_vocal()`` builds the namespace that replaces ``chimera_hermes._vocal.api()`` (the C5-internal seam to the
  engine package, INTERFACES §5/§7/§9.4): EngineConfig, get_engine, presets, runtime, the error classes.
* ``FakeEngine`` implements the Engine API of INTERFACES §5 with deterministic sine audio, simulated durations,
  configurable failures and stream bookkeeping (active streams, cancel events).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
import wave
import zlib
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

HERE = Path(__file__).resolve().parent
FAKEHERMES_DIR = HERE / "fakehermes"
REPO_ROOT = HERE.parent.parent.parent
FS = 48000

# --------------------------------------------------------------------------------------------- plugin loading

_counter = itertools.count()


def load_plugin(name: str | None = None, root: Path = REPO_ROOT) -> types.ModuleType:
    """Import the plugin directory as ``hermes_plugins.<name>`` exactly like Hermes' directory loader."""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []  # type: ignore[attr-defined]
        ns.__package__ = "hermes_plugins"
        sys.modules["hermes_plugins"] = ns
    name = name or f"chimera_t{next(_counter)}"
    full = f"hermes_plugins.{name}"
    unload_plugin(full)
    spec = importlib.util.spec_from_file_location(full, Path(root) / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full
    mod.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules[full] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        unload_plugin(full)
        raise
    return mod


def unload_plugin(plugin: "types.ModuleType | str") -> None:
    full = plugin if isinstance(plugin, str) else plugin.__name__
    for k in [k for k in sys.modules if k == full or k.startswith(full + ".")]:
        del sys.modules[k]


def submodule(plugin: types.ModuleType, rel: str) -> types.ModuleType:
    return importlib.import_module(f"{plugin.__name__}.{rel}")


# --------------------------------------------------------------------------------------------- FakeCtx

class FakeCtx:
    """Records registrations like Hermes' PluginContext; plugin settings in ``settings``."""

    def __init__(self, settings: Mapping[str, Any] | None = None, plugin_id: str = "chimera") -> None:
        self.plugin_id = plugin_id
        self.settings: dict[str, Any] = dict(settings or {})
        self.config_writes: list[tuple[str, Any]] = []
        self.tts_providers: list[Any] = []
        self.tools: dict[str, dict] = {}
        self.commands: dict[str, dict] = {}
        self.cli_commands: dict[str, dict] = {}
        self.skills: dict[str, dict] = {}
        self.order: list[tuple[str, str]] = []

    def register_tts_provider(self, provider) -> None:
        self.tts_providers.append(provider)
        self.order.append(("tts_provider", getattr(provider, "name", "?")))

    def register_tool(self, name: str, toolset: str, schema: dict, handler: Callable, check_fn: Callable | None = None,
                      requires_env: list | None = None, is_async: bool = False, description: str = "", emoji: str = "",
                      override: bool = False):
        self.tools[name] = {"name": name, "toolset": toolset, "schema": schema, "handler": handler,
                            "check_fn": check_fn, "requires_env": requires_env, "is_async": is_async,
                            "override": override}
        self.order.append(("tool", name))

    def register_command(self, name: str, handler: Callable, description: str = "", args_hint: str = "",
                         argument_mode: str | None = None):
        self.commands[name] = {"handler": handler, "description": description, "args_hint": args_hint,
                               "argument_mode": argument_mode}
        self.order.append(("command", name))

    def register_cli_command(self, name: str, help: str, setup_fn: Callable, handler_fn: Callable | None = None,
                             description: str = ""):
        self.cli_commands[name] = {"help": help, "setup_fn": setup_fn, "handler_fn": handler_fn,
                                   "description": description}
        self.order.append(("cli_command", name))

    def register_skill(self, name: str, path, description: str = "", frontmatter: Mapping[str, Any] | None = None):
        if ":" in name:
            raise ValueError("skill name must not contain ':'")
        if not Path(path).exists():  # Hermes raises here too (plugins_content.py:52-53)
            raise FileNotFoundError(f"SKILL.md not found at {path}")
        self.skills[name] = {"path": path, "description": description, "frontmatter": dict(frontmatter or {})}
        self.order.append(("skill", name))

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        self.settings[key] = value
        self.config_writes.append((key, value))


# --------------------------------------------------------------------------------------------- engine data types
# Field names follow INTERFACES §5/§7/§8.2; the plugin only reads attributes, so dataclasses suffice.

class FakeChimeraError(Exception):
    pass


class FakeScoreError(FakeChimeraError, ValueError):
    def __init__(self, message: str = "invalid score", errors: list | None = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


class FakeTextError(FakeChimeraError, ValueError):
    pass


class FakeRenderCancelled(FakeChimeraError):
    pass


class FakeBackendError(FakeChimeraError, RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class FakeEngineConfig:  # = EngineConfig, INTERFACES §5
    backend: str = "auto"
    workers: int = 0
    in_process_fallback: bool = True
    cache_dir: str | None = None
    sample_rate: int = 48000
    text_prep: bool = True
    max_voices: int = 8
    max_score_seconds: float = 300.0
    max_voice_seconds: float = 1200.0
    worker_idle_s: float = 60.0


@dataclasses.dataclass(frozen=True)
class FakeSpeechStyle:
    rate: float = 1.0
    f0: float | None = None
    accent: float | None = None


@dataclasses.dataclass(frozen=True)
class FakePreset:  # = VoicePreset, INTERFACES §7 (body/source replaced by the spec string)
    id: str
    display: str
    gender: str
    speech: FakeSpeechStyle
    description: str = ""
    spec: str = ""


# id -> (display, gender, f0, rate) from the INTERFACES §7 table
PRESET_TABLE: dict[str, tuple[str, str, float, float]] = {
    "male": ("Male", "male", 112.0, 1.0),
    "female": ("Female", "female", 205.0, 1.0),
    "child": ("Child", "neutral", 250.0, 1.0),
    "androgynous": ("Androgynous", "neutral", 150.17, 1.0),
    "teen": ("Teen", "neutral", 197.59, 1.05),
    "male-deep": ("Male (deep)", "male", 92.0, 0.95),
    "female-bright": ("Female (bright)", "female", 225.0, 1.0),
    "narrator": ("Narrator", "male", 104.0, 0.92),
    "whisper": ("Whisper", "female", 205.0, 0.95),
    "male-whisper": ("Male whisper", "male", 112.0, 0.95),
    "robot": ("Robot", "male", 110.0, 1.0),
    "elder": ("Elder", "male", 118.0, 0.9),
}
PRESETS: dict[str, FakePreset] = {
    k: FakePreset(k, d, g, FakeSpeechStyle(rate=r, f0=f0), description=f"{d} voice", spec=k)
    for k, (d, g, f0, r) in PRESET_TABLE.items()
}
_OPTION_RANGES = {"pitch": (50.0, 600.0), "semitones": (-24.0, 24.0), "rate": (0.5, 1.8), "accent": (0.0, 0.4),
                  "breathiness": (0.0, 1.0), "tension": (0.0, 1.0), "brightness": (0.0, 1.0),
                  "roughness": (0.0, 1.0)}


def parse_voice_spec(spec: str | None) -> FakePreset | None:
    """Subset of the INTERFACES §7 grammar; unknown preset/option or bad value -> None (never raises)."""
    if not isinstance(spec, str) or not spec.strip():
        return None
    parts = [p.strip() for p in spec.strip().split(";")]
    base = PRESETS.get(parts[0].lower())
    if base is None:
        return None
    rate = base.speech.rate
    f0 = base.speech.f0
    for opt in parts[1:]:
        key, sep, val = opt.partition("=")
        key = key.strip().lower()
        if not sep:
            return None
        if key in ("whisper",):
            if val.strip().lower() not in ("true", "false"):
                return None
            continue
        if key == "seed":
            if not val.strip().isdigit() or int(val) >= 2 ** 32:
                return None
            continue
        if key == "body":
            try:
                f, c = (float(x) for x in val.split(","))
            except ValueError:
                return None
            if not (0 <= f <= 1 and 0 <= c <= 1):
                return None
            continue
        if key not in _OPTION_RANGES:
            return None
        try:
            v = float(val)
        except ValueError:
            return None
        lo, hi = _OPTION_RANGES[key]
        if not (lo <= v <= hi):
            return None
        if key == "rate":
            rate = v
        if key == "pitch":
            f0 = v
    return dataclasses.replace(base, speech=FakeSpeechStyle(rate=rate, f0=f0), spec=spec.strip())


def list_voices() -> list[dict]:
    return [{"id": p.id, "display": p.display, "language": "en", "gender": p.gender} for p in PRESETS.values()]


@dataclasses.dataclass(frozen=True)
class FakeScoreIssue:
    path: str
    severity: str
    message: str


@dataclasses.dataclass(frozen=True)
class FakeVoiceSummary:
    name: str
    body: str
    items: int
    sounding: int
    spoken: int
    first_beat: float
    end_beat: float
    seconds: float
    lowest: str
    highest: str
    syllables: list


@dataclasses.dataclass(frozen=True)
class FakeScoreReport:
    ok: bool
    issues: list
    voices: list
    seconds: float
    voice_seconds: float
    lead_in: float
    memory_mb: float
    disk_mb: float
    estimate_s: dict
    normalized: dict


@dataclasses.dataclass(frozen=True)
class FakeMixInfo:
    peak_before: float
    gain_db: float
    channels: int
    clipped_samples: int


@dataclasses.dataclass(frozen=True)
class FakeScoreResult:
    path_wav: str
    path_mp3: str | None
    seconds: float
    sample_rate: int
    channels: int
    voices: int
    kernel: str
    render_s: float
    warnings: list
    mix: FakeMixInfo


@dataclasses.dataclass(frozen=True)
class FakeBackendStatus:
    numpy_ok: bool
    numba_installed: bool
    numba_version: str | None
    numba_state: str
    numba_error: str | None
    cache_dir: str | None
    marker: dict | None
    rtf_estimate: dict
    workers: int
    workers_alive: int
    restart_needed: bool


@dataclasses.dataclass(frozen=True)
class FakeSelfTestResult:
    backend: str
    passed: bool
    max_rel_err: float
    seconds: float
    detail: str


# --------------------------------------------------------------------------------------------- audio helpers

def fake_samples(text: str, seconds: float | None = None):
    """Deterministic float32 sine for ``text`` (numpy imported here, not at module level)."""
    import numpy as np

    if seconds is None:
        seconds = min(2.0, 0.1 + 0.02 * len(text))
    n = max(1, int(round(FS * seconds)))
    f = 110.0 + (zlib.crc32(text.encode("utf-8")) % 300)
    t = np.arange(n, dtype=np.float64) / FS
    return (0.3 * np.sin(2.0 * math.pi * f * t)).astype(np.float32)


def pcm16(samples) -> bytes:
    import numpy as np

    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    return np.round(x * 32767.0).astype("<i2").tobytes()


def write_wav16(path: str | os.PathLike, samples, rate: int = FS, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16(samples))


def wav_info(path: str | os.PathLike) -> dict:
    """RIFF/WAVE header facts of a file (magic, channels, bits, rate, frames)."""
    with open(path, "rb") as f:
        head = f.read(12)
    info = {"magic": head[:4], "wave": head[8:12]}
    if head[:4] == b"RIFF":
        with wave.open(str(path), "rb") as w:
            info.update(channels=w.getnchannels(), bits=8 * w.getsampwidth(), rate=w.getframerate(),
                        frames=w.getnframes())
    return info


@dataclasses.dataclass(frozen=True)
class FakeAudio:  # = Audio, INTERFACES §5
    samples: Any
    sample_rate: int = FS

    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def to_wav(self, path, *, bits: int = 16) -> str:
        assert bits == 16
        write_wav16(path, self.samples, self.sample_rate)
        return str(path)

    def pcm16_bytes(self) -> bytes:
        return pcm16(self.samples)


# --------------------------------------------------------------------------------------------- FakeEngine

class FakeEngine:
    """Engine stand-in (INTERFACES §5). Thread-safe bookkeeping; every public call is recorded in ``calls``.

    Knobs: ``speak_delay``, ``render_delay`` (s, slept in steps with progress), ``chunk_delay`` (s between PCM
    chunks), ``chunk_samples``, ``fail`` {method: exception instance or class}, ``numba_state``,
    ``numba_installed``, ``numpy_ok``, ``restart_needed``, ``warm_crash`` (warm-up runs a child process that dies
    with SIGSEGV and the state becomes ``failed``), ``score_seconds``, ``estimate_s``, ``render_warnings``.
    """

    def __init__(self, **knobs: Any) -> None:
        self.speak_delay = 0.0
        self.render_delay = 0.0
        self.chunk_delay = 0.0
        self.chunk_samples = 2400
        self.fail: dict[str, Any] = {}
        self.numpy_ok = True
        self.numba_installed = False
        self.numba_version: str | None = None
        self.numba_state = "absent"
        self.numba_error: str | None = None
        self.restart_needed = False
        self.warm_crash = False
        self.score_seconds = 3.0
        self.estimate_s = {"numba": 1.5, "numpy": 15.0}
        self.render_warnings: list[str] = []
        self.cache_dir: str | None = None
        for k, v in knobs.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict]] = []
        self.cancel_events: list[threading.Event] = []
        self.active_streams = 0
        self.max_active_streams = 0
        self.closed_streams = 0
        self.hold_warm_calls: list[bool] = []

    # bookkeeping
    def _record(self, name: str, **kw: Any) -> None:
        with self._lock:
            self.calls.append((name, kw))

    def calls_of(self, name: str) -> list[dict]:
        with self._lock:
            return [kw for n, kw in self.calls if n == name]

    def _maybe_fail(self, name: str) -> None:
        exc = self.fail.get(name)
        if exc is not None:
            raise exc() if isinstance(exc, type) else exc

    @property
    def kernel(self) -> str:
        return "numba" if self.numba_state == "ready" else "numpy"

    # helpers for assertions
    @staticmethod
    def voice_id(voice: Any) -> str:
        return voice if isinstance(voice, str) else getattr(voice, "id", repr(voice))

    @staticmethod
    def expected_samples(text: str):
        return fake_samples(text)

    @classmethod
    def expected_pcm(cls, text: str) -> bytes:
        return pcm16(cls.expected_samples(text))

    # Engine API
    def speak(self, text: str, voice: Any = "male", *, speed: float | None = None, settle_first: bool = True,
              cancel: threading.Event | None = None):
        self._record("speak", text=text, voice=voice, speed=speed, settle_first=settle_first, cancel=cancel)
        self._maybe_fail("speak")
        if self.speak_delay:
            end = time.monotonic() + self.speak_delay
            while time.monotonic() < end:
                if cancel is not None and cancel.is_set():
                    raise FakeRenderCancelled("cancelled")
                time.sleep(0.01)
        return FakeAudio(self.expected_samples(text)), self.kernel

    def speak_pcm16(self, text: str, voice: Any = "male", *, speed: float | None = None,
                    cancel: threading.Event | None = None) -> Iterator[bytes]:
        self._record("speak_pcm16", text=text, voice=voice, speed=speed, cancel=cancel)
        if cancel is not None:
            with self._lock:
                self.cancel_events.append(cancel)
        self._maybe_fail("speak_pcm16")
        return self._pcm_gen(self.expected_pcm(text), cancel)

    def _pcm_gen(self, data: bytes, cancel: threading.Event | None) -> Iterator[bytes]:
        with self._lock:
            self.active_streams += 1
            self.max_active_streams = max(self.max_active_streams, self.active_streams)
        try:
            step = 2 * self.chunk_samples
            for i in range(0, len(data), step):
                if cancel is not None and cancel.is_set():
                    return
                if self.chunk_delay:
                    time.sleep(self.chunk_delay)
                yield data[i:i + step]
        finally:
            with self._lock:
                self.active_streams -= 1
                self.closed_streams += 1

    def check_score(self, score: "dict | str") -> FakeScoreReport:
        self._record("check_score", score=score)
        self._maybe_fail("check_score")
        if isinstance(score, str):
            score = json.loads(score)
        issues: list[FakeScoreIssue] = []
        if not isinstance(score, dict):
            issues.append(FakeScoreIssue("score", "error", "score must be an object"))
            score = {}
        bpm = score.get("bpm")
        if not isinstance(bpm, (int, float)) or isinstance(bpm, bool) or not 30 <= bpm <= 240:
            issues.append(FakeScoreIssue("bpm", "error", "bpm must be a number 30..240"))
        voices = score.get("voices")
        if not isinstance(voices, list) or not voices:
            issues.append(FakeScoreIssue("voices", "error", "give 1..8 voices"))
            voices = []
        summaries = [FakeVoiceSummary(str(v.get("name", f"voice{i}")) if isinstance(v, dict) else f"voice{i}",
                                      "male", 1, 1, 0, 0.0, 1.0, self.score_seconds, "C4", "C4", ["ah"])
                     for i, v in enumerate(voices)]
        return FakeScoreReport(ok=not issues, issues=issues, voices=summaries, seconds=self.score_seconds,
                               voice_seconds=self.score_seconds * max(1, len(voices)), lead_in=0.3,
                               memory_mb=12.0, disk_mb=3.0, estimate_s=dict(self.estimate_s),
                               normalized=dict(score))

    def render_score(self, score: "dict | str", out_dir: str, stem: str, *, want_mp3: bool,
                     progress: Callable[[float], None] | None = None,
                     cancel: threading.Event | None = None) -> FakeScoreResult:
        self._record("render_score", score=score, out_dir=out_dir, stem=stem, want_mp3=want_mp3,
                     cancel=cancel)
        self._maybe_fail("render_score")
        t0 = time.monotonic()
        steps = max(1, int(self.render_delay / 0.02)) if self.render_delay else 1
        for i in range(steps):
            if cancel is not None and cancel.is_set():
                raise FakeRenderCancelled("cancelled")
            if self.render_delay:
                time.sleep(self.render_delay / steps)
            if progress is not None:
                progress((i + 1) / steps)
        if isinstance(score, str):
            score = json.loads(score)
        path = os.path.join(out_dir, stem + ".wav")
        write_wav16(path, fake_samples(stem, seconds=min(self.score_seconds, 1.0)))
        voices = len(score.get("voices") or []) if isinstance(score, dict) else 1
        return FakeScoreResult(path_wav=path, path_mp3=None, seconds=self.score_seconds, sample_rate=FS, channels=1,
                               voices=voices, kernel=self.kernel, render_s=time.monotonic() - t0,
                               warnings=list(self.render_warnings),
                               mix=FakeMixInfo(peak_before=0.3, gain_db=8.5, channels=1, clipped_samples=0))

    def status(self) -> FakeBackendStatus:
        self._record("status")
        self._maybe_fail("status")
        return FakeBackendStatus(numpy_ok=self.numpy_ok, numba_installed=self.numba_installed,
                                 numba_version=self.numba_version, numba_state=self.numba_state,
                                 numba_error=self.numba_error, cache_dir=self.cache_dir,
                                 marker={"content_hash": "0123456789abcdef"} if self.numba_state == "ready" else None,
                                 rtf_estimate={"numba": 0.45, "numpy": 5.0}, workers=3, workers_alive=1,
                                 restart_needed=self.restart_needed)

    def warmup(self) -> None:
        self._record("warmup")
        if self.warm_crash:
            # test hook: the warm-up "worker" is a real child process that dies with SIGSEGV
            r = subprocess.run([sys.executable, "-I", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"],
                               capture_output=True)
            self.numba_state = "failed"
            self.numba_error = f"warm-up worker died (returncode {r.returncode})"
        self._maybe_fail("warmup")

    def hold_warm(self, on: bool) -> None:
        self._record("hold_warm", on=on)
        self.hold_warm_calls.append(bool(on))
        self._maybe_fail("hold_warm")

    def selftest(self) -> FakeSelfTestResult:
        self._record("selftest")
        self._maybe_fail("selftest")
        return FakeSelfTestResult(backend=self.kernel, passed=True, max_rel_err=0.0, seconds=0.1, detail="fake")

    def shutdown(self) -> None:
        self._record("shutdown")


# --------------------------------------------------------------------------------------------- fake engine package

@dataclasses.dataclass
class FakeRuntime:  # = Runtime, INTERFACES §9.4 (jobs filled lazily by chimera_hermes, request C5-1)
    engine: Any
    scheduler: Any = None
    jobs: Any = None
    lock: Any = dataclasses.field(default_factory=threading.Lock)


def make_fake_vocal(engine: FakeEngine | None = None) -> types.SimpleNamespace:
    """Namespace standing in for the ``chimera_vocal`` package as seen through ``chimera_hermes._vocal.api()``."""
    engine = engine or FakeEngine()
    rt = FakeRuntime(engine=engine)
    configs: list[FakeEngineConfig] = []

    def get_engine(config: FakeEngineConfig | None = None) -> FakeEngine:
        configs.append(config if config is not None else FakeEngineConfig())
        return engine

    presets = types.SimpleNamespace(PRESETS=PRESETS, DEFAULT_VOICE="male", parse_voice_spec=parse_voice_spec,
                                    list_voices=list_voices, VoicePreset=FakePreset)
    return types.SimpleNamespace(
        __version__="0.1.0", EngineConfig=FakeEngineConfig, get_engine=get_engine, reconfigure=get_engine,
        presets=presets, runtime=types.SimpleNamespace(runtime=lambda: rt, Runtime=FakeRuntime),
        ChimeraError=FakeChimeraError, ScoreError=FakeScoreError, TextError=FakeTextError,
        RenderCancelled=FakeRenderCancelled, BackendError=FakeBackendError,
        engine=engine, configs=configs, rt=rt,
    )
