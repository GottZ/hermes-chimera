"""chimera_vocal: Python port of the chimera-vocal-synth engine (public API).

Importing this package never imports numpy, numba or anything from Hermes: the plugin must load (and report what is
missing) in an environment without numpy, and the host process never imports numba. numpy is
imported inside functions only, numba only inside render worker processes (``worker.py``).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib.util
import itertools
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from .music.mixer import MixInfo
    from .music.schema import ScoreIssue, ScoreReport
    from .presets import VoicePreset

__version__ = "0.1.0"

__all__ = [
    "__version__", "EngineConfig", "Audio", "BackendStatus", "ScoreResult", "SpeechResult", "Engine", "get_engine",
    "reconfigure",
    "ChimeraError", "ScoreError", "TextError", "RenderCancelled", "BackendError",
]


# --- errors -------------------------------------------------------------------------------------------------

class ChimeraError(Exception):
    """Base class of every error the engine raises on purpose."""


class ScoreError(ChimeraError, ValueError):
    """Invalid score; ``errors`` lists the error-severity issues (path + message with the fix)."""

    def __init__(self, errors: "list[ScoreIssue]", message: str | None = None) -> None:
        self.errors = list(errors)
        super().__init__(message or "; ".join(f"{e.path}: {e.message}" for e in self.errors) or "invalid score")


class TextError(ChimeraError, ValueError):
    """Nothing speakable after text preparation (the TTS provider maps it to RuntimeError)."""


class RenderCancelled(ChimeraError):
    """The request's cancel event was set; queued jobs were dropped and running workers stopped."""


class BackendError(ChimeraError, RuntimeError):
    """numpy missing, or no worker could render and the in-process fallback is off / failed."""


# --- configuration and results ------------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    backend: Literal["auto", "numba", "numpy"] = "auto"
    workers: int = 0                 # 0 -> min(4, max(2, os.cpu_count() - 1)); one slot reserved for speech
    in_process_fallback: bool = True
    cache_dir: str | None = None     # numba cache base; None -> numba-style user cache dir + "/chimera"
    sample_rate: int = 48000         # fixed: 48 kHz out / 96 kHz internal; other values -> ValueError
    text_prep: bool = True
    max_voices: int = 8
    max_score_seconds: float = 300.0
    max_voice_seconds: float = 1200.0
    worker_idle_s: float = 60.0

    def __post_init__(self) -> None:
        if self.sample_rate != 48000:
            raise ValueError(f"sample_rate must be 48000 (48 kHz out, 96 kHz internal), not {self.sample_rate}")
        if self.backend not in ("auto", "numba", "numpy"):
            raise ValueError(f"backend must be auto, numba or numpy, not {self.backend!r}")
        if not isinstance(self.workers, int) or self.workers < 0:
            raise ValueError("workers must be an integer >= 0 (0 = automatic)")


@dataclass(frozen=True)
class Audio:
    samples: "np.ndarray"            # float32, (n,) mono or (n, 2) stereo
    sample_rate: int

    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def to_wav(self, path, *, bits: int = 16) -> str:
        from .core.wavio import write_wav
        write_wav(path, self.samples, self.sample_rate, bits=bits)
        return str(path)

    def pcm16_bytes(self) -> bytes:
        from .core.wavio import wav_bytes
        return wav_bytes(self.samples, self.sample_rate)[44:]


@dataclass(frozen=True)
class BackendStatus:
    numpy_ok: bool
    numba_installed: bool                   # find_spec only - the host never imports numba
    numba_version: str | None
    numba_state: Literal["absent", "cold", "warming", "ready", "failed"]
    numba_error: str | None
    cache_dir: str | None
    marker: dict | None
    rtf_estimate: dict[str, float]          # {"numba": …, "numpy": …} EMA of render_s/audio_s per job
    workers: int
    workers_alive: int
    restart_needed: bool                    # numpy/numba became importable after this process started


@dataclass(frozen=True)
class ScoreResult:
    path_wav: str
    path_mp3: str | None
    seconds: float
    sample_rate: int
    channels: int
    voices: int
    kernel: Literal["numba", "numpy"]       # ONE kernel per request (deterministic output)
    render_s: float
    warnings: list[str]
    mix: "MixInfo"


@dataclass(frozen=True)
class SpeechResult:
    path_wav: str
    seconds: float
    sample_rate: int
    kernel: Literal["numba", "numpy"]
    render_s: float


# --- engine -------------------------------------------------------------------------------------------------

def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _default_cache_base() -> str:
    """numba-style user cache directory + "/chimera" (no numba import)."""
    if sys.platform == "win32":  # pragma: no cover
        base = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "numba", "Cache")
    elif sys.platform == "darwin":  # pragma: no cover
        base = os.path.join(os.path.expanduser("~"), "Library", "Caches", "numba")
    else:
        base = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
                            "numba")
    return os.path.join(base, "chimera")


class _LazyG2P:
    """The default G2P, loaded on first use (structural score errors never need the dictionary)."""

    def __init__(self) -> None:
        self._g = None

    def __getattr__(self, name: str) -> Any:
        if self._g is None:
            from .music.planning import planner_api
            self._g = planner_api().default_g2p()
        return getattr(self._g, name)


class Engine:
    """Thread-safe; one per process via the runtime registry. Construction is light (no numpy, no process)."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        from .scheduler import RenderScheduler, default_workers
        self.config = config or EngineConfig()
        self.workers = self.config.workers or default_workers(os.cpu_count())
        self.cache_base = self.config.cache_dir or _default_cache_base()
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend = os.environ.get("CHIMERA_BACKEND") or self.config.backend
        if backend not in ("auto", "numba", "numpy"):
            backend = self.config.backend
        self.scheduler = RenderScheduler(workers=self.workers, cache_base=self.cache_base, root=self.root,
                                         idle_s=self.config.worker_idle_s,
                                         in_process_fallback=self.config.in_process_fallback, backend=backend)
        self._seq = itertools.count(1)
        self._cfg_lock = threading.Lock()
        self._numpy_at_start = _has("numpy")
        self._numba_at_start = _has("numba")

    def apply_settings(self, config: EngineConfig) -> None:
        """Take over the user-facing settings of ``config`` (backend, workers, text_prep, max_voices) for the next
        requests; running requests keep their kernel. cache_dir and the other fields stay as built."""
        from .scheduler import default_workers
        with self._cfg_lock:
            cur = self.config
            new = dataclasses.replace(cur, backend=config.backend, workers=config.workers,
                                      text_prep=config.text_prep, max_voices=config.max_voices)
            if new == cur:
                return
            self.config = new
            backend = os.environ.get("CHIMERA_BACKEND") or new.backend
            self.scheduler.set_backend(backend if backend in ("auto", "numba", "numpy") else new.backend)
            if new.workers != cur.workers:
                self.workers = new.workers or default_workers(os.cpu_count())
                self.scheduler.set_workers(self.workers)

    # helpers -----------------------------------------------------------------------------------------------
    def _need_numpy(self) -> None:
        if not _has("numpy"):
            raise BackendError("numpy is not available in this process - if chimera was just installed, restart; "
                               "otherwise reinstall it with its dependencies")

    def _preset(self, voice: "str | VoicePreset") -> "VoicePreset":
        from .presets import VoicePreset, parse_voice_spec
        if isinstance(voice, VoicePreset):
            return voice
        p = parse_voice_spec(voice)
        if p is None:
            raise ChimeraError(f"unknown voice {voice!r}")
        return p

    def _req_id(self, kind: str) -> str:
        return f"{kind}{next(self._seq)}-{uuid.uuid4().hex[:8]}"

    def choose_kernel(self, backend: "Literal['auto', 'numba', 'numpy'] | None" = None) -> Literal["numba", "numpy"]:
        """Kernel for one request. ``backend`` is the setting of whoever makes the request (None = the engine's
        current setting). One engine serves every Hermes profile of a process and its own setting is the one applied
        last, so a request that names its backend is not steered by another profile: ``numpy`` stays numpy, and
        ``auto``/``numba`` render with numba once it is warm even while the engine's setting is ``numpy`` (until then
        with numpy, while such a request starts the warm-up)."""
        return self.scheduler.choose_kernel(backend)

    def _submit_all(self, req, jobs, sinks, ev: threading.Event) -> list:
        out = []
        for j, s in zip(jobs, sinks):
            out.append(self.scheduler.submit(req, dataclasses.replace(j, id=f"{req.id}.{j.id}"), sink=s, cancel=ev))
        return out

    @staticmethod
    def _wait(futs, ev: threading.Event, cancel: threading.Event | None, tick: Callable[[], None] | None = None):
        pending = set(futs)
        try:
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=0.05,
                                                        return_when=concurrent.futures.FIRST_EXCEPTION)
                for f in done:
                    if f.exception() is not None:
                        raise f.exception()
                if tick is not None:
                    tick()
                if cancel is not None and cancel.is_set():
                    raise RenderCancelled("cancelled")
        except BaseException:
            ev.set()
            raise

    # speech ------------------------------------------------------------------------------------------------
    def _plan_speech(self, text: str, voice, speed, settle_first: bool, faithful: bool = False,
                     text_prep: bool | None = None):
        from .music.planning import planner_api
        from .speech import plan_speech
        preset = self._preset(voice)
        if faithful:
            # exactly `chimera render text` of the template: no text preparation, one window, the pitch glides in
            # from 110 Hz, the preset's seeds (the template's own for the stock voices)
            return plan_speech(text, preset, speed=speed, g2p=planner_api().default_g2p(), text_prep=False,
                               split=False, settle_first=False)
        return plan_speech(text, preset, speed=speed, g2p=planner_api().default_g2p(),
                           text_prep=self.config.text_prep if text_prep is None else bool(text_prep),
                           settle_first=settle_first)

    def speak(self, text: str, voice: "str | VoicePreset" = "male", *, speed: float | None = None,
              settle_first: bool = True, faithful: bool = False, text_prep: bool | None = None,
              backend: "Literal['auto', 'numba', 'numpy'] | None" = None,
              cancel: threading.Event | None = None) -> "tuple[Audio, str]":
        """Speak ``text`` into memory. ``faithful=True`` renders like the template's ``chimera render text`` (sample
        for sample within rounding): no text preparation, no phrase windows, no settled first pitch. ``text_prep``
        and ``backend`` fix those settings for this request (None = the engine's current settings). For long texts
        prefer ``speak_to_wav`` (memory stays small)."""
        self._need_numpy()
        from .scheduler import MemorySink, Request
        from .speech import assemble
        jobs, n_total = self._plan_speech(text, voice, speed, settle_first, faithful, text_prep)
        if cancel is not None and cancel.is_set():
            raise RenderCancelled("cancelled before start")
        req = Request(self._req_id("speak"), "speech", self.choose_kernel(backend))
        ev = threading.Event()
        sinks = [MemorySink(j.n_out) for j in jobs]
        futs = self._submit_all(req, jobs, sinks, ev)
        self._wait(futs, ev, cancel)
        return Audio(assemble([s.data for s in sinks], jobs, n_total), 48000), req.kernel

    def speak_to_wav(self, text: str, path: str, voice: "str | VoicePreset" = "male", *,
                     speed: float | None = None, settle_first: bool = True, faithful: bool = False,
                     text_prep: bool | None = None, backend: "Literal['auto', 'numba', 'numpy'] | None" = None,
                     tmp_dir: str | None = None, cancel: threading.Event | None = None) -> "SpeechResult":
        """Speak ``text`` into a 16-bit WAV at ``path`` (the same bytes as ``speak(...)[0].to_wav(path)``). The
        windows go to one temporary file in ``tmp_dir`` (default: the folder of ``path``) and the WAV is written
        block by block, so memory does not grow with the length of the text. ``text_prep``/``backend`` as in
        ``speak``."""
        self._need_numpy()
        from .core.wavio import write_wav_chunked
        from .scheduler import Request
        from .speech import WindowStore
        t_start = time.perf_counter()
        jobs, n_total = self._plan_speech(text, voice, speed, settle_first, faithful, text_prep)
        if cancel is not None and cancel.is_set():
            raise RenderCancelled("cancelled before start")
        req = Request(self._req_id("speak"), "speech", self.choose_kernel(backend))
        out_dir = os.path.dirname(os.path.abspath(os.fspath(path)))
        os.makedirs(tmp_dir or out_dir, exist_ok=True)
        store = WindowStore(jobs, n_total, tmp_dir or out_dir)
        ev = threading.Event()
        try:
            futs = self._submit_all(req, jobs, [store.sink(k) for k in range(len(jobs))], ev)
            self._wait(futs, ev, cancel)
            write_wav_chunked(path, store.chunks(), 48000, 1, n_total)
        except BaseException:
            ev.set()
            raise
        finally:
            store.close()
        return SpeechResult(os.fspath(path), n_total / 48000.0, 48000, req.kernel, time.perf_counter() - t_start)

    def speak_pcm16(self, text: str, voice: "str | VoicePreset" = "male", *, speed: float | None = None,
                    faithful: bool = False, text_prep: bool | None = None,
                    backend: "Literal['auto', 'numba', 'numpy'] | None" = None, tmp_dir: str | None = None,
                    cancel: threading.Event | None = None) -> Iterator[bytes]:
        """Generator of int16 LE mono bytes at 48 kHz, window by window; closing it cancels the remaining windows.
        Windows are kept in a temporary file in ``tmp_dir`` (default: the system temp folder). ``text_prep``/
        ``backend`` as in ``speak``."""
        self._need_numpy()
        from .core.wavio import wav_bytes
        from .scheduler import Request
        from .speech import WindowStore
        jobs, n_total = self._plan_speech(text, voice, speed, True, faithful, text_prep)
        if cancel is not None and cancel.is_set():
            raise RenderCancelled("cancelled before start")
        req = Request(self._req_id("stream"), "speech", self.choose_kernel(backend))
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
        store = WindowStore(jobs, n_total, tmp_dir or tempfile.gettempdir())
        ev = threading.Event()
        try:
            futs = self._submit_all(req, jobs, [store.sink(k) for k in range(len(jobs))], ev)
            emitted = 0
            for k, fut in enumerate(futs):
                self._wait([fut], ev, cancel)
                # samples before the next window's start get nothing from later windows: they are final now
                limit = 64 * jobs[k + 1].start_block if k + 1 < len(jobs) else n_total
                if limit > emitted:
                    for block in store.chunks(emitted, limit):
                        yield wav_bytes(block, 48000)[44:]
                    emitted = limit
        finally:
            ev.set()
            store.close()

    # music -------------------------------------------------------------------------------------------------
    def check_score(self, score: "dict | str", *, max_voices: int | None = None) -> "ScoreReport":
        """Validate ``score``; ``max_voices`` fixes the ensemble limit for this request (None = the engine's
        current setting)."""
        from .music.estimate import DEFAULT_COMPILE_S, estimate_kernels
        from .music.schema import validate_score
        c = self.config
        limit = c.max_voices if max_voices is None else int(max_voices)
        report = validate_score(score, g2p=_LazyG2P(), max_voices=limit, max_seconds=c.max_score_seconds,
                                max_voice_seconds=c.max_voice_seconds)
        if not report.ok:
            return report
        st = self.scheduler.status()
        est = estimate_kernels([v.seconds for v in report.voices], workers=self.workers, rtf=st["rtf"],
                               numba_state=st["numba_state"], compile_s=DEFAULT_COMPILE_S)
        return dataclasses.replace(report, estimate_s=est)

    def render_score(self, score: "dict | str", out_dir: str, stem: str, *, want_mp3: bool = False,
                     progress: Callable[[float], None] | None = None,
                     cancel: threading.Event | None = None, tmp_dir: str | None = None,
                     max_voices: int | None = None,
                     backend: "Literal['auto', 'numba', 'numpy'] | None" = None) -> ScoreResult:
        """Render a score into ``<out_dir>/<stem>.wav``. Voice tracks and the mix sum live in a temporary folder
        under ``tmp_dir`` (default ``out_dir``) that is removed afterwards; give a folder your housekeeping cleans,
        because a killed process cannot remove it. ``max_voices``/``backend`` fix those settings for this request
        (None = the engine's current settings)."""
        if want_mp3:
            raise ValueError("render_score writes WAV only; convert to MP3 outside the engine (want_mp3=False)")
        self._need_numpy()
        from .music.compile import compile_score
        from .music.mixer import MixWriter
        from .music.planning import planner_api
        from .scheduler import FileSink, Request
        t_start = time.perf_counter()
        report = self.check_score(score, max_voices=max_voices)
        if not report.ok:
            raise ScoreError([i for i in report.issues if i.severity == "error"])
        if cancel is not None and cancel.is_set():
            raise RenderCancelled("cancelled before start")
        jobs, mixes, n_mix = compile_score(report, g2p=planner_api().default_g2p())
        norm = report.normalized
        stereo = bool(norm["stereo"]) if norm["stereo"] != "auto" else any(v["pan"] != 0 for v in norm["voices"])
        req = Request(self._req_id("score"), "music", self.choose_kernel(backend))
        os.makedirs(out_dir, exist_ok=True)
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
        tmp = os.path.join(tmp_dir or out_dir, f".chimera-tmp-{uuid.uuid4().hex[:12]}")
        os.makedirs(tmp, mode=0o700)
        ev = threading.Event()
        sinks: list = []
        total = float(sum(j.n_out for j in jobs)) or 1.0
        last = {"p": -1.0}

        def report_progress(p: float) -> None:
            if progress is not None and p > last["p"]:
                last["p"] = p
                progress(p)

        try:
            sinks = [FileSink(os.path.join(tmp, f"voice-{i}.f32"), j.n_out) for i, j in enumerate(jobs)]
            futs = self._submit_all(req, jobs, sinks, ev)
            report_progress(0.0)
            self._wait(futs, ev, cancel, tick=lambda: report_progress(
                0.95 * sum(s.written for s in sinks) / total))
            report_progress(0.95)
            if cancel is not None and cancel.is_set():
                raise RenderCancelled("cancelled")
            for s in sinks:
                s.close()
            writer = MixWriter(tmp, n_mix, stereo=stereo)
            try:
                for s, j, m in zip(sinks, jobs, mixes):
                    writer.add_voice(s.path, j.n_out, m)
                wav = os.path.join(out_dir, f"{stem}.wav")
                info = writer.finish(wav, normalize=bool(norm["normalize"]))
            except BaseException:
                writer.abort()
                raise
            warnings = [i.message for i in report.issues if i.severity == "warning"]
            report_progress(1.0)
            return ScoreResult(wav, None, n_mix / 48000.0, 48000, 2 if stereo else 1, len(jobs), req.kernel,
                               time.perf_counter() - t_start, warnings, info)
        except BaseException:
            ev.set()
            raise
        finally:
            for s in sinks:
                s.close()
            shutil.rmtree(tmp, ignore_errors=True)

    # backend -------------------------------------------------------------------------------------------------
    def status(self) -> BackendStatus:
        from .scheduler import _marker_ok, _numba_installed
        importlib.invalidate_caches()
        numpy_ok = _has("numpy")
        installed = _numba_installed()
        version = None
        if installed:
            try:
                from importlib.metadata import version as _v
                version = _v("numba")
            except Exception:  # noqa: BLE001
                version = None
        st = self.scheduler.status()
        restart = (numpy_ok and not self._numpy_at_start) or (installed and not self._numba_at_start)
        return BackendStatus(numpy_ok, installed, version, st["numba_state"], st["numba_error"], self.cache_base,
                             _marker_ok(self.cache_base) if installed else None, dict(st["rtf"]), self.workers,
                             st["workers_alive"], bool(restart))

    def warmup(self) -> None:
        """Non-blocking: start a numba warm-up when numba is installed, not disabled and not warm (an earlier failure
        is retried - this is the explicit user request), then start one idle worker and load the dictionary."""
        try:
            self.scheduler.start_warmup(retry_failed=True)
        except Exception:  # noqa: BLE001 - warm-up is best effort
            pass
        threading.Thread(target=self._warm_bg, name="chimera-warmup", daemon=True).start()

    def _warm_bg(self) -> None:
        try:
            self.scheduler._spawn_idle()
            from .music.planning import planner_api
            planner_api().default_g2p()
        except Exception:  # noqa: BLE001 - warm-up is best effort
            pass

    def numba_cache_dir(self) -> str | None:
        """The numba cache folder this package version uses (``<cache_dir>/<hash of the JIT files>``); None if it
        cannot be determined. Housekeeping must keep it: numba does not touch the folder on a cache hit."""
        try:
            from .dsp.backend import numba_cache_dir
            return numba_cache_dir(self.cache_base)
        except Exception:  # noqa: BLE001
            return None

    def hold_warm(self, on: bool) -> None:
        self.scheduler.hold_warm(on)

    def selftest(self) -> Any:
        self._need_numpy()
        return self.scheduler.selftest(self.scheduler.choose_kernel())

    def shutdown(self) -> None:
        self.scheduler.shutdown()


def get_engine(config: EngineConfig | None = None) -> Engine:
    """The process-wide engine of this package version (runtime registry); the first config wins."""
    from .runtime import _runtime
    return _runtime(config).engine


def reconfigure(config: EngineConfig) -> Engine:
    """Shut the registered engine down and replace it with one built from ``config``."""
    from .runtime import _replace
    return _replace(config).engine
