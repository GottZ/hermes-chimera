"""RenderScheduler: worker processes for both kernels, one kernel per request, a reserved speech slot,
cancellation by process end. No numpy import at module level.

Slots: ``workers`` = W; music runs on at most max(1, W − 1) workers, speech may use every free slot, and the total
is max(1, W − 1) + 1 (with W = 1 speech starts a second worker instead of waiting). FIFO within a priority, speech
first. Cancellation and idle time-outs are watched by one maintenance thread (50 ms poll); a running job is cancelled
by ending its worker (terminate, wait 2 s, kill, wait - always reaped)."""

from __future__ import annotations

import collections
import concurrent.futures
import importlib.util
import itertools
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .music.compile import RenderJob

THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
CHUNK = 48000
POLL_S = 0.05


@dataclass(frozen=True)
class Request:
    id: str
    priority: Literal["speech", "music"]
    kernel: Literal["numba", "numpy"]        # fixed at submit: all jobs of one request use the same kernel


class PcmSink(Protocol):
    def write(self, offset: int, pcm: "np.ndarray") -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class RenderStats:
    n_out: int
    seconds: float
    kernel: str
    worker_pid: int | None


def default_workers(cpu_count: int | None) -> int:
    """EngineConfig.workers = 0 -> min(4, max(2, cpu_count − 1)) (cpu_count None -> 2)."""
    return min(4, max(2, (cpu_count or 1) - 1))


class MemorySink:
    """PcmSink into a preallocated float32 array of ``n`` samples (speech windows)."""

    def __init__(self, n: int) -> None:
        import numpy as np
        self._data = np.zeros(int(n), np.float32)
        self.written = 0

    @property
    def data(self) -> "np.ndarray":
        return self._data

    def write(self, offset: int, pcm: "np.ndarray") -> None:
        k = len(pcm)
        self._data[offset:offset + k] = pcm
        self.written += k

    def close(self) -> None:
        pass


class FileSink:
    """PcmSink into a raw little-endian float32 file of ``n`` samples (music voices: memory stays O(chunk))."""

    def __init__(self, path: str, n: int) -> None:
        self.path = path
        self.n = int(n)
        self.written = 0
        self._f = open(path, "w+b")
        self._f.truncate(4 * self.n)

    def write(self, offset: int, pcm: "np.ndarray") -> None:
        import numpy as np
        self._f.seek(4 * int(offset))
        self._f.write(np.ascontiguousarray(pcm, dtype="<f4").tobytes())
        self.written += len(pcm)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


# --- worker process ---------------------------------------------------------------------------------------------

class _Worker:
    def __init__(self, sched: "RenderScheduler") -> None:
        from .worker import BOOTSTRAP, write_frame
        import json
        env = dict(os.environ)
        env.update(THREAD_ENV)
        kw: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover - Windows
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        self.proc = subprocess.Popen([sys.executable, "-I", "-S", "-u", "-c", BOOTSTRAP], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=sched.root, env=env,
                                     close_fds=True, **kw)
        self.pid = self.proc.pid
        self.job_handle = _assign_job_object(self.proc) if os.name == "nt" else None
        self.err_tail: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._drain, name=f"chimera-worker-{self.pid}-stderr", daemon=True).start()
        self.last_used = time.monotonic()
        self.dead = False
        try:
            self.proc.stdin.write((json.dumps(sched.handshake()) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except OSError:
            self.kill()
            raise
        self._write = write_frame

    def _drain(self) -> None:
        try:
            for line in iter(self.proc.stderr.readline, b""):
                self.err_tail.append(line.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def alive(self) -> bool:
        return not self.dead and self.proc.poll() is None

    def send(self, msg: dict) -> None:
        self._write(self.proc.stdin, msg)

    def read(self):
        from .worker import read_frame
        return read_frame(self.proc.stdout)

    def tail(self) -> str:
        return " | ".join(list(self.err_tail)[-6:])

    def kill(self) -> None:
        """terminate, wait 2 s, kill, wait - the process is always reaped."""
        self.dead = True
        self._close_job()
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            except OSError:
                pass
        self.proc.wait()
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                s.close()
            except OSError:
                pass

    def _close_job(self) -> None:
        h, self.job_handle = self.job_handle, None
        if h:  # pragma: no cover - Windows
            import ctypes
            ctypes.WinDLL("kernel32").CloseHandle(h)

    def quit(self) -> None:
        self.dead = True
        try:
            self.send({"req": "quit", "op": "quit"})
            self.proc.stdin.close()
            self.proc.wait(2.0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        self.kill()


def _assign_job_object(proc: subprocess.Popen) -> Any:  # pragma: no cover - Windows only
    """Windows: put the worker into a job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The handle stays open in
    this process; when the process ends (even killed), the OS closes it and ends the worker, so a dying gateway
    leaves no rendering orphans."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                          "OtherOperationCount", "ReadTransferCount",
                                                          "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # Extended info
            k32.CloseHandle(job)
            return None
        if not k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:  # noqa: BLE001 - best effort; the EOF watch still ends the worker
        return None


# --- scheduler --------------------------------------------------------------------------------------------------

@dataclass
class _Task:
    req: Request
    job: "RenderJob"
    sink: PcmSink
    cancel: threading.Event
    future: concurrent.futures.Future
    worker: _Worker | None = None
    cancelled: bool = False
    seq: int = 0


class RenderScheduler:
    def __init__(self, *, workers: int, cache_base: str | None, root: str, idle_s: float = 60.0,
                 in_process_fallback: bool = True, backend: Literal["auto", "numba", "numpy"] = "auto") -> None:
        self.workers = max(1, int(workers))
        self.music_cap = max(1, self.workers - 1)
        self.total_cap = self.music_cap + 1
        self.cache_base = cache_base
        self.root = str(root)
        self.idle_s = float(idle_s)
        self.in_process_fallback = in_process_fallback
        self.backend = backend
        self._lock = threading.RLock()
        self._queues: dict[str, collections.deque[_Task]] = {"speech": collections.deque(),
                                                              "music": collections.deque()}
        self._running: list[_Task] = []
        self._idle: list[_Worker] = []
        self._busy: list[_Worker] = []
        self._held = False
        self._seq = itertools.count()
        self._maint: threading.Thread | None = None
        self._stop = threading.Event()
        self._numba_crashes = 0
        self._numba_failed = False
        self._numba_error: str | None = None
        self._warming = False
        self._warm_started = False
        self._spawn_failed = False
        self._inproc_lock = threading.Lock()
        from .music.estimate import DEFAULT_RTF
        self._rtf = dict(DEFAULT_RTF)
        self._closed = False

    # kernel choice --------------------------------------------------------------------------------------------
    def numba_state(self) -> str:
        if not _numba_installed():
            return "absent"
        if self._numba_failed:
            return "failed"
        if _marker_ok(self.cache_base):
            return "ready"
        return "warming" if self._warming else "cold"

    def choose_kernel(self) -> Literal["numba", "numpy"]:
        if self.backend == "numpy" or self._numba_failed or self._spawn_failed or not _numba_installed():
            return "numpy"
        if _marker_ok(self.cache_base):
            return "numba"
        self._start_warm()
        return "numpy"

    def _start_warm(self) -> None:
        with self._lock:
            if self._warm_started or self._closed:
                return
            self._warm_started = True
            self._warming = True
        threading.Thread(target=self._warm, name="chimera-warm", daemon=True).start()

    def _warm(self) -> None:
        w = None
        try:
            w = _Worker(self)
            w.send({"req": "warm", "op": "warm"})
            while True:
                msg, _ = w.read()
                if msg.get("req") == "warm":
                    break
            if msg.get("ok") is False:
                self._numba_failed = True
                self._numba_error = f"warm-up failed: {msg.get('error')}"
        except (EOFError, OSError, ValueError) as e:
            self._numba_failed = True
            self._numba_error = f"warm-up worker died: {e} {w.tail() if w else ''}".strip()
        finally:
            self._warming = False
            if w is not None:
                w.quit()

    # submit ---------------------------------------------------------------------------------------------------
    def submit(self, req: Request, job: "RenderJob", *, sink: PcmSink,
               cancel: threading.Event) -> "concurrent.futures.Future[RenderStats]":
        fut: concurrent.futures.Future = concurrent.futures.Future()
        task = _Task(req, job, sink, cancel, fut, seq=next(self._seq))
        with self._lock:
            if self._closed:
                from . import BackendError
                fut.set_exception(BackendError("scheduler is shut down"))
                return fut
            self._queues[req.priority].append(task)
            self._ensure_maint()
            self._pump()
        return fut

    def _ensure_maint(self) -> None:
        if self._maint is None or not self._maint.is_alive():
            self._maint = threading.Thread(target=self._maintain, name="chimera-scheduler", daemon=True)
            self._maint.start()

    def _pump(self) -> None:
        """Start every runnable task (lock held)."""
        while True:
            total = len(self._running)
            music = sum(1 for t in self._running if t.req.priority == "music")
            if self._queues["speech"] and total < self.total_cap:
                task = self._queues["speech"].popleft()
            elif self._queues["music"] and total < self.total_cap and music < self.music_cap:
                task = self._queues["music"].popleft()
            else:
                return
            if task.cancel.is_set():
                self._cancelled(task)
                continue
            self._running.append(task)
            threading.Thread(target=self._run, args=(task,), name=f"chimera-job-{task.job.id}",
                             daemon=True).start()

    def _cancelled(self, task: _Task) -> None:
        from . import RenderCancelled
        if not task.future.done():
            task.future.set_exception(RenderCancelled(f"request {task.req.id} cancelled"))

    def _finish(self, task: _Task, *, result: RenderStats | None = None, error: BaseException | None = None) -> None:
        with self._lock:
            if task in self._running:
                self._running.remove(task)
            self._pump()
        if not task.future.done():
            if error is not None:
                task.future.set_exception(error)
            else:
                task.future.set_result(result)

    # workers --------------------------------------------------------------------------------------------------
    def handshake(self) -> dict:
        import site
        cwd = os.path.normcase(os.path.abspath(os.getcwd()))
        path = []
        for p in sys.path:
            if p in ("", "."):
                continue
            ap = os.path.abspath(p)
            if os.path.normcase(ap) == cwd:
                continue
            path.append(p if os.path.isabs(p) else ap)
        dirs: list[str] = []
        try:
            dirs = list(site.getsitepackages())
        except AttributeError:  # pragma: no cover - embedded interpreters
            pass
        try:
            dirs.append(site.getusersitepackages())
        except AttributeError:  # pragma: no cover
            pass
        # -I makes the worker ignore PYTHONDONTWRITEBYTECODE: hand the host's effective setting over explicitly
        return {"proto": 2, "sys_path": path, "site_dirs": [d for d in dirs if d in path], "root": self.root,
                "cache_base": self.cache_base, "dont_write_bytecode": bool(sys.dont_write_bytecode)}

    def _acquire(self) -> _Worker:
        with self._lock:
            while self._idle:
                w = self._idle.pop()
                if w.alive():
                    self._busy.append(w)
                    return w
                w.kill()
        w = _Worker(self)
        with self._lock:
            self._busy.append(w)
        return w

    def _release(self, w: _Worker) -> None:
        with self._lock:
            if w in self._busy:
                self._busy.remove(w)
            if w.alive() and not self._closed:
                w.last_used = time.monotonic()
                self._idle.append(w)
                return
        w.kill()

    def _discard(self, w: _Worker) -> None:
        with self._lock:
            if w in self._busy:
                self._busy.remove(w)
            if w in self._idle:
                self._idle.remove(w)
        w.kill()

    # running one task -----------------------------------------------------------------------------------------
    def _run(self, task: _Task) -> None:
        from . import BackendError, ChimeraError, RenderCancelled
        rid_base = f"{task.req.id}:{task.job.id}"
        for attempt in (1, 2):
            if task.cancel.is_set():
                return self._finish(task, error=RenderCancelled(f"request {task.req.id} cancelled"))
            try:
                w = self._acquire()
            except OSError as e:
                with self._lock:
                    self._spawn_failed = True
                if self.in_process_fallback:
                    return self._run_in_process(task)
                return self._finish(task, error=BackendError(f"no render worker could start ({e}) and the "
                                                             f"in-process fallback is off"))
            task.worker = w
            if task.cancel.is_set():
                self._release(w)
                return self._finish(task, error=RenderCancelled(f"request {task.req.id} cancelled"))
            rid = f"{rid_base}:{attempt}"
            t0 = time.perf_counter()
            try:
                w.send({"req": rid, "op": "render", "job": task.job.to_json(), "kernel": task.req.kernel,
                        "chunk": CHUNK})
                while True:
                    msg, raw = w.read()
                    if msg.get("req") != rid:
                        continue
                    if "chunk" in msg:
                        import numpy as np
                        task.sink.write(int(msg["offset"]), np.frombuffer(raw, dtype="<f4"))
                    elif msg.get("done"):
                        secs = float(msg.get("seconds", time.perf_counter() - t0))
                        self._note_rtf(task.req.kernel, secs, task.job.n_out)
                        self._release(w)
                        return self._finish(task, result=RenderStats(task.job.n_out, secs, task.req.kernel, w.pid))
                    elif msg.get("ok") is False:
                        self._release(w)
                        return self._finish(task, error=ChimeraError(f"render failed: {msg.get('error')}"))
            except (EOFError, OSError, ValueError) as e:
                tail = w.tail()
                self._discard(w)
                if task.cancelled or task.cancel.is_set():
                    return self._finish(task, error=RenderCancelled(f"request {task.req.id} cancelled"))
                if task.req.kernel == "numba":
                    with self._lock:
                        self._numba_crashes += 1
                        if self._numba_crashes >= 2:
                            self._numba_failed = True
                            self._numba_error = f"numba render worker crashed twice; last: {tail or e}"
                if attempt == 2:
                    return self._finish(task, error=BackendError(
                        f"render worker died twice on job {task.job.id}: {tail or type(e).__name__}"))
            except Exception as e:  # noqa: BLE001 - sink errors etc.
                self._discard(w)
                return self._finish(task, error=e)

    def _run_in_process(self, task: _Task) -> None:
        from . import RenderCancelled
        from .music.compile import render_job
        try:
            with self._inproc_lock:
                t0 = time.perf_counter()
                written = {"n": 0}
                buf_holder: dict = {}

                def progress(pos: int) -> None:
                    a = written["n"]
                    task.sink.write(a, buf_holder["buf"][a:pos])
                    written["n"] = pos

                import numpy as np
                buf_holder["buf"] = np.zeros(task.job.n_out, np.float32)
                render_job(task.job, None, None, out=buf_holder["buf"], chunk=CHUNK, progress=progress,
                           cancel=task.cancel.is_set)
                secs = time.perf_counter() - t0
            self._finish(task, result=RenderStats(task.job.n_out, secs, "numpy", None))
        except RenderCancelled as e:
            self._finish(task, error=e)
        except Exception as e:  # noqa: BLE001
            self._finish(task, error=e)

    def _note_rtf(self, kernel: str, secs: float, n_out: int) -> None:
        audio = n_out / 48000.0
        if audio >= 0.5:
            with self._lock:
                self._rtf[kernel] = 0.7 * self._rtf[kernel] + 0.3 * (secs / audio)

    # maintenance ----------------------------------------------------------------------------------------------
    def _maintain(self) -> None:
        while not self._stop.wait(POLL_S):
            kill: list[_Worker] = []
            with self._lock:
                for q in self._queues.values():
                    for task in [t for t in q if t.cancel.is_set()]:
                        q.remove(task)
                        self._cancelled(task)
                for task in self._running:
                    if task.cancel.is_set() and not task.cancelled:
                        task.cancelled = True
                        if task.worker is not None:
                            kill.append(task.worker)
                now = time.monotonic()
                keep = 1 if self._held else 0
                idle_sorted = sorted(self._idle, key=lambda w: w.last_used, reverse=True)
                for w in idle_sorted[keep:]:
                    if now - w.last_used > self.idle_s or not w.alive():
                        self._idle.remove(w)
                        kill.append(w)
                if self._held and not self._idle and not self._busy and not self._closed:
                    threading.Thread(target=self._spawn_idle, daemon=True).start()
            for w in kill:
                if w.dead:
                    continue
                if w in self._busy:
                    w.kill()
                else:
                    w.quit()

    def _spawn_idle(self) -> None:
        with self._lock:
            if self._idle or self._busy or getattr(self, "_spawning", False):
                return
            self._spawning = True
        try:
            w = _Worker(self)
            with self._lock:
                self._idle.append(w)
        except OSError:
            pass
        finally:
            self._spawning = False

    def selftest(self, kernel: str) -> Any:
        """Run the selftest op in a fresh worker; returns dsp.backend.SelfTestResult (or a dict from fakes)."""
        w = _Worker(self)
        try:
            w.send({"req": "selftest", "op": "selftest", "kernel": kernel})
            while True:
                msg, _ = w.read()
                if msg.get("req") == "selftest":
                    break
        finally:
            w.quit()
        if msg.get("ok") is False:
            from . import BackendError
            raise BackendError(f"selftest failed: {msg.get('error')}")
        res = msg.get("selftest", {})
        try:
            from .dsp.backend import SelfTestResult
            return SelfTestResult(**res)
        except (ImportError, TypeError):
            return res

    def hold_warm(self, on: bool) -> None:
        with self._lock:
            self._held = bool(on)
            self._ensure_maint()

    def status(self) -> dict:
        with self._lock:
            procs = [w for w in self._idle + self._busy if w.alive()]
            return {"workers": self.workers, "workers_alive": len(procs), "worker_pids": [w.pid for w in procs],
                    "running": len(self._running), "queued_speech": len(self._queues["speech"]),
                    "queued_music": len(self._queues["music"]), "numba_state": self.numba_state(),
                    "numba_error": self._numba_error, "rtf": dict(self._rtf), "held_warm": self._held,
                    "numba_installed": _numba_installed(), "marker": _marker_ok(self.cache_base),
                    "in_process": self._spawn_failed}

    def shutdown(self) -> None:
        from . import BackendError
        with self._lock:
            self._closed = True
            self._stop.set()
            queued = [t for q in self._queues.values() for t in q]
            for q in self._queues.values():
                q.clear()
            running = list(self._running)
            for t in running:
                t.cancelled = True
            workers = self._idle + self._busy
            self._idle, self._busy = [], []
        for t in queued:
            if not t.future.done():
                t.future.set_exception(BackendError("scheduler shut down"))
        for w in workers:
            if any(t.worker is w for t in running):
                w.kill()
            else:
                w.quit()


def _numba_installed() -> bool:
    """find_spec("numba") is not None - never imports numba: compiled machine code runs only in worker processes,
    where a crash cannot take the host down."""
    try:
        return importlib.util.find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


def _marker_ok(cache_base: str | None) -> dict | None:
    """dsp.backend.marker_ok(cache_base) (a file read, no numba import); None if the backend is unavailable."""
    if cache_base is None:
        return None
    try:
        from .dsp.backend import marker_ok
    except ImportError:
        return None
    try:
        return marker_ok(cache_base)
    except Exception:  # noqa: BLE001 - a broken marker is no marker
        return None
