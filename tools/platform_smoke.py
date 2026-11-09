#!/usr/bin/env python3
"""Platform smoke: real renders through real worker processes on this OS, checked against the in-repo references,
then a cancelled render and a killed host, after which no render worker may be left. Written for CI (Linux, macOS,
Windows); runs anywhere the test suite runs.

    python tools/platform_smoke.py --out DIR [--expect-numba yes|no|auto]

Phases (numpy always, numba when it is importable):
  warm-up  numba only. Two engines warm up on one fresh cache folder at the same time (one worker compiles, the other
           waits for its marker: lock file, atomic replace); then the marker is removed and a third warm-up runs on
           the kept cache - it must not compile again (no cache file added or changed) and take at most half as long.
           The cache folder name contains a space and a non-ASCII letter, as user profile paths do.
  speech   Engine.speak("Hello world.", "male", faithful=True) through a worker = the template's render
           (tests/vectors/audio-head.json: length exact, the first 4 800 samples T3, the file peak within T3);
           Engine.speak_to_wav of a sentence that splits into several phrase windows (parallel workers).
  score    a two-voice canon (female "ta ta ta ta" at beats 0 and 8, seed 0, lead_in 0.3 s) through the scheduler on
           two worker processes against the native template renders in tests/measure/data/canon-native.json (voice
           at beat 0: length, onsets, samples around the first onset T3; voice at beat 8: onsets within one glottal
           period) and Engine.render_score of the same score = the 16-bit encoding of the in-memory mix, byte for byte.
  cancel   a long two-voice score is cancelled while two workers render: RenderCancelled, both workers ended and
           reaped, no output or temporary file left.
  host     a child process renders and is killed hard; its workers must end (EOF watch; Windows: job object).
A process sweep (ps / Win32_Process) after the engines shut down and after the host kill must find no worker.
Writes report.json and the rendered WAVs to --out; exit status 1 if any check failed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "tests"))      # the suite's helpers (support, support.pitch): same tolerances

WORKER_MARK = "chimera_vocal.worker"         # in the command line of every render worker (scheduler BOOTSTRAP)
SPAWNED: list[int] = []                      # pid of every worker the scheduler started in this process
TAG_VAR = "CHIMERA_SMOKE_TAG"                # set once per run; workers inherit it (scheduler copies os.environ)
BASELINE: set[int] = set()                   # worker processes that existed before this run
HELLO = "Hello world."
SENTENCE = "Hello from chimera. This is a test, in two parts."
BPM = 120
TA_PITCHES = (62, 64, 65, 67)


# --- small helpers ---------------------------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.data: dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
        return bool(ok)

    def fail(self, name: str, exc: BaseException) -> None:
        self.check(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def wait_until(pred: Callable[[], bool], timeout: float, step: float = 0.05) -> float | None:
    t0 = time.monotonic()
    while True:
        if pred():
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(step)


def track_workers() -> None:
    """Record the pid of every render worker the scheduler starts (warm-up, render, idle and selftest workers)."""
    import chimera_vocal.scheduler as sched
    orig = sched._Worker.__init__

    def init(self, s):
        orig(self, s)
        SPAWNED.append(self.pid)

    sched._Worker.__init__ = init


def pid_state(pid: int) -> str:
    """"gone", "zombie" (ended, not reaped) or "alive"."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return "gone"
        return "zombie" if stat.rsplit(")", 1)[1].split()[0] in ("Z", "X") else "alive"
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000 | 0x00100000, False, pid)     # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if not h:
            return "gone"
        try:
            return "gone" if k32.WaitForSingleObject(h, 0) == 0 else "alive"
        finally:
            k32.CloseHandle(h)
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if not out:
        return "gone"
    return "zombie" if out.startswith("Z") else "alive"


def sweep() -> list[str]:
    """Every running process (other than this one) whose command line names the render worker module. Catches
    workers the pid bookkeeping cannot see, e.g. the real interpreter behind a Windows venv launcher. On Linux and
    macOS only processes that inherited this run's tag count (other test runs on the same machine are not ours);
    processes that existed before this run started are ignored everywhere."""
    me = os.getpid()
    tag = f"{TAG_VAR}={os.environ.get(TAG_VAR, '')}"
    found: list[str] = []
    if sys.platform.startswith("linux"):
        for d in Path("/proc").iterdir():
            if not d.name.isdigit() or int(d.name) == me or int(d.name) in BASELINE:
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                if WORKER_MARK not in cmd:
                    continue
                env = (d / "environ").read_bytes().split(b"\0")
            except OSError:
                continue
            if tag.encode() in env:
                found.append(f"{d.name} {cmd[:60]!r}")
        return found
    if os.name == "nt":
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and "
              "$_.CommandLine.Contains('" + WORKER_MARK + "') } | ForEach-Object { \"$($_.ProcessId) "
              "$($_.ParentProcessId) $($_.Name)\" }")
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip() and int(ln.split()[0]) not in BASELINE]
    out = subprocess.run(["ps", "-axwwE", "-o", "pid=,command="], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if (len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != me and int(parts[0]) not in BASELINE
                and WORKER_MARK in parts[1] and tag in parts[1]):
            found.append(f"{parts[0]} {parts[1][:60]!r}")
    return found


def make_engine(kernel: str, cache: Path, workers: int = 3):
    from chimera_vocal import Engine, EngineConfig
    return Engine(EngineConfig(backend=kernel, workers=workers, in_process_fallback=False, cache_dir=str(cache),
                               worker_idle_s=600.0))


def cache_files(d: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            p = Path(root) / f
            st = p.stat()
            out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns)
    return out


def ta_voice(start: int) -> dict:
    return {"body": "female", "start": start, "seed": 0,
            "notes": [{"pitch": p, "beats": 1, "lyric": "ta"} for p in TA_PITCHES]}


def long_score() -> dict:
    notes_m = [{"pitch": 48 + (i % 5), "beats": 4, "lyric": "ah"} for i in range(16)]
    notes_f = [{"pitch": 60 + (i % 5), "beats": 4, "lyric": "ah"} for i in range(16)]
    return {"bpm": BPM, "voices": [{"body": "male", "seed": 0, "notes": notes_m},
                                   {"body": "female", "seed": 0, "notes": notes_f}]}


def running_worker_pids(eng) -> list[int]:
    with eng.scheduler._lock:
        return [t.worker.pid for t in eng.scheduler._running if t.worker is not None]


def tmp_leftovers(*dirs: Path) -> list[str]:
    left = []
    for d in dirs:
        if d.is_dir():
            left += [p.name for p in d.iterdir() if p.name.startswith(".chimera-tmp-") or ".part-" in p.name]
    return left


# --- phases ----------------------------------------------------------------------------------------------------

def phase_warm(rep: Report, cache: Path) -> Any:
    """numba cold warm-up (two engines, one cache) then warm (marker removed, cache kept). Returns the engine."""
    from chimera_vocal.dsp.backend import MARKER_NAME, numba_cache_dir
    e1 = make_engine("numba", cache, 3)
    e2 = make_engine("numba", cache, 2)
    try:
        rep.check("warm: cold state before warm-up", e1.status().numba_state == "cold" == e2.status().numba_state,
                  f"{e1.status().numba_state}/{e2.status().numba_state}")
        t0 = time.monotonic()
        e1.warmup()
        e2.warmup()
        took = wait_until(lambda: all(e.status().numba_state in ("ready", "failed") for e in (e1, e2)), 900, 0.1)
        t_cold = time.monotonic() - t0
        s1, s2 = e1.status(), e2.status()
        rep.data["warm_cold_s"] = round(t_cold, 2)
        rep.check("warm: cold warm-up of two engines ends ready", took is not None and s1.numba_state == "ready"
                  and s2.numba_state == "ready", f"{s1.numba_state} ({s1.numba_error}) / {s2.numba_state} "
                  f"({s2.numba_error}) after {t_cold:.1f} s")
        d = Path(numba_cache_dir(str(cache)))
        a = cache_files(d)
        rep.data["cache_files"] = len(a)
        rep.check("warm: cache folder holds numba index and data files and the marker",
                  any(n.endswith(".nbi") for n in a) and any(n.endswith(".nbc") for n in a) and MARKER_NAME in a,
                  f"{sum(n.endswith('.nbi') for n in a)} nbi, {sum(n.endswith('.nbc') for n in a)} nbc in {d.name}")
        m = s1.marker or {}
        st = m.get("selftest") or {}
        rep.data["warm_selftest"] = st
        rep.check("warm: marker selftest passed", st.get("passed") is True, f"max_rel_err {st.get('max_rel_err')}")
        rep.check("warm: no lock file left", not (d / ".warm.lock").exists())
        e2.shutdown()
        os.remove(d / MARKER_NAME)
        rep.check("warm: state cold once the marker is gone", e1.status().numba_state == "cold",
                  e1.status().numba_state)
        t1 = time.monotonic()
        e1.warmup()
        took = wait_until(lambda: e1.status().numba_state in ("ready", "failed"), 900, 0.1)
        t_warm = time.monotonic() - t1
        rep.data["warm_warm_s"] = round(t_warm, 2)
        rep.check("warm: warm-up on the kept cache ends ready", took is not None and
                  e1.status().numba_state == "ready", f"{e1.status().numba_state} after {t_warm:.1f} s")
        b = cache_files(d)
        added = sorted(set(b) - set(a) - {MARKER_NAME})
        changed = sorted(n for n in a if n in b and n != MARKER_NAME and a[n] != b[n])
        rep.check("warm: kept cache reused (no cache file added or rewritten)", not added and not changed,
                  f"added {added[:5]}, changed {changed[:5]}")
        rep.check("warm: warm run at most half the cold run", t_warm <= 0.5 * t_cold,
                  f"cold {t_cold:.1f} s, warm {t_warm:.1f} s")
        return e1
    except BaseException:
        e1.shutdown()
        e2.shutdown()
        raise


def phase_speech(rep: Report, eng, kernel: str, out: Path, work: Path, keep: dict) -> None:
    import numpy as np

    import support
    from chimera_vocal.core.wavio import read_wav
    case = next(c for c in support.vector("audio-head")["cases"] if c["name"] == "speech_male_hello_world")
    t0 = time.monotonic()
    audio, k = eng.speak(HELLO, "male", faithful=True)
    dt = time.monotonic() - t0
    x = np.asarray(audio.samples)
    keep["hello"] = x
    rep.data[f"{kernel}_hello_s"] = round(dt, 2)
    rep.check(f"{kernel} speech: kernel used", k == kernel, k)
    rep.check(f"{kernel} speech: length = template", x.size == case["total_samples"],
              f"{x.size} vs {case['total_samples']}")
    rep.check(f"{kernel} speech: finite", bool(np.all(np.isfinite(x))))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rep.check(f"{kernel} speech: peak <= 1", peak <= 1.0, f"{peak:.6f}")
    d = support.compare_audio(support.f32(case["head"]), x[:case["head_samples"]])
    rep.data[f"{kernel}_hello_head_err"] = d.max_err
    rep.check(f"{kernel} speech: first {case['head_samples']} samples = template head (T3)", d.first_diverge is None,
              f"max err {d.max_err:.3e} x head peak")
    rel_peak = abs(peak - case["peak_full"]) / case["peak_full"]
    rep.check(f"{kernel} speech: file peak = template file peak (T3)", rel_peak <= 1e-6, f"{rel_peak:.3e}")
    audio.to_wav(out / f"speech-hello-faithful-{kernel}.wav")

    jobs, n_total = eng._plan_speech(SENTENCE, "female", None, True)
    rep.check(f"{kernel} speech_to_wav: sentence splits into several windows", len(jobs) >= 2, f"{len(jobs)} windows")
    tmp = work / f"speech-tmp-{kernel}"
    path = out / f"speech-sentence-{kernel}.wav"
    t0 = time.monotonic()
    res = eng.speak_to_wav(SENTENCE, str(path), "female", tmp_dir=str(tmp))
    rep.data[f"{kernel}_sentence_s"] = round(time.monotonic() - t0, 2)
    rep.data[f"{kernel}_sentence_audio_s"] = round(res.seconds, 3)
    rep.check(f"{kernel} speech_to_wav: kernel used", res.kernel == kernel, res.kernel)
    sr, y = read_wav(path)
    rep.check(f"{kernel} speech_to_wav: WAV length = plan", sr == 48000 and y.size == n_total,
              f"{sr} Hz, {y.size} vs {n_total}")
    rep.check(f"{kernel} speech_to_wav: finite, peak <= 1", bool(np.all(np.isfinite(y))) and
              float(np.max(np.abs(y))) <= 1.0, f"peak {float(np.max(np.abs(y))):.4f}")
    left = [p.name for p in tmp.iterdir()] if tmp.is_dir() else []
    rep.check(f"{kernel} speech_to_wav: temporary window file removed", not left, str(left))


def phase_score(rep: Report, eng, kernel: str, out: Path, work: Path, keep: dict) -> None:
    import numpy as np

    from chimera_vocal.core.wavio import read_wav, wav_bytes
    from chimera_vocal.music.compile import compile_score
    from chimera_vocal.music.mixer import mix
    from chimera_vocal.music.planning import planner_api
    from chimera_vocal.scheduler import MemorySink, Request
    from support.pitch import hz, onsets
    native = json.loads((REPO / "tests" / "measure" / "data" / "canon-native.json").read_text(encoding="utf-8"))
    entry = {e["start_beats"]: e for e in native["entries"]}
    score = {"bpm": BPM, "normalize": False, "voices": [ta_voice(0), ta_voice(8)]}
    report = eng.check_score(score)
    rep.check(f"{kernel} score: valid", report.ok, str([i.message for i in report.issues][:3]))
    jobs, mixes, n_mix = compile_score(report, g2p=planner_api().default_g2p())
    rep.check(f"{kernel} score: lead-in of the native renders", report.lead_in == entry[0]["t0"], str(report.lead_in))

    req = Request(f"smoke-{kernel}", "music", kernel)
    sinks = [MemorySink(j.n_out) for j in jobs]
    ev = threading.Event()
    t0 = time.monotonic()
    futs = [eng.scheduler.submit(req, dataclasses.replace(j, id=f"smoke-{kernel}.{j.id}"), sink=s, cancel=ev)
            for j, s in zip(jobs, sinks)]
    stats = [f.result(timeout=900) for f in futs]
    rep.data[f"{kernel}_score_s"] = round(time.monotonic() - t0, 2)
    rep.check(f"{kernel} score: kernel used", all(s.kernel == kernel for s in stats), str([s.kernel for s in stats]))
    pids = {s.worker_pid for s in stats}
    rep.check(f"{kernel} score: two voices on two worker processes", len(pids) == 2 and None not in pids, str(pids))

    x0, e0 = sinks[0].data, entry[0]
    rep.check(f"{kernel} score: voice 1 starts at block 0, length = native", jobs[0].start_block == 0 and
              x0.size == e0["n"], f"{x0.size} vs {e0['n']}")
    rep.check(f"{kernel} score: voice 1 finite, peak <= 1", bool(np.all(np.isfinite(x0))) and
              float(np.max(np.abs(x0))) <= 1.0)
    on0 = onsets(x0, threshold=0.2, quiet=4800)
    rep.check(f"{kernel} score: voice 1 onsets = native", on0 == e0["onsets"], f"{on0} vs {e0['onsets']}")
    ref = np.asarray(e0["region"], dtype=np.float64).astype(np.float32).astype(np.float64)
    a = e0["region_start"]
    got = x0[a:a + ref.size].astype(np.float64)
    err = float(np.max(np.abs(got - ref))) / e0["peak"] if got.size == ref.size else float("inf")
    rep.data[f"{kernel}_canon_region_err"] = err
    rep.check(f"{kernel} score: voice 1 samples around the first onset = native (T3)", err <= 1e-6,
              f"{err:.3e} x native peak")

    x1, e8 = sinks[1].data, entry[8]
    rep.check(f"{kernel} score: voice 2 starts late", jobs[1].start_block > 0 and mixes[1].offset ==
              64 * jobs[1].start_block, f"start block {jobs[1].start_block}")
    rep.check(f"{kernel} score: voice 2 finite, peak <= 1", bool(np.all(np.isfinite(x1))) and
              float(np.max(np.abs(x1))) <= 1.0)
    on1 = [o + mixes[1].offset for o in onsets(x1, threshold=0.2, quiet=4800)]
    ok = len(on1) == len(e8["onsets"]) and all(abs(g - n) <= 48000 / hz(p)
                                               for g, n, p in zip(on1, e8["onsets"], TA_PITCHES))
    rep.check(f"{kernel} score: voice 2 onsets within one glottal period of native", ok, f"{on1} vs {e8['onsets']}")

    m, info = mix([s.data for s in sinks], mixes, n_mix, stereo=False, normalize=False)
    rep.check(f"{kernel} score: mix unscaled, voices apart around the first onset", info.gain_db == 0.0 and
              mixes[1].offset > a + ref.size and bool(np.array_equal(m[a:a + ref.size], x0[a:a + ref.size])),
              f"gain {info.gain_db} dB, voice 2 offset {mixes[1].offset}")
    tmp = work / f"score-tmp-{kernel}"
    t0 = time.monotonic()
    res = eng.render_score(score, str(out), f"score-canon-{kernel}", tmp_dir=str(tmp))
    rep.data[f"{kernel}_render_score_s"] = round(time.monotonic() - t0, 2)
    rep.check(f"{kernel} render_score: kernel, voices, mono, length", res.kernel == kernel and res.voices == 2 and
              res.channels == 1 and res.seconds == n_mix / 48000.0, f"{res.kernel}, {res.voices}, {res.channels} ch, "
              f"{res.seconds} s")
    data = Path(res.path_wav).read_bytes()
    rep.check(f"{kernel} render_score: WAV = 16-bit encoding of the in-memory mix (byte for byte)",
              data == wav_bytes(m, 48000), f"{len(data)} bytes")
    sr, y = read_wav(res.path_wav)
    rep.check(f"{kernel} render_score: finite, peak <= 1", bool(np.all(np.isfinite(y))) and
              float(np.max(np.abs(y))) <= 1.0 and y.size == n_mix)
    rep.check(f"{kernel} render_score: no temporary files left", not tmp_leftovers(out, tmp),
              str(tmp_leftovers(out, tmp)))
    keep["score_wav"] = data


def phase_cancel(rep: Report, eng, kernel: str, out: Path, work: Path) -> None:
    from chimera_vocal import RenderCancelled
    ev = threading.Event()
    box: dict[str, Any] = {}
    tmp = work / f"cancel-tmp-{kernel}"

    def go() -> None:
        try:
            box["res"] = eng.render_score(long_score(), str(out), f"cancel-{kernel}", cancel=ev, tmp_dir=str(tmp))
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e

    th = threading.Thread(target=go, daemon=True)
    th.start()
    busy: list[int] = []
    seen = wait_until(lambda: len(running_worker_pids(eng)) >= 2, 120, 0.02)
    if seen is not None:
        time.sleep(1.0)
        busy = running_worker_pids(eng)
    rep.check(f"{kernel} cancel: two workers rendering before the cancel", len(busy) >= 2, str(busy))
    seen_by_sweep = sweep()
    rep.check(f"{kernel} cancel: the process sweep sees the running workers (control for the sweep checks)",
              len(seen_by_sweep) >= len(busy) >= 2, f"{len(seen_by_sweep)} found: {seen_by_sweep[:4]}")
    t0 = time.monotonic()
    ev.set()
    th.join(30)
    rep.data[f"{kernel}_cancel_return_s"] = round(time.monotonic() - t0, 2)
    exc = box.get("exc")
    rep.check(f"{kernel} cancel: render_score raised RenderCancelled", isinstance(exc, RenderCancelled) and
              not th.is_alive(), repr(exc) if exc else ("still running" if th.is_alive() else "finished"))
    gone = wait_until(lambda: all(pid_state(p) == "gone" for p in busy), 10, 0.05)
    rep.check(f"{kernel} cancel: the rendering workers ended and were reaped", gone is not None,
              str({p: pid_state(p) for p in busy}))
    rep.check(f"{kernel} cancel: no output or temporary file left",
              not (out / f"cancel-{kernel}.wav").exists() and not tmp_leftovers(out, tmp), str(tmp_leftovers(out, tmp)))


def phase_host_kill(rep: Report, work: Path, cache: Path) -> None:
    hostdir = work / "host"
    hostdir.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--host-child", str(hostdir),
                              "--cache", str(cache)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
    pids: list[int] = []
    box: dict[str, Any] = {}

    def read() -> None:
        for line in child.stdout:
            if line.startswith("PIDS "):
                box["pids"] = json.loads(line[5:])
                return

    th = threading.Thread(target=read, daemon=True)
    th.start()
    th.join(180)
    pids = box.get("pids") or []
    rep.check("host: child host started two rendering workers", len(pids) >= 2, str(pids))
    child.kill()
    child.wait()
    err = child.stderr.read()[-2000:] if child.stderr else ""
    gone = wait_until(lambda: all(pid_state(p) != "alive" for p in pids), 10, 0.05)
    rep.data["host_workers_end_s"] = round(gone, 2) if gone is not None else None
    rep.check("host: workers end after the host was killed", gone is not None and bool(pids),
              f"{ {p: pid_state(p) for p in pids} } {err[-300:] if gone is None else ''}")
    left = wait_until(lambda: not sweep(), 15, 0.5)
    rep.check("host: no worker process left (process sweep)", left is not None, str(sweep()))


def host_child(hostdir: str, cache: str) -> None:
    """--host-child: render a long score and print the worker pids; the parent kills this process."""
    eng = make_engine("numpy", Path(cache), 3)
    threading.Thread(target=lambda: eng.render_score(long_score(), hostdir, "host", tmp_dir=hostdir),
                     daemon=True).start()
    if wait_until(lambda: len(running_worker_pids(eng)) >= 2, 120, 0.02) is None:
        print("PIDS []", flush=True)
    else:
        print("PIDS " + json.dumps(running_worker_pids(eng)), flush=True)
    time.sleep(600)


# --- main ------------------------------------------------------------------------------------------------------

def environment() -> dict:
    from importlib import metadata

    def ver(n: str) -> str | None:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            return None

    return {"platform": sys.platform, "machine": platform.machine(), "system": platform.platform(),
            "python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "executable": sys.executable, "venv": sys.prefix != sys.base_prefix, "cpus": os.cpu_count(),
            "numpy": ver("numpy"), "numba": ver("numba"), "llvmlite": ver("llvmlite")}


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="smoke-out", help="folder for report.json and the WAVs")
    ap.add_argument("--expect-numba", choices=("yes", "no", "auto"), default="auto",
                    help="fail unless numba is (yes) / is not (no) importable here")
    ap.add_argument("--host-child", help=argparse.SUPPRESS)
    ap.add_argument("--cache", help=argparse.SUPPRESS)
    args = ap.parse_args()
    for k in ("CHIMERA_BACKEND", "CHIMERA_TESTING", "CHIMERA_WORKER_VOICE_FACTORY", "CHIMERA_FAKE"):
        os.environ.pop(k, None)
    if args.host_child:
        host_child(args.host_child, args.cache)
        return 0

    import importlib.util
    import uuid
    os.environ[TAG_VAR] = uuid.uuid4().hex
    BASELINE.update(int(ln.split()[0]) for ln in sweep())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="chimera-smoke-"))
    cache = work / "numba cache ä"              # space and a non-ASCII letter, like many profile paths
    rep = Report()
    rep.data["env"] = environment()
    print(json.dumps(rep.data["env"]), flush=True)
    track_workers()
    has_numba = importlib.util.find_spec("numba") is not None
    if args.expect_numba != "auto":
        rep.check("numba importable as expected for this platform", has_numba == (args.expect_numba == "yes"),
                  f"importable={has_numba}, expected {args.expect_numba}")
    engines: list = []
    keep: dict[str, dict] = {}
    t_start = time.monotonic()
    try:
        kernels = ["numpy"] + (["numba"] if has_numba else [])
        for kernel in kernels:
            keep[kernel] = {}
            eng = None
            try:
                t0 = time.monotonic()
                eng = phase_warm(rep, cache) if kernel == "numba" else make_engine(kernel, cache, 3)
                engines.append(eng)
                phase_speech(rep, eng, kernel, out, work, keep[kernel])
                phase_score(rep, eng, kernel, out, work, keep[kernel])
                phase_cancel(rep, eng, kernel, out, work)
                rep.data[f"{kernel}_phase_s"] = round(time.monotonic() - t0, 2)
            except BaseException as e:  # noqa: BLE001 - report and go on with the next kernel
                rep.fail(f"{kernel}: unexpected error", e)
        if "hello" in keep.get("numpy", {}) and "hello" in keep.get("numba", {}):
            import numpy as np
            a, b = keep["numpy"]["hello"], keep["numba"]["hello"]
            d = float(np.max(np.abs(a.astype(np.float64) - b))) / float(np.max(np.abs(a)))
            rep.data["numba_vs_numpy_hello_err"] = d
            rep.check("numba vs numpy: faithful speech within T3", d <= 1e-6, f"{d:.3e} x peak")
            rep.data["numba_vs_numpy_score_wav_equal"] = keep["numpy"].get("score_wav") == keep["numba"].get(
                "score_wav")
        for eng in engines:
            eng.shutdown()
        engines.clear()
        ended = wait_until(lambda: all(pid_state(p) == "gone" for p in SPAWNED), 10, 0.05)
        rep.check("shutdown: every worker started here ended and was reaped", ended is not None,
                  f"{len(SPAWNED)} workers; " + str({p: pid_state(p) for p in SPAWNED if pid_state(p) != 'gone'}))
        left = wait_until(lambda: not sweep(), 15, 0.5)
        rep.check("shutdown: no worker process left (process sweep)", left is not None, str(sweep()))
        rep.data["workers_spawned"] = len(SPAWNED)
        try:
            phase_host_kill(rep, work, cache)
        except BaseException as e:  # noqa: BLE001
            rep.fail("host: unexpected error", e)
        rep.check("host process never imported numba", "numba" not in sys.modules)
    finally:
        for eng in engines:
            eng.shutdown()
        rep.data["total_s"] = round(time.monotonic() - t_start, 2)
        (out / "report.json").write_text(json.dumps({"ok": rep.ok, "checks": rep.checks, "data": rep.data},
                                                    indent=1, default=str), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)
    failed = [c["name"] for c in rep.checks if not c["ok"]]
    print(f"{len(rep.checks) - len(failed)}/{len(rep.checks)} checks passed in {rep.data['total_s']} s"
          + (f"; FAILED: {failed}" if failed else ""), flush=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
