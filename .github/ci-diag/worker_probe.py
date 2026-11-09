"""Temporary CI diagnostic: can this platform start one render worker and get a selftest reply?

Tries the interpreter as Hermes would start it (sys.executable, a venv launcher on Windows) and the base interpreter
(sys._base_executable). Each attempt has a deadline; on a hang it prints the worker's stderr tail, the process tree
and every thread of this process, then kills the worker."""

import faulthandler
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def tree():
    if os.name == "nt":
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | "
              "ForEach-Object { \"$($_.ProcessId) parent=$($_.ParentProcessId) $($_.Name) "
              "$($_.CommandLine.Substring(0, [Math]::Min(90, $_.CommandLine.Length)))\" }")
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return subprocess.run(["ps", "-ax", "-o", "pid,ppid,stat,command"], capture_output=True, text=True).stdout


def attempt(label, exe, op):
    import chimera_vocal.scheduler as S
    print(f"--- {label}: {exe} op={op}", flush=True)
    old = sys.executable
    sys.executable = exe
    try:
        sched = S.RenderScheduler(workers=2, cache_base=str(Path.cwd() / "probe-cache"), root=str(REPO))
        box = {}
        t0 = time.monotonic()

        def run():
            try:
                w = S._Worker(sched)
                box["w"] = w
                print(f"spawned pid {w.pid} after {time.monotonic() - t0:.2f} s", flush=True)
                w.send({"req": "probe", "op": op, "kernel": "numpy"})
                while True:
                    msg, _ = w.read()
                    if msg.get("req") == "probe":
                        box["msg"] = msg
                        break
            except BaseException as e:  # noqa: BLE001
                box["exc"] = repr(e)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(90)
        w = box.get("w")
        if th.is_alive():
            print(f"HANG after {time.monotonic() - t0:.1f} s; worker alive={w.alive() if w else None}", flush=True)
            if w:
                print("worker stderr tail:", w.tail(), flush=True)
            print(tree(), flush=True)
            faulthandler.dump_traceback(all_threads=True)
        else:
            msg = box.get("msg") or {}
            print(f"reply after {time.monotonic() - t0:.2f} s: exc={box.get('exc')} "
                  f"msg={json.dumps(msg)[:300]}", flush=True)
            if w:
                print("worker stderr tail:", w.tail(), flush=True)
        if w:
            w.kill()
            print(f"killed; pid state poll={w.proc.poll()}", flush=True)
    finally:
        sys.executable = old


if __name__ == "__main__":
    print(json.dumps({"executable": sys.executable, "base": getattr(sys, "_base_executable", None),
                      "prefix": sys.prefix, "base_prefix": sys.base_prefix}), flush=True)
    for op in ("selftest",):
        attempt("as started", sys.executable, op)
        base = getattr(sys, "_base_executable", None)
        if base and os.path.normcase(base) != os.path.normcase(sys.executable):
            attempt("base interpreter", base, op)
    print(tree(), flush=True)
