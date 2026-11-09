"""Temporary CI diagnostic: why does a render worker not answer on this platform? Bisects the worker's pipe set-up
in small child processes, each with a deadline, then starts the real worker with faulthandler stack dumps."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FLAGS = ["-I", "-S", "-u"]

CASES = {
    "A plain echo": r"""
import sys
line = sys.stdin.buffer.readline()
sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
""",
    "B dup(1) + dup2(2, 1), write on the dup": r"""
import os, sys
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
p.write(line); p.flush()
""",
    "C reader thread on stdin, write from main": r"""
import os, sys, threading, time
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
def watch():
    sys.stdin.buffer.read(1)
threading.Thread(target=watch, daemon=True).start()
time.sleep(0.5)
p.write(line); p.flush()
""",
    "D reader thread blocked in readinto, write from main": r"""
import os, sys, threading, time
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
def watch():
    buf = bytearray(8)
    sys.stdin.buffer.readinto(memoryview(buf))
threading.Thread(target=watch, daemon=True).start()
time.sleep(0.5)
p.write(line); p.flush()
""",
    "E no dup2, reader thread, write on sys.stdout.buffer": r"""
import sys, threading, time
line = sys.stdin.buffer.readline()
def watch():
    sys.stdin.buffer.read(1)
threading.Thread(target=watch, daemon=True).start()
time.sleep(0.5)
sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
""",
    "F reader thread on a raw (unbuffered) stdin, write from main": r"""
import os, sys, threading, time
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
raw = os.fdopen(os.dup(0), "rb", buffering=0)
def watch():
    raw.read(1)
threading.Thread(target=watch, daemon=True).start()
time.sleep(0.5)
p.write(line); p.flush()
""",
    "G reader thread blocked on stdin, then import numpy": r"""
import json, os, sys, threading, time
sys.path[:0] = json.loads(sys.argv[1])
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
threading.Thread(target=lambda: sys.stdin.buffer.read(1), daemon=True).start()
time.sleep(0.5)
import numpy
p.write(line); p.flush()
""",
    "H import numpy first, then reader thread": r"""
import json, os, sys, threading, time
sys.path[:0] = json.loads(sys.argv[1])
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
import numpy
threading.Thread(target=lambda: sys.stdin.buffer.read(1), daemon=True).start()
time.sleep(0.5)
p.write(line); p.flush()
""",
    "I reader thread blocked on stdin, then import llvmlite.binding": r"""
import json, os, sys, threading, time
sys.path[:0] = json.loads(sys.argv[1])
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
threading.Thread(target=lambda: sys.stdin.buffer.read(1), daemon=True).start()
time.sleep(0.5)
try:
    import llvmlite.binding
except ImportError:
    pass
p.write(line); p.flush()
""",
    "J reader thread polls PeekNamedPipe (no pending read), then import numpy": r"""
import json, os, sys, threading, time
sys.path[:0] = json.loads(sys.argv[1])
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
def watch():
    if os.name == "nt":
        import _winapi, msvcrt
        h = msvcrt.get_osfhandle(0)
        while True:
            try:
                if _winapi.PeekNamedPipe(h, 0)[0] > 0:
                    break
            except OSError:
                break
            time.sleep(0.02)
    os.read(0, 1)
threading.Thread(target=watch, daemon=True).start()
time.sleep(0.5)
import numpy
p.write(line); p.flush()
""",
    "K reader thread blocked on stdin, then import ctypes and _decimal": r"""
import json, os, sys, threading, time
sys.path[:0] = json.loads(sys.argv[1])
line = sys.stdin.buffer.readline()
p = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
threading.Thread(target=lambda: sys.stdin.buffer.read(1), daemon=True).start()
time.sleep(0.5)
import ctypes, _decimal
p.write(line); p.flush()
""",
}


def run_case(name, code, exe):
    paths = [q for q in sys.path if q and os.path.isdir(q)]
    p = subprocess.Popen([exe, *FLAGS, "-c", code, json.dumps(paths)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    box = {}

    def go():
        p.stdin.write(b'{"hello": 1}\n')
        p.stdin.flush()
        box["out"] = p.stdout.readline()

    th = threading.Thread(target=go, daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(20)
    ok = not th.is_alive() and box.get("out") == b'{"hello": 1}\n'
    print(f"{'OK  ' if ok else 'HANG' if th.is_alive() else 'BAD '} {name} ({time.monotonic() - t0:.2f} s) "
          f"out={box.get('out')!r}", flush=True)
    p.kill()
    try:
        err = p.communicate(timeout=10)[1]
    except Exception as e:  # noqa: BLE001
        err = repr(e).encode()
    if err.strip():
        print("   stderr:", err.decode("utf-8", "replace")[-800:], flush=True)


def real_worker(exe):
    import chimera_vocal.scheduler as S
    import chimera_vocal.worker as W
    orig = W.BOOTSTRAP
    W.BOOTSTRAP = ("import faulthandler, sys\nfaulthandler.dump_traceback_later(8, repeat=True, file=sys.stderr)\n"
                   + orig)
    old = sys.executable
    sys.executable = exe
    try:
        sched = S.RenderScheduler(workers=2, cache_base=str(Path.cwd() / "probe-cache"), root=str(REPO))
        w = S._Worker(sched)
        box = {}

        def go():
            try:
                w.send({"req": "probe", "op": "selftest", "kernel": "numpy"})
                while True:
                    msg, _ = w.read()
                    if msg.get("req") == "probe":
                        box["msg"] = msg
                        return
            except BaseException as e:  # noqa: BLE001
                box["exc"] = repr(e)

        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(30)
        print(f"real worker via {exe}: {'HANG' if th.is_alive() else 'reply'} {json.dumps(box)[:300]}", flush=True)
        time.sleep(1.0)
        print("worker stderr (last 40 lines):\n" + "\n".join(w.err_tail), flush=True)
        w.kill()
    finally:
        sys.executable = old
        W.BOOTSTRAP = orig


if __name__ == "__main__":
    base = getattr(sys, "_base_executable", None) or sys.executable
    print(json.dumps({"executable": sys.executable, "base": base, "platform": sys.platform}), flush=True)
    for exe in dict.fromkeys([sys.executable, base]):
        print(f"=== interpreter {exe}", flush=True)
        for name, code in CASES.items():
            run_case(name, code, exe)
    real_worker(sys.executable)
