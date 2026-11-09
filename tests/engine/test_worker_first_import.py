"""A fresh render worker answers its first request, which is where it first imports numpy.

The worker reads requests in a watch thread that is already waiting on stdin when the main thread handles the
first request and imports numpy (and later numba). On Windows a read pending on a synchronous pipe blocks every
other call on that handle, and loading a DLL that brings its own C runtime queries the standard handles; with a
blocking read on stdin the worker hung inside ``import numpy`` and never answered (GitHub Actions, windows-latest,
run 42672961183: worker_probe "HANG", run 42676376139: probe "no answer"). Checked here with a deadline instead of
a hang. Red proof: a watch thread that blocks in a read on stdin while the main thread imports (Windows only).
"""

import threading

from chimera_vocal.worker import read_frame, write_frame
from support.fakes import spawn_worker

DEADLINE_S = 120.0


def test_first_request_imports_numpy_and_answers(tmp_path):
    p = spawn_worker(tmp_path / "stderr.txt", env={"CHIMERA_TESTING": "0"})
    box = {}

    def ask() -> None:
        try:
            write_frame(p.stdin, {"req": "first", "op": "selftest", "kernel": "numpy"})
            while True:
                msg, _ = read_frame(p.stdout)
                if msg.get("req") == "first":
                    box["msg"] = msg
                    return
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e

    th = threading.Thread(target=ask, daemon=True)
    th.start()
    th.join(DEADLINE_S)
    try:
        assert not th.is_alive(), (f"no answer within {DEADLINE_S:.0f} s; worker stderr: "
                                    f"{(tmp_path / 'stderr.txt').read_text(errors='replace')[-3000:]}")
        assert "exc" not in box, repr(box.get("exc"))
        msg = box["msg"]
        assert msg.get("ok") is True and msg["selftest"]["passed"] is True, msg
    finally:
        p.kill()
        p.wait(timeout=30)
        for s in (p.stdin, p.stdout):
            try:
                s.close()
            except OSError:
                pass
