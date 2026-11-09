"""Render worker process - the only place numba is ever imported (via dsp.backend).

Started by the scheduler as ``python -I -S -u -c BOOTSTRAP`` with stdin/stdout pipes. First stdin line: JSON
handshake ``{"proto": 2, "sys_path", "site_dirs", "root", "cache_base"}``. Afterwards frames in both directions:
``b"CHW2" + uint32_le(len) + utf8-JSON``, a ``chunk`` reply followed by a raw frame of float32 LE samples. The
worker's first frame is its hello ``{"hello": 2, "pid"}``: the pid of the interpreter that renders, which behind a
Windows venv launcher is a child of the process the host started.
Ops: render, warm, selftest, quit. EOF on stdin ends the process at once, so a host that dies (or closes the pipe)
leaves no worker rendering on.

Test hook: with ``CHIMERA_TESTING=1``, ``CHIMERA_WORKER_VOICE_FACTORY=module:attr`` installs a voice factory
(compile.set_voice_factory); the worker then never loads a real kernel (the factory receives the kernel name).
"""

from __future__ import annotations

import json
import os
import queue
import struct
import sys
import threading
import time
import traceback
from typing import Any, BinaryIO

MAGIC = b"CHW2"
PROTO = 2

BOOTSTRAP = r"""
import json, os, sys
_line = bytearray()
while not _line.endswith(b"\n"):
    _c = os.read(0, 1)
    if not _c:
        os._exit(0)
    _line += _c
_hs = json.loads(_line.decode("utf-8"))
sys.dont_write_bytecode = bool(_hs.get("dont_write_bytecode", False))
sys.path[:] = [_hs["root"]] + list(_hs["sys_path"])
if _hs.get("site_dirs"):
    import site
    for _d in _hs["site_dirs"]:
        site.addsitedir(_d)
_proto = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
import chimera_vocal.worker as _w
_w.HANDSHAKE = _hs
_w.serve(sys.stdin.buffer.raw if os.name == "nt" else sys.stdin.buffer, _proto)
"""

HANDSHAKE: dict = {}


class ProtocolError(Exception):
    """Malformed frame (bad magic, truncated length or payload)."""


def write_frame(f: BinaryIO, msg: dict, raw: bytes | memoryview | None = None) -> None:
    """One JSON frame, optionally followed by one raw frame; flushes."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    f.write(MAGIC + struct.pack("<I", len(body)) + body)
    if raw is not None:
        mv = memoryview(raw).cast("B")
        f.write(MAGIC + struct.pack("<I", mv.nbytes))
        f.write(mv)
    f.flush()


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = f.readinto(view[got:]) if hasattr(f, "readinto") else None
        if k is None:
            b = f.read(n - got)
            k = len(b)
            view[got:got + k] = b
        if not k:
            break
        got += k
    return bytes(buf[:got]) if got < n else bytes(buf)


def _header(f: BinaryIO, first: bool) -> int:
    hdr = _read_exact(f, 8)
    if not hdr and first:
        raise EOFError("end of stream")
    if len(hdr) < 8:
        raise ProtocolError("truncated frame header")
    if hdr[:4] != MAGIC:
        raise ProtocolError(f"bad frame magic {hdr[:4]!r}")
    return struct.unpack("<I", hdr[4:])[0]


def read_frame(f: BinaryIO) -> "tuple[dict, bytes | None]":
    """Next message; a ``chunk`` reply (no "op") comes with its raw frame. EOF before a frame -> EOFError."""
    n = _header(f, True)
    body = _read_exact(f, n)
    if len(body) < n:
        raise ProtocolError("truncated frame payload")
    try:
        msg = json.loads(body.decode("utf-8"))
    except ValueError as e:
        raise ProtocolError(f"frame payload is not JSON: {e}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("frame payload is not an object")
    raw = None
    if "chunk" in msg and "op" not in msg:     # a reply piece (requests carry "op"; render requests a "chunk" size)
        nb = _header(f, False)
        raw = _read_exact(f, nb)
        if len(raw) < nb:
            raise ProtocolError("truncated raw frame")
    return msg, raw


class _State:
    def __init__(self, fake: bool) -> None:
        self.fake = fake
        self.numba: Any = None

    def kernels(self, name: str) -> tuple[Any, Any]:
        if self.fake:
            return name, None
        if name == "numba":
            if self.numba is None:
                from .dsp.backend import load_numba
                self.numba = load_numba(HANDSHAKE.get("cache_base"))
            return self.numba
        return None, None      # HumanVoice defaults: KERNEL_NUMPY, GEOMETRY_VEC


def _jsonable(x: Any) -> Any:
    import dataclasses
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _render(state: _State, out: BinaryIO, msg: dict) -> None:
    import numpy as np
    from .music.compile import RenderJob, render_job
    req = msg.get("req")
    job = RenderJob.from_json(msg["job"])
    kname = msg.get("kernel", "numpy")
    chunk = max(1, int(msg.get("chunk", 48000)))
    kernel, geometry = state.kernels(kname)
    buf = np.zeros(job.n_out, np.float32)
    pos = {"sent": 0, "i": 0}
    t0 = time.perf_counter()

    def progress(p: int) -> None:
        a = pos["sent"]
        piece = np.ascontiguousarray(buf[a:p], dtype="<f4")
        write_frame(out, {"req": req, "chunk": pos["i"], "offset": a, "n": p - a}, piece.data)
        pos["sent"], pos["i"] = p, pos["i"] + 1

    render_job(job, kernel, geometry, out=buf, chunk=chunk, progress=progress)
    write_frame(out, {"req": req, "done": True, "seconds": time.perf_counter() - t0, "pid": os.getpid()})


def _handle(state: _State, out: BinaryIO, msg: dict) -> None:
    op = msg.get("op")
    req = msg.get("req")
    if op == "render":
        _render(state, out, msg)
    elif op == "warm":
        if state.fake:
            write_frame(out, {"req": req, "ok": True, "warm": {"fake": True}})
            return
        from .dsp.backend import warm_numba
        write_frame(out, {"req": req, "ok": True, "warm": _jsonable(warm_numba(HANDSHAKE.get("cache_base")))})
    elif op == "selftest":
        kname = msg.get("kernel", "numpy")
        if state.fake:
            write_frame(out, {"req": req, "ok": True, "selftest": {"backend": kname, "passed": True,
                                                                  "max_rel_err": 0.0, "seconds": 0.0,
                                                                  "detail": "fake voice"}})
            return
        from .dsp.backend import selftest
        kernel, geometry = state.kernels(kname)
        if kernel is None:
            from .control.geometry import GEOMETRY_VEC
            from .dsp.kernel_numpy import KERNEL_NUMPY
            kernel, geometry = KERNEL_NUMPY, GEOMETRY_VEC
        write_frame(out, {"req": req, "ok": True, "selftest": _jsonable(selftest(kernel, geometry))})
    else:
        raise ValueError(f"unknown op {op!r}")


WIN_POLL_S = 0.05


def _wait_for_input(f: BinaryIO) -> None:
    """Windows: return once the stdin pipe holds data or its writing end is closed, without a read pending on it.

    With a read pending on stdin in this thread, ``import numpy`` in the main thread never returns on Windows (seen
    with Python 3.11 and 3.14; importing numpy first, or polling like this instead of reading, both work). Windows
    serialises calls on a synchronous pipe handle, so the import presumably waits on stdin inside a DLL's start-up.
    The handshake is read without read-ahead and ``f`` is the unbuffered stdin, so no request can sit in a buffer
    the pipe does not show."""
    import _winapi
    import msvcrt
    h = msvcrt.get_osfhandle(f.fileno())
    while True:
        try:
            if _winapi.PeekNamedPipe(h, 0)[0]:
                return
        except OSError:            # writing end closed: the read that follows sees the end of the stream
            return
        time.sleep(WIN_POLL_S)


def serve(inp: BinaryIO, out: BinaryIO) -> None:
    """Worker main loop: hello frame, EOF watcher thread, request queue, ops render/warm/selftest/quit."""
    try:
        write_frame(out, {"hello": PROTO, "pid": os.getpid()})
    except OSError:
        os._exit(0)
    state = _State(fake=_install_test_factory() is not None)
    q: "queue.Queue[dict]" = queue.Queue()

    def watch() -> None:
        try:
            while True:
                if os.name == "nt":
                    _wait_for_input(inp)
                msg, _ = read_frame(inp)
                q.put(msg)
        except BaseException:
            pass
        os._exit(0)        # parent gone or stdin closed: end now, even in the middle of a render

    threading.Thread(target=watch, name="chimera-eof-watch", daemon=True).start()
    while True:
        msg = q.get()
        if msg.get("op") == "quit":
            out.flush()
            # no normal interpreter shutdown: the EOF-watch daemon thread holds the stdin reader lock, and CPython
            # aborts (SIGABRT, "could not acquire lock … at interpreter shutdown") when it finalises that object
            os._exit(0)
        try:
            _handle(state, out, msg)
        except Exception as e:  # noqa: BLE001 - every failure becomes an error reply; the worker lives on
            try:
                write_frame(out, {"req": msg.get("req"), "ok": False, "error": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()})
            except OSError:
                os._exit(0)


def main() -> None:   # pragma: no cover - entered from BOOTSTRAP
    serve(sys.stdin.buffer, sys.stdout.buffer)


def _install_test_factory() -> Any:
    """With CHIMERA_TESTING=1: ``CHIMERA_WORKER_VOICE_FACTORY=module:attr`` -> compile.set_voice_factory(attr)."""
    spec = os.environ.get("CHIMERA_WORKER_VOICE_FACTORY")
    if not spec or os.environ.get("CHIMERA_TESTING") != "1":
        return None
    import importlib
    from .music.compile import set_voice_factory
    mod, _, attr = spec.partition(":")
    f = getattr(importlib.import_module(mod), attr)
    set_voice_factory(f)
    return f
