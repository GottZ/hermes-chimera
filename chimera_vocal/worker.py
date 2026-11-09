"""Render worker process (INTERFACES §9.3) - the only place numba is ever imported (via dsp.backend).

Started by the scheduler as ``python -I -S -u -c BOOTSTRAP`` with stdin/stdout pipes. First stdin line: JSON
handshake ``{"proto": 2, "sys_path", "site_dirs", "root", "cache_base"}``. Afterwards frames in both directions:
``b"CHW2" + uint32_le(len) + utf8-JSON``, a ``chunk`` reply followed by a raw frame of float32 LE samples.
Ops: render, warm, selftest, quit. EOF on stdin ends the process at once (P1-08).
"""

from __future__ import annotations

from typing import Any, BinaryIO

MAGIC = b"CHW2"
PROTO = 2

BOOTSTRAP = r"""
raise NotImplementedError
"""


class ProtocolError(Exception):
    """Malformed frame (bad magic, truncated length or payload)."""


def write_frame(f: BinaryIO, msg: dict, raw: bytes | memoryview | None = None) -> None:
    """One JSON frame, optionally followed by one raw frame; flushes."""
    raise NotImplementedError


def read_frame(f: BinaryIO) -> "tuple[dict, bytes | None]":
    """Next message; a ``chunk`` message comes with its raw frame. EOF before a frame -> EOFError."""
    raise NotImplementedError


def serve(inp: BinaryIO, out: BinaryIO) -> None:
    """Worker main loop: EOF watcher thread, request queue, ops render/warm/selftest/quit."""
    raise NotImplementedError


def main() -> None:   # pragma: no cover - entered from BOOTSTRAP
    raise NotImplementedError


def _install_test_factory() -> Any:
    """``CHIMERA_WORKER_VOICE_FACTORY=module:attr`` -> compile.set_voice_factory(attr) (tests; fake kernels)."""
    raise NotImplementedError
