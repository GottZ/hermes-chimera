"""WAV -> MP3 via an optional ffmpeg and the MEDIA line (INTERFACES §11.3; P1-21).

S1: signatures only.
"""

from __future__ import annotations

import os

CREATE_NO_WINDOW = 0x08000000  # subprocess.CREATE_NO_WINDOW exists only on Windows


def ffmpeg_path() -> str | None:
    """``shutil.which("ffmpeg")`` (None on any failure)."""
    raise NotImplementedError


def media_line(path: "str | os.PathLike") -> str:
    """``"MEDIA:" + path``."""
    raise NotImplementedError


def to_mp3(wav: str, mp3: str, *, seconds: float) -> bool:
    """ffmpeg -hide_banner -loglevel error -y -i wav -codec:a libmp3lame -q:a 2 mp3; stdin=DEVNULL,
    capture_output, timeout=max(60, 2·seconds), creationflags=CREATE_NO_WINDOW on Windows; any failure -> False."""
    raise NotImplementedError
