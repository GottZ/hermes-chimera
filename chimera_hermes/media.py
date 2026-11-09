"""MP3 through an optional ffmpeg, and the MEDIA line that hands a file to Hermes.

ffmpeg runs the way Hermes runs it for its own speech files: no stdin, a time limit, no console window on Windows.
Any failure keeps the WAV file, which every Hermes surface can play.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000  # subprocess.CREATE_NO_WINDOW exists only on Windows


def ffmpeg_path() -> str | None:
    try:
        return shutil.which("ffmpeg")
    except Exception:
        return None


def media_line(path: "str | os.PathLike") -> str:
    """``MEDIA:<absolute path>``: Hermes delivers a file when the model's reply carries this line (outside code)."""
    return "MEDIA:" + os.fspath(path)


def to_mp3(wav: str, mp3: str, *, seconds: float) -> bool:
    """Encode ``wav`` to ``mp3`` (libmp3lame, VBR quality 2); time limit max(60 s, 2 × audio length).
    Returns False on any failure and leaves no partial MP3 behind."""
    exe = ffmpeg_path()
    if not exe:
        return False
    try:
        timeout = max(60.0, 2.0 * float(seconds))
    except (TypeError, ValueError):
        timeout = 60.0
    try:
        r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-q:a", "2", str(mp3)], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                           creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        ok = r.returncode == 0 and os.path.isfile(mp3) and os.path.getsize(mp3) > 0
        if not ok:
            logger.warning("chimera: ffmpeg could not encode MP3 (rc %s): %s", r.returncode,
                           (r.stderr or b"")[-300:].decode("utf-8", "replace").strip())
    except Exception as e:  # timeout, missing binary, permission, encoder missing in this ffmpeg build ...
        logger.warning("chimera: ffmpeg failed: %s", e)
        ok = False
    if not ok:
        try:
            if os.path.isfile(mp3):
                os.unlink(mp3)
        except OSError:
            pass
    return ok
