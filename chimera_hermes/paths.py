"""Where chimera writes: rendered audio under ``<cache/audio>/chimera/`` of the active Hermes profile (resolved on
every call, accepted by Hermes' media delivery check even in strict mode), the numba cache and temporary render files
under ``<HERMES_HOME>/cache/chimera/``. Nothing else is written. Folders are created private (0700) and rendered files
are private (0600) like Hermes' own media cache: rendered speech contains what the user had read aloud."""

from __future__ import annotations

import os
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Iterable

_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                           *(f"lpt{i}" for i in range(1, 10))})
_OWN_FILE = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]{1,60}\.(wav|mp3)$")
_LEFTOVER = re.compile(r"^\.chimera-tmp-[0-9a-z_]+(\.f32)?$|^\d{8}-\d{6}-[a-z0-9-]{1,60}\.wav\.part-[0-9a-f]+$")
SLUG_MAX = 48
TMP_KEEP_DAYS = 1.0


def slug(raw: str | None, default: str) -> str:
    """File stem from a title: ASCII lower-case letters, digits and single hyphens, at most 48 characters. Empty
    results and Windows device names give ``default``; the stem never contains CR, LF or spaces, so the MEDIA line
    built from it cannot be split or extended."""
    if not isinstance(raw, str):
        return default
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:SLUG_MAX].strip("-")
    if not s or s in _DEVICE_NAMES:
        return default
    return s


def hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home())


def _mkdir_private(d: Path) -> None:
    """Create ``d`` and every missing parent with mode 0700 (``mkdir(parents=True)`` would give the parents the
    default mode); existing folders are left as they are."""
    missing = []
    p = d
    while not p.exists() and p != p.parent:
        missing.append(p)
        p = p.parent
    for m in reversed(missing):
        try:
            m.mkdir(mode=0o700)
        except FileExistsError:
            pass


def private_file(path: "str | os.PathLike") -> None:
    """Best effort: make a rendered file readable by its owner only (no-op where chmod has no such meaning)."""
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def audio_dir(create: bool = True) -> Path:
    """``<cache/audio>/chimera`` of the current profile (a populated legacy ``audio_cache`` wins, like in Hermes)."""
    from hermes_constants import get_hermes_dir

    d = Path(get_hermes_dir("cache/audio", "audio_cache")) / "chimera"
    if create:
        _mkdir_private(d)
    return d


def cache_dir() -> Path:
    return hermes_home() / "cache" / "chimera"


def numba_cache_dir() -> Path:
    return cache_dir() / "numba"


def tmp_dir(create: bool = False) -> Path:
    """``<HERMES_HOME>/cache/chimera/tmp``: temporary files of running renders (voice tracks, mix sum, speech
    windows); whatever an interrupted render leaves there is removed after a day."""
    d = cache_dir() / "tmp"
    if create:
        _mkdir_private(d)
    return d


def new_output_stem(raw: str | None, default: str, *, now: float | None = None) -> tuple[Path, str]:
    """(audio_dir, "<YYYYMMDD-HHMMSS>-<slug>[-n]"); ``<stem>.wav`` is reserved at once (created empty, exclusive) so
    two renders in the same second never share a file."""
    d = audio_dir()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + slug(raw, default)
    for n in range(1, 1000):
        stem = base if n == 1 else f"{base}-{n}"
        try:
            fd = os.open(d / f"{stem}.wav", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return d, stem
    raise FileExistsError(f"no free file name for {base!r} in {d}")


def cleanup(keep_days: float, *, now: float | None = None, keep_cache: Iterable[str] | None = None) -> int:
    """Housekeeping before each render; never raises; returns the number of entries removed.

    - chimera's own rendered files in the audio folder older than ``keep_days`` (0 keeps them);
    - temporary files older than a day: everything in ``tmp_dir()`` and leftovers of interrupted renders in the
      audio folder (``.chimera-tmp-*`` of older versions, ``*.wav.part-*`` of the mixer);
    - numba cache folders older than ``keep_days`` other than those named in ``keep_cache``. With ``keep_cache``
      None the numba cache is left alone: numba does not touch the folder in use on a cache hit, so its age says
      nothing, and only the caller knows which folder is in use.
    """
    now = time.time() if now is None else now
    removed = 0

    def old(p: Path, days: float) -> bool:
        try:
            return now - p.lstat().st_mtime > days * 86400.0
        except OSError:
            return False

    def rm(p: Path) -> None:
        nonlocal removed
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except OSError:
            pass

    try:
        keep = float(keep_days)
    except (TypeError, ValueError):
        keep = 7.0
    try:
        d = audio_dir(create=False)
        if d.is_dir():
            for p in d.iterdir():
                if _OWN_FILE.match(p.name) and p.is_file() and keep > 0 and old(p, keep):
                    rm(p)
                elif _LEFTOVER.match(p.name) and old(p, TMP_KEEP_DAYS):
                    rm(p)
        nd = numba_cache_dir()
        if keep > 0 and keep_cache is not None and nd.is_dir():
            keep_names = set(keep_cache)
            for p in nd.iterdir():
                if p.is_dir() and p.name not in keep_names and old(p, keep):
                    rm(p)
        td = tmp_dir()
        if td.is_dir():
            for p in td.iterdir():
                if old(p, TMP_KEEP_DAYS):
                    rm(p)
    except Exception:
        pass
    return removed
