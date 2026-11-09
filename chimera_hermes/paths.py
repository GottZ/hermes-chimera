"""Where chimera writes: rendered audio under ``<cache/audio>/chimera/`` of the active Hermes profile (resolved on
every call, accepted by Hermes' media delivery check even in strict mode), engine cache and temporary mixer files
under ``<HERMES_HOME>/cache/chimera/``. Nothing else is written."""

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
SLUG_MAX = 48


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


def audio_dir(create: bool = True) -> Path:
    """``<cache/audio>/chimera`` of the current profile (a populated legacy ``audio_cache`` wins, like in Hermes)."""
    from hermes_constants import get_hermes_dir

    d = Path(get_hermes_dir("cache/audio", "audio_cache")) / "chimera"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    return hermes_home() / "cache" / "chimera"


def numba_cache_dir() -> Path:
    return cache_dir() / "numba"


def tmp_dir() -> Path:
    return cache_dir() / "tmp"


def new_output_stem(raw: str | None, default: str, *, now: float | None = None) -> tuple[Path, str]:
    """(audio_dir, "<YYYYMMDD-HHMMSS>-<slug>[-n]"); ``<stem>.wav`` is reserved at once (created empty, exclusive) so
    two renders in the same second never share a file."""
    d = audio_dir()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + slug(raw, default)
    for n in range(1, 1000):
        stem = base if n == 1 else f"{base}-{n}"
        try:
            fd = os.open(d / f"{stem}.wav", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return d, stem
    raise FileExistsError(f"no free file name for {base!r} in {d}")


def cleanup(keep_days: float, *, now: float | None = None, keep_cache: Iterable[str] = ()) -> int:
    """Delete chimera's own rendered files older than ``keep_days`` (0 keeps everything), temporary mixer files older
    than a day and numba cache folders other than ``keep_cache`` older than ``keep_days``. Never raises; returns the
    number of entries removed."""
    now = time.time() if now is None else now
    removed = 0

    def old(p: Path, days: float) -> bool:
        try:
            return now - p.stat().st_mtime > days * 86400.0
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
        if keep > 0:
            d = audio_dir(create=False)
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file() and _OWN_FILE.match(p.name) and old(p, keep):
                        rm(p)
            nd = numba_cache_dir()
            if nd.is_dir():
                keep_names = set(keep_cache)
                for p in nd.iterdir():
                    if p.is_dir() and p.name not in keep_names and old(p, keep):
                        rm(p)
        td = tmp_dir()
        if td.is_dir():
            for p in td.iterdir():
                if old(p, 1.0):
                    rm(p)
    except Exception:
        pass
    return removed
