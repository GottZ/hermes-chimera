"""Shipped text files outside tests/ carry no invisible or bidi-control characters.

Hermes' plugin security scan (`hermes plugins validate`, tools/skills_guard.py INVISIBLE_CHARS) flags zero-width and
directional characters as `invisible_unicode`; found on merge in S3 (english.py held literal U+FEFF/U+2028 in a
string). Such characters must be written as escapes. Stricter than the scanner: every format character (Cf), line and
paragraph separator (Zl, Zp) and, in Python sources, every non-ASCII space (Zs) counts. tests/ is out of scope like
in the scanner, which only notes findings there (tools/plugin_guard.py): byte-identical template fixtures and
text-preparation inputs contain such characters on purpose.
"""

import subprocess
import unicodedata

from support import REPO_ROOT

TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".rs", ".patch", ".cfg")
SCANNER_SET = {"​", "‌", "‍", "⁠", "⁢", "⁣", "⁤", "﻿", "‪", "‫",
               "‬", "‭", "‮", "⁦", "⁧", "⁨", "⁩"}


def _tracked_text_files():
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    for name in out.decode("utf-8").split("\0"):
        if name.endswith(TEXT_SUFFIXES) and not name.startswith("tests/"):
            yield REPO_ROOT / name


def _bad(ch: str, python: bool) -> bool:
    if ch in SCANNER_SET:
        return True
    cat = unicodedata.category(ch)
    if cat in ("Cf", "Zl", "Zp"):
        return True
    return python and cat == "Zs" and ch != " "


def test_no_invisible_characters_in_tracked_text_files():
    found = []
    n = 0
    for path in _tracked_text_files():
        n += 1
        python = path.suffix == ".py"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                if ord(ch) > 127 and _bad(ch, python):
                    found.append(f"{path.relative_to(REPO_ROOT)}:{i}: U+{ord(ch):04X} {unicodedata.name(ch, '?')}")
    assert n > 30
    assert found == [], "\n".join(found[:50])
