#!/usr/bin/env python3
"""Set up a work directory for the vector exporters (outside the repository).

    python3 -I prepare.py --upstream <checkout of jamesondh/chimera-vocal-synth at 092b16ce12b9> --work <new dir>

The work directory receives a copy of this directory, `upstream` (a symlink to the template checkout, or a copy
without .git/target where symlinks are not allowed) and `crate-chimera` (the template's crates/chimera Cargo.toml
and src plus its LICENSE, with crate-chimera.patch applied). The patch is applied by the small, strict unified-diff
reader below (exact context, no fuzz), so neither `patch` nor `git apply` is needed. Standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE_COMMIT = "092b16ce12b927d473c884ca2dfae53ce1b995df"
HERE = Path(__file__).resolve().parent
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_patch(patch: str, root: Path) -> list[str]:
    """Apply a unified diff (paths with one leading component, as `git diff --no-prefix a b`) below root."""
    lines = patch.splitlines(keepends=True)
    touched = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("--- "):
            i += 1
            continue
        old = lines[i][4:].rstrip("\r\n").split("\t")[0]
        if not lines[i + 1].startswith("+++ "):
            raise ValueError(f"line {i + 2}: '+++' expected")
        new = lines[i + 1][4:].rstrip("\r\n").split("\t")[0]
        i += 2
        rel = new.split("/", 1)[1]
        path = root / rel
        if old == "/dev/null":
            src: list[str] = []
            eol = "\n"
        else:
            src = path.read_text(encoding="utf-8", newline="").splitlines(keepends=True)
            eol = "\r\n" if src and src[0].endswith("\r\n") else "\n"
        out: list[str] = []
        pos = 0
        while i < len(lines) and lines[i].startswith("@@"):
            m = HUNK.match(lines[i])
            if not m:
                raise ValueError(f"line {i + 1}: bad hunk header")
            ostart, olen = int(m[1]), int(m[2]) if m[2] is not None else 1
            start = ostart - 1 if olen > 0 else ostart
            if start < pos:
                raise ValueError(f"{rel}: overlapping hunks")
            out.extend(src[pos:start])
            pos = start
            i += 1
            while i < len(lines) and lines[i][:1] in (" ", "-", "+", "\\"):
                tag, body = lines[i][0], lines[i][1:]
                if tag == "\\":
                    raise ValueError(f"{rel}: 'No newline at end of file' is not supported")
                if tag in (" ", "-"):
                    if pos >= len(src) or src[pos].rstrip("\r\n") != body.rstrip("\r\n"):
                        raise ValueError(f"{rel}: context mismatch at line {pos + 1}")
                    if tag == " ":
                        out.append(src[pos])
                    pos += 1
                else:
                    out.append(body.rstrip("\r\n") + eol)
                i += 1
        out.extend(src[pos:])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(out), encoding="utf-8", newline="")
        touched.append(rel)
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, type=Path, help="template checkout at " + TEMPLATE_COMMIT)
    ap.add_argument("--work", required=True, type=Path, help="new work directory (outside the repository)")
    a = ap.parse_args()
    up, work = a.upstream.resolve(), a.work.resolve()
    if not (up / "crates" / "chimera" / "Cargo.toml").is_file():
        print(f"{up}: not a chimera-vocal-synth checkout", file=sys.stderr)
        return 2
    try:
        head = subprocess.run(["git", "-C", str(up), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if head != TEMPLATE_COMMIT:
            print(f"warning: template HEAD is {head}, the vectors were exported from {TEMPLATE_COMMIT}", file=sys.stderr)
    except (OSError, subprocess.CalledProcessError):
        print("warning: could not read the template commit (git unavailable or not a checkout)", file=sys.stderr)
    if work.exists() and any(work.iterdir()):
        print(f"{work}: exists and is not empty", file=sys.stderr)
        return 2
    shutil.copytree(HERE, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "target", "out"))
    link = work / "upstream"
    try:
        os.symlink(up, link, target_is_directory=True)
    except OSError:
        shutil.copytree(up, link, ignore=shutil.ignore_patterns(".git", "target", "node_modules"))
    crate = work / "crate-chimera"
    crate.mkdir()
    shutil.copy2(up / "crates" / "chimera" / "Cargo.toml", crate / "Cargo.toml")
    shutil.copytree(up / "crates" / "chimera" / "src", crate / "src")
    shutil.copy2(up / "LICENSE", crate / "LICENSE")
    touched = apply_patch((HERE / "crate-chimera.patch").read_text(encoding="utf-8"), crate)
    print(f"work directory {work}: crate-chimera patched ({len(touched)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
