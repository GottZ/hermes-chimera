#!/usr/bin/env python3
"""Regenerate ``chimera_vocal/data/`` from a checkout of the template engine
(https://github.com/jamesondh/chimera-vocal-synth, MIT; data pinned at commit 092b16ce12b9).

    python tools/regen_data.py <template-checkout>            write the three files
    python tools/regen_data.py <template-checkout> --check    compare only; exit status 1 on a difference

Files (stored as plain text: the plugin ships no archives):
  cmudict.txt            the gunzipped ``public/data/cmudict.txt.gz`` (CMU Pronouncing Dictionary, BSD-style
                         license, see THIRD_PARTY_NOTICES.md), byte for byte
  vowel_targets.json     ``crates/chimera/src/phonetics/vowelTargets.generated.json``, byte for byte
  consonant_targets.json ``crates/chimera/src/phonetics/consonantTargets.generated.json``, byte for byte

A template whose data no longer matches the pinned size/sha256 is reported (the goldens in ``tests/`` were made
with the pinned data, so a newer dictionary is a deliberate update, not a silent one).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "chimera_vocal" / "data"

SOURCES = {
    "cmudict.txt": ("public/data/cmudict.txt.gz", True),
    "vowel_targets.json": ("crates/chimera/src/phonetics/vowelTargets.generated.json", False),
    "consonant_targets.json": ("crates/chimera/src/phonetics/consonantTargets.generated.json", False),
}
PINNED = {
    "cmudict.txt": (3330130, "cac51a2d86b7312240d4c16b52d9c606a6989d8e03ced24615b2b573510a01af"),
    "vowel_targets.json": (25987, "975ab3fec65772b6c232fea732808147390e0da9d6240dbffdea4b97a557235d"),
    "consonant_targets.json": (1907, "aac1d0868d0af07e39e482a05c1e27f0bb628d5640e54231aeede162d0a8f17d"),
}


def derive(template: Path, name: str) -> bytes:
    rel, gunzip = SOURCES[name]
    raw = (template / rel).read_bytes()
    return gzip.decompress(raw) if gunzip else raw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("template", type=Path, help="checkout of jamesondh/chimera-vocal-synth")
    ap.add_argument("--check", action="store_true", help="compare with chimera_vocal/data instead of writing")
    args = ap.parse_args(argv)
    status = 0
    for name in SOURCES:
        data = derive(args.template, name)
        sha = hashlib.sha256(data).hexdigest()
        pinned = PINNED[name] == (len(data), sha)
        target = DATA_DIR / name
        if args.check:
            same = target.is_file() and target.read_bytes() == data
            status |= 0 if same else 1
            state = "same" if same else "DIFFERS"
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            state = "written"
        print(f"{name}: {len(data)} bytes sha256 {sha} {state}{'' if pinned else ' (not the pinned template data)'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
