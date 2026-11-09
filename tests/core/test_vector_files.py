"""Reference data files: present, parseable, inside the size budget, helpers resolve them."""

import json

from support import GOLDEN_DIR, PACKAGE_DIR, VECTORS_DIR, audio_head, load_json

BUDGET = 3_000_000  # bytes of JSON test data in the repository (READMEs not counted)


def _files():
    return sorted(p for d in (GOLDEN_DIR, VECTORS_DIR) for p in d.iterdir() if p.is_file()) + [
        PACKAGE_DIR / "dsp" / "selftest.json"]


def test_reference_files_parse_and_fit_budget():
    total = 0
    for p in _files():
        if p.suffix == ".json":
            total += p.stat().st_size
            json.loads(p.read_text(encoding="utf-8"))
    assert total <= BUDGET, total


def test_manifest_covers_every_vector_file():
    m = load_json(VECTORS_DIR / "manifest.json")
    listed = {f["path"] for f in m["repo_files"]}
    for p in _files():
        if p.name != "manifest.json":
            assert p.relative_to(PACKAGE_DIR.parent).as_posix() in listed, p      # manifest paths use "/"


def test_audio_heads_resolve():
    for name in ("aa-back", "a-front", "hello_world"):
        h = audio_head(name)
        assert len(h) == 4800
