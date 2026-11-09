"""K-1.9 — the shipped data files and the dictionary loader.

``chimera_vocal/data/cmudict.txt`` is byte-identical to the template's gunzipped ``public/data/cmudict.txt.gz``
(3 330 130 B, 126 052 entries, sha256 cac51a2d...01af); the two calibration tables are byte-identical to
``crates/chimera/src/phonetics/{vowel,consonant}Targets.generated.json``. ``EnglishG2P.default()`` loads the
dictionary lazily, once per process, thread-safe, in under 1 s.

Red proof: ship the gzip file instead of the text.
"""

import hashlib
import subprocess
import sys

from support import PACKAGE_DIR, REPO_ROOT

DATA = PACKAGE_DIR / "data"
FILES = {
    "cmudict.txt": (3330130, "cac51a2d86b7312240d4c16b52d9c606a6989d8e03ced24615b2b573510a01af"),
    "vowel_targets.json": (25987, "975ab3fec65772b6c232fea732808147390e0da9d6240dbffdea4b97a557235d"),
    "consonant_targets.json": (1907, "aac1d0868d0af07e39e482a05c1e27f0bb628d5640e54231aeede162d0a8f17d"),
}


def _check(name: str) -> bytes:
    p = DATA / name
    size, sha = FILES[name]
    assert p.is_file(), f"missing {p}"
    raw = p.read_bytes()
    assert len(raw) == size, f"{name}: {len(raw)} bytes, expected {size}"
    assert hashlib.sha256(raw).hexdigest() == sha, f"{name}: sha256 differs"
    return raw


def test_dictionary_file():
    raw = _check("cmudict.txt")
    text = raw.decode("utf-8")
    lines = text.split("\n")
    entries = [ln[: ln.find("\t")] for ln in lines if ln.find("\t") > 0]
    assert len(lines) == len(entries) == 126052
    assert len(set(entries)) == 126052
    assert not text.endswith("\n")


def test_calibration_tables():
    import json

    for name in ("vowel_targets.json", "consonant_targets.json"):
        data = json.loads(_check(name))
        assert set(data) == {"male", "female", "child"}, name
    vt = json.loads((DATA / "vowel_targets.json").read_bytes())
    assert all(len(vt[b]) == 22 for b in vt)


LOAD = r"""
import sys, threading, time
sys.path.insert(0, sys.argv[1])
t = time.perf_counter()
from chimera_vocal.text.english import EnglishG2P
got = []
def first_use():
    got.append(EnglishG2P.default())
threads = [threading.Thread(target=first_use) for _ in range(8)]
for th in threads: th.start()
for th in threads: th.join()
dt = time.perf_counter() - t
assert len(got) == 8 and all(g is got[0] for g in got), "default() must return one shared instance"
assert EnglishG2P.default() is got[0]
assert [s.ipa for s in got[0].word("hello", False)] == ["h", "ə", "l", "oʊ"]
print(f"{dt:.3f}")
"""


def test_default_loads_once_thread_safe_under_one_second():
    r = subprocess.run([sys.executable, "-I", "-c", LOAD, str(REPO_ROOT)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert float(r.stdout.strip()) < 1.0, f"first load took {r.stdout.strip()} s"
