"""tools/regen_data.py (K-1.9 generator): derives chimera_vocal/data/ byte for byte from a template checkout.

Runs against a fake checkout built from the shipped files (gzip of cmudict.txt, the two JSON tables), so no
template clone is needed: --check must report "same" for all three, and a changed byte must fail.
"""

import gzip
import shutil
import subprocess
import sys

from support import PACKAGE_DIR, REPO_ROOT

TOOL = REPO_ROOT / "tools" / "regen_data.py"
DATA = PACKAGE_DIR / "data"


def fake_template(root):
    (root / "public" / "data").mkdir(parents=True)
    ph = root / "crates" / "chimera" / "src" / "phonetics"
    ph.mkdir(parents=True)
    (root / "public" / "data" / "cmudict.txt.gz").write_bytes(
        gzip.compress((DATA / "cmudict.txt").read_bytes(), compresslevel=1))
    shutil.copyfile(DATA / "vowel_targets.json", ph / "vowelTargets.generated.json")
    shutil.copyfile(DATA / "consonant_targets.json", ph / "consonantTargets.generated.json")
    return ph


def run(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)], capture_output=True, text=True,
                          timeout=120)


def test_regen_check_matches_shipped_data(tmp_path):
    fake_template(tmp_path)
    r = run(tmp_path, "--check")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3 and all(ln.endswith(" same") for ln in lines), r.stdout


def test_regen_check_detects_a_changed_table(tmp_path):
    ph = fake_template(tmp_path)
    p = ph / "consonantTargets.generated.json"
    p.write_bytes(p.read_bytes().replace(b"9.1", b"9.2", 1))
    r = run(tmp_path, "--check")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "consonant_targets.json" in r.stdout and "DIFFERS" in r.stdout and "not the pinned" in r.stdout
