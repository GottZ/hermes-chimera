"""Text and planning are pure Python: importing them never imports numpy/numba (the package must load without numpy)."""

import subprocess
import sys

from support import REPO_ROOT

CODE = r"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "numba"):
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import chimera_vocal.text.english, chimera_vocal.text.ipa, chimera_vocal.text.syllables, chimera_vocal.text.textprep
import chimera_vocal.plan.durations, chimera_vocal.plan.inventory, chimera_vocal.plan.planner, chimera_vocal.plan.sing
assert not any(m.split(".")[0] in ("numpy", "numba") for m in sys.modules), sorted(sys.modules)
print("ok")
"""


def test_text_and_plan_import_without_numpy():
    r = subprocess.run([sys.executable, "-I", "-c", CODE, str(REPO_ROOT)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
