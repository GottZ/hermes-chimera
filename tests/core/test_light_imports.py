"""Importing the package and the numpy-free core modules never imports numpy/numba."""

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
import chimera_vocal
import chimera_vocal.core.nbmath, chimera_vocal.core.jsmath, chimera_vocal.core.keys, chimera_vocal.core.physics
import chimera_vocal.core.body, chimera_vocal.core.types, chimera_vocal.core.jit
from chimera_vocal.core.types import Seeds, SourceParams
from chimera_vocal.core.body import default_f0
assert Seeds.for_voice(1) != Seeds() and default_f0("female") == 205.0
assert not any(m.split(".")[0] in ("numpy", "numba") for m in sys.modules), sorted(sys.modules)
print("ok")
"""


def test_core_imports_without_numpy():
    r = subprocess.run([sys.executable, "-I", "-c", CODE, str(REPO_ROOT)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
