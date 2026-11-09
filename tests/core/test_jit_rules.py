"""JIT-file rules and the core JIT manifest (INTERFACES §4, §6.4a; the AST check is shared with K-3.13e)."""

from chimera_vocal.core.jit import CORE_JIT_FILES, JIT_PLACEHOLDER_LINE, JitFile
from support import JIT_LINE, PACKAGE_DIR, check_jit_file


def test_core_manifest():
    assert CORE_JIT_FILES == (JitFile("core/nbmath.py", "_chimera_nbmath", ()),)
    assert JIT_PLACEHOLDER_LINE == JIT_LINE
    for f in CORE_JIT_FILES:
        assert (PACKAGE_DIR / f.relpath).is_file()


def test_nbmath_follows_jit_rules():
    assert check_jit_file(PACKAGE_DIR / "core" / "nbmath.py") == []


def test_checker_catches_violations(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('"""doc"""\nimport math\nfrom chimera_vocal.core import types\n'
                   'jit = globals().get("CHIMERA_JIT", lambda f: f)\n\n'
                   'class X:\n    pass\n\ndef f(x):\n    return x\n', encoding="utf-8")
    errs = check_jit_file(bad)
    assert any("from-import" in e for e in errs)
    assert any("class X" in e for e in errs)
    assert any("function f" in e for e in errs)
    good = tmp_path / "good.py"
    good.write_text('import math\ntry:\n    import _chimera_nbmath as nbm\nexcept ImportError:\n'
                    '    from ..core import nbmath as nbm\n'
                    'jit = globals().get("CHIMERA_JIT", lambda f: f)\n\n@jit\ndef f(x):\n    return nbm.clamp(x, 0.0, 1.0)\n',
                    encoding="utf-8")
    assert check_jit_file(good) == []
