"""Shared test helpers (C0 owns this file; every cluster may import it as ``support``).

Paths to the in-repo references, cached JSON loaders, the lab-reference location and the tolerance
helpers of TESTPLAN §1 (T1n, T2, T3) with the template's own comparison semantics
(crates/chimera/tests/common/mod.rs ``rel``/``worst``/``compare_audio``). numpy is imported lazily.
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = TESTS_DIR / "golden"
VECTORS_DIR = TESTS_DIR / "vectors"
PACKAGE_DIR = REPO_ROOT / "chimera_vocal"


@functools.lru_cache(maxsize=None)
def load_json(path: str | os.PathLike) -> Any:
    """Parsed JSON, cached per path. Treat the result as read-only (shared between tests)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def golden(name: str) -> Any:
    """``tests/golden/<name>.json`` (name without extension)."""
    return load_json(GOLDEN_DIR / f"{name}.json")


def vector(name: str) -> Any:
    """``tests/vectors/<name>.json`` (name without extension)."""
    return load_json(VECTORS_DIR / f"{name}.json")


def lab_refs() -> Path | None:
    """Directory of the full-length lab references (env CHIMERA_LAB_REFS), or None."""
    p = os.environ.get("CHIMERA_LAB_REFS")
    if p and Path(p).is_dir():
        return Path(p)
    return None


def f32(values: Sequence[float]):
    """Text samples (shortest float32 repr) -> numpy float32 array, bit-exact."""
    import numpy as np

    return np.asarray(values, dtype=np.float64).astype(np.float32)


# --- tolerances (TESTPLAN §1) -------------------------------------------------------------------------

def rel(a: float, b: float, floor: float) -> float:
    """Relative difference with an absolute floor; NaN == NaN (common/mod.rs ``rel``).

    Stricter than the template where it is blind: one NaN, or inf against a finite value, gives +inf (the Rust
    helper yields NaN there, which ``f64::max`` and Python ``max`` silently drop -- found by the K-0.4 red proof).
    """
    if a == b or (a != a and b != b):
        return 0.0
    d = abs(a - b) / max(abs(a), abs(b), floor)
    return math.inf if d != d else d


def worst(a: Sequence[float], b: Sequence[float], floor: float, label: str = "") -> float:
    """Largest ``rel`` over two equal-length sequences (common/mod.rs ``worst``)."""
    a, b = list(a), list(b)
    assert len(a) == len(b), f"{label}: length {len(a)} vs {len(b)}"
    w = 0.0
    for x, y in zip(a, b):
        w = max(w, rel(float(x), float(y), floor))
    return w


def assert_t1n(actual: Sequence[float], expected: Sequence[float], label: str = "") -> None:
    """T1n: plan numbers <= 1e-12 relative, floor 1e-12."""
    w = worst(actual, expected, 1e-12, label)
    assert w <= 1e-12, f"{label}: worst relative difference {w:.3e} > 1e-12"


def assert_t2(actual: Sequence[float], expected: Sequence[float], label: str = "", *, tol: float = 1e-9,
              floor: float = 1e-6) -> float:
    """T2: <= 1e-9 relative with floor 1e-6 (golden_geometry.rs:47-63). Returns the worst value."""
    w = worst(actual, expected, floor, label)
    assert w < tol, f"{label}: worst relative difference {w:.3e} >= {tol:.0e}"
    return w


def assert_exact(actual: Sequence[Any], expected: Sequence[Any], label: str = "") -> None:
    """T1/T2i/T6: equal element by element (floats bit-equal, NaN == NaN)."""
    a, b = list(actual), list(expected)
    assert len(a) == len(b), f"{label}: length {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        same = (x == y) and not (isinstance(x, float) and x == 0.0 and math.copysign(1, x) != math.copysign(1, y))
        if isinstance(x, float) and x != x and isinstance(y, float) and y != y:
            same = True
        assert same, f"{label}: first difference at {i}: {x!r} != {y!r}"


@dataclass(frozen=True)
class AudioDiff:
    peak: float               # peak |reference| (>= 1e-30)
    max_err: float            # max |got - reference| / peak
    first_diverge: int | None  # first index with error > tol
    env_db: float             # largest |RMS ratio| in dB over windows above -60 dB of peak


def compare_audio(reference, got, tol: float = 1e-6, win: int = 480) -> AudioDiff:
    """Template semantics (common/mod.rs:96-120): lengths must match exactly."""
    import numpy as np

    r = np.asarray(reference, dtype=np.float32).astype(np.float64).ravel()
    g = np.asarray(got, dtype=np.float32).astype(np.float64).ravel()
    assert r.shape == g.shape, f"length {r.shape[0]} vs {g.shape[0]}"
    peak = max(float(np.max(np.abs(r))) if r.size else 0.0, 1e-30)
    err = np.abs(r - g) / peak
    max_err = float(err.max()) if err.size else 0.0
    over = np.flatnonzero(err > tol)
    first = int(over[0]) if over.size else None
    env = 0.0
    for i in range(0, r.size, win):
        ca, cb = r[i:i + win], g[i:i + win]
        ra = math.sqrt(float(np.mean(ca * ca)))
        rb = math.sqrt(float(np.mean(cb * cb)))
        if max(ra, rb) / peak < 1e-3:
            continue
        env = max(env, abs(20.0 * math.log10((rb + 1e-30) / (ra + 1e-30))))
    return AudioDiff(peak, max_err, first, env)


def assert_t3(reference, got, label: str = "") -> AudioDiff:
    """T3: max |delta| <= 1e-6 x peak over the whole compared range, length exact."""
    d = compare_audio(reference, got, 1e-6)
    assert d.first_diverge is None, (
        f"{label}: diverges from sample {d.first_diverge} (max error {d.max_err:.2e} of peak {d.peak:.4g})")
    return d


# --- JIT-file rules (INTERFACES §6.4a; used by K-0 and K-3.13e) ------------------------------------------

JIT_LINE = 'jit = globals().get("CHIMERA_JIT", lambda f: f)'


def check_jit_file(path: str | os.PathLike) -> list[str]:
    """Return rule violations of a JIT file (empty list = conforming).

    (1) after the module docstring only imports precede ``jit = globals().get("CHIMERA_JIT", lambda f: f)``;
    (2) imports limited to ``math``, ``numpy`` and ``_chimera_nbmath`` (with the ``..core.nbmath`` fallback in a
    try/except ImportError); (3) every top-level function is decorated with exactly ``@jit``; (4) no classes.
    """
    import ast

    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    errors: list[str] = []
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    allowed = {"math", "numpy", "_chimera_nbmath"}

    def check_import(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    errors.append(f"line {node.lineno}: import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            ok = node.level == 2 and node.module == "core" and [a.name for a in node.names] == ["nbmath"]
            ok = ok or (node.level == 0 and (node.module or "").split(".")[0] in allowed)
            if not ok:
                errors.append(f"line {node.lineno}: from-import {'.' * node.level}{node.module} not allowed")

    i = 0
    while i < len(body) and isinstance(body[i], (ast.Import, ast.ImportFrom, ast.Try)):
        node = body[i]
        if isinstance(node, ast.Try):
            handlers_ok = all(
                isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in node.handlers)
            if not handlers_ok:
                errors.append(f"line {node.lineno}: try block must only catch ImportError")
            for sub in node.body + [s for h in node.handlers for s in h.body]:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    check_import(sub)
                else:
                    errors.append(f"line {sub.lineno}: only imports allowed in the import try block")
        else:
            check_import(node)
        i += 1
    if i >= len(body) or ast.get_source_segment(src, body[i]) != JIT_LINE:
        errors.append(f"first statement after the imports must be: {JIT_LINE}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in body[:i]:
            if not any(node in getattr(t, "body", []) + [s for h in getattr(t, "handlers", []) for s in h.body]
                       for t in body[:i]):
                errors.append(f"line {node.lineno}: imports only at the top of a JIT file")
        if isinstance(node, ast.ClassDef):
            errors.append(f"line {node.lineno}: class {node.name} not allowed in a JIT file")
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decos = [d.id if isinstance(d, ast.Name) else ast.dump(d) for d in node.decorator_list]
            if decos != ["jit"]:
                errors.append(f"line {node.lineno}: function {node.name} must be decorated with exactly @jit")
    return errors


def audio_head(name: str) -> list[float]:
    """Samples of a V7 case in tests/vectors/audio-head.json by (sub)name, resolving ``head_ref``.

    The aa-back head is stored once, in chimera_vocal/dsp/selftest.json (key ``expected``).
    """
    cases = [c for c in vector("audio-head")["cases"] if name in c["name"]]
    assert len(cases) == 1, f"{name!r} matches {[c['name'] for c in cases]}"
    c = cases[0]
    if "head" in c:
        return c["head"]
    ref = c["head_ref"]
    return load_json(REPO_ROOT / ref["file"])[ref["key"]]
