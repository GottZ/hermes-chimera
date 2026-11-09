"""Text tests: shared helpers as fixtures (test folders have no __init__.py; pytest runs with --import-mode=importlib).

``json_of`` turns typed results into the template's serde JSON; ``t1_diff`` is the T1 comparison of
golden_phonetics_text.rs ``same``: deep equality, numbers compared as f64 (``1`` equals ``1.0``),
booleans never equal numbers.
"""

from __future__ import annotations

import pytest


def json_of(x):
    """Typed value -> JSON value (objects via ``to_json``, enums via ``value``)."""
    if hasattr(x, "to_json"):
        return x.to_json()
    if isinstance(x, (list, tuple)):
        return [json_of(v) for v in x]
    if isinstance(x, dict):
        return {k: json_of(v) for k, v in x.items()}
    if hasattr(x, "value") and type(x).__module__ != "builtins":
        return x.value
    return x


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def t1_diff(expected, got, path: str = "") -> str | None:
    """First difference between two JSON values (None = same), golden_phonetics_text.rs ``same``."""
    if _is_num(expected) and _is_num(got):
        return None if float(expected) == float(got) else f"{path}: {expected!r} != {got!r}"
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return f"{path}: length {len(expected)} != {len(got)}"
        for i, (a, b) in enumerate(zip(expected, got)):
            d = t1_diff(a, b, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(expected, dict) and isinstance(got, dict):
        if set(expected) != set(got):
            return f"{path}: keys {sorted(expected)} != {sorted(got)}"
        for k in expected:
            d = t1_diff(expected[k], got[k], f"{path}.{k}")
            if d:
                return d
        return None
    if type(expected) is type(got) and expected == got:
        return None
    return f"{path}: {expected!r} != {got!r}"


class Mismatches:
    """Collects differences of one area and fails once, like the template's ``report``."""

    def __init__(self, area: str) -> None:
        self.area, self.n, self.items = area, 0, []

    def check(self, label: str, expected, got) -> None:
        self.n += 1
        d = t1_diff(expected, json_of(got))
        if d:
            self.items.append(f"{label}: {d}")

    def report(self) -> None:
        shown = "\n".join(self.items[:12])
        assert not self.items, f"{self.area}: {len(self.items)} of {self.n} differ:\n{shown}"


@pytest.fixture
def mismatches():
    return Mismatches
