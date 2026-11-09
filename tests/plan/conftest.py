"""Planning tests: the plan comparison of golden_planning.rs ``cmp`` at the port's tolerances.

Objects need the same keys; strings, labels and booleans exact; ``speed``, ``releaseHold`` and ``tag`` exact
(the template's EXACT list); every other number T1n: <= 1e-12 relative with floor 1e-12 (the template
allows 1e-9). Folder without __init__.py (pytest --import-mode=importlib).
"""

from __future__ import annotations

import pytest

from support import rel

EXACT = ("speed", "releaseHold", "tag")
T1N = 1e-12


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class PlanCompare:
    """Compare JSON values area by area; collect all differences and fail once with the first ones."""

    def __init__(self, area: str) -> None:
        self.area, self.cases, self.numbers, self.worst, self.worst_at = area, 0, 0, 0.0, "-"
        self.errors: list[str] = []

    def check(self, label: str, want, got) -> None:
        self.cases += 1
        before = len(self.errors)
        self._cmp(label, want, got)
        del self.errors[before + 3:]          # at most three per case

    def _cmp(self, path: str, want, got) -> None:
        if isinstance(want, dict):
            if not isinstance(got, dict) or set(want) != set(got):
                self.errors.append(f"{path}: keys {sorted(want)} != "
                                   f"{sorted(got) if isinstance(got, dict) else type(got).__name__}")
                return
            for k, v in want.items():
                p = f"{path}.{k}"
                if k in EXACT:
                    self.numbers += 1
                    if not (_is_num(v) and _is_num(got[k]) and float(v) == float(got[k])):
                        self.errors.append(f"{p}: must be exact: {v!r} != {got[k]!r}")
                else:
                    self._cmp(p, v, got[k])
        elif isinstance(want, list):
            if not isinstance(got, list) or len(want) != len(got):
                self.errors.append(f"{path}: length {len(want)} != "
                                   f"{len(got) if isinstance(got, list) else type(got).__name__}")
                return
            for i, (a, b) in enumerate(zip(want, got)):
                self._cmp(f"{path}[{i}]", a, b)
        elif _is_num(want):
            if not _is_num(got):
                self.errors.append(f"{path}: {want!r} != {got!r}")
                return
            self.numbers += 1
            e = rel(float(want), float(got), T1N)
            if e > self.worst:
                self.worst, self.worst_at = e, path
            if e > T1N:
                self.errors.append(f"{path}: {want!r} vs {got!r} (rel {e:.2e})")
        elif type(want) is not type(got) or want != got:
            self.errors.append(f"{path}: {want!r} != {got!r}")

    def report(self) -> str:
        summary = (f"{self.area}: {self.cases} cases, {self.numbers} numbers, worst rel {self.worst:.2e} "
                   f"at {self.worst_at}")
        shown = "\n".join(self.errors[:12])
        assert not self.errors, f"{summary}; {len(self.errors)}+ differences:\n{shown}"
        return summary


@pytest.fixture
def plan_compare():
    return PlanCompare
