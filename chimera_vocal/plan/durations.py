"""Segment durations by the Klatt (1979) / MITalk rules (port of phonetics/durations.rs).

S1 stub (INTERFACES §14): types and signatures only; the implementation follows in S2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Union

from ..core.types import PlanOptions, Segment
from .inventory import Manner


@dataclass(frozen=True)
class SegInfo:
    """What the duration rules need to know about a segment (durations.rs:71-82, serde camelCase)."""

    vowel: bool
    manner: Manner | None = None
    voiced: bool | None = None
    dur: float | None = None          # fallback inherent duration (s) for phonemes outside the table

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SegInfo":
        m = d.get("manner")
        return cls(bool(d["vowel"]), Manner(m) if m is not None else None, d.get("voiced"), d.get("dur"))

    def to_json(self) -> dict:
        out: dict[str, Any] = {"vowel": self.vowel}
        if self.manner is not None:
            out["manner"] = self.manner.value
        if self.voiced is not None:
            out["voiced"] = self.voiced
        if self.dur is not None:
            out["dur"] = self.dur
        return out


InfoSource = Union[PlanOptions, Callable[[Segment], SegInfo]]


def klatt_durations(segs: Sequence[Segment], opts: InfoSource) -> list[float]:
    """Durations (s) for one phrase of segments; explicit ``dur`` values are kept (durations.rs:85-191).

    ``opts`` is either the frozen ``PlanOptions`` (segment info from the inventory for ``opts.body``, as
    ``plan_utterance`` does, planner.rs:185-191) or the template's info callback ``Segment -> SegInfo``
    (golden fixtures, design/requests/C1-1.md).
    """
    raise NotImplementedError
