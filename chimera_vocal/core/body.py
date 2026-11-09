"""Human bodies, body mixes and default pitch (anatomy/human.rs:262-415, engine.rs:24-35)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from .jsmath import clamp, exp, ln

BODY_NAMES: tuple[str, ...] = ("male", "female", "child")


@dataclass(frozen=True)
class HumanBody:
    scale: float          # overall size scale (1 = adult male)
    pharynx: float        # extra pharynx length scale
    tongue_radius: float  # tongue body radius (cm, before scale)
    lip_width: float      # max lip width (cm, before scale)
    name: str = field(default="", compare=False)


BODY_MALE = HumanBody(1.0, 1.0, 2.25, 4.6, "Adult male")
BODY_FEMALE = HumanBody(0.9, 0.88, 2.2, 4.4, "Adult female")
BODY_CHILD = HumanBody(0.74, 0.8, 2.15, 4.0, "Child")
BODIES: dict[str, HumanBody] = {"male": BODY_MALE, "female": BODY_FEMALE, "child": BODY_CHILD}


def parse_body_name(s: str) -> str:
    """BodyName::parse: "female"/"child", anything else is male (human.rs:300-307)."""
    return s if s in ("female", "child") else "male"


def _unit(x: float) -> float:
    """BodyMix::new clamp: non-finite -> 0, else clamp to 0..1."""
    x = float(x)
    return clamp(x, 0.0, 1.0) if math.isfinite(x) else 0.0


@dataclass(frozen=True)
class BodyMix:
    """``female`` from adult male (0) to adult female (1), ``child`` from adult (0) to child (1).

    Fields are stored as given (like the Rust struct); weights() clamps them as BodyMix::new does.
    """

    female: float = 0.0
    child: float = 0.0

    @classmethod
    def parse(cls, spec: "str | Mapping[str, float] | BodyMix") -> "BodyMix":
        """Serde form of the template: a preset name or ``{"female": f, "child": c}`` (missing = 0)."""
        if isinstance(spec, BodyMix):
            return spec
        if isinstance(spec, str):
            return {"male": cls(0.0, 0.0), "female": cls(1.0, 0.0), "child": cls(0.0, 1.0)}[parse_body_name(spec)]
        if isinstance(spec, Mapping):
            return cls(_unit(spec.get("female", 0.0)), _unit(spec.get("child", 0.0)))
        raise TypeError(f"body must be a name, a mapping or a BodyMix, not {type(spec).__name__}")

    def weights(self) -> tuple[float, float, float]:
        """(male, female, child) = ((1-c)(1-f), (1-c)f, c) after clamping (human.rs:339-343)."""
        f, c = _unit(self.female), _unit(self.child)
        return ((1.0 - c) * (1.0 - f), (1.0 - c) * f, c)

    def preset(self) -> str | None:
        """The preset this is exactly (a weight == 1.0), first in male/female/child order."""
        for name, w in zip(BODY_NAMES, self.weights()):
            if w == 1.0:
                return name
        return None

    def nearest(self) -> str:
        """The preset with the largest weight (first wins on ties, human.rs:350-360)."""
        w = self.weights()
        best = 0
        for i in (1, 2):
            if w[i] > w[best]:
                best = i
        return BODY_NAMES[best]

    def blend(self, f: Callable[[str], float]) -> float:
        """f(preset) exactly at a preset, else the weighted left fold from 0.0 (human.rs:362-368)."""
        p = self.preset()
        if p is not None:
            return f(p)
        acc = 0.0
        for name, w in zip(BODY_NAMES, self.weights()):
            acc = acc + w * f(name)
        return acc

    def body(self) -> HumanBody:
        """The blended anatomy, named after the nearest preset (human.rs:370-381)."""
        p = self.preset()
        if p is not None:
            return BODIES[p]
        return HumanBody(
            scale=self.blend(lambda b: BODIES[b].scale),
            pharynx=self.blend(lambda b: BODIES[b].pharynx),
            tongue_radius=self.blend(lambda b: BODIES[b].tongue_radius),
            lip_width=self.blend(lambda b: BODIES[b].lip_width),
            name=BODIES[self.nearest()].name,
        )

    def to_json(self) -> dict:
        return {"female": self.female, "child": self.child}


BODY_MIX_MALE = BodyMix(0.0, 0.0)
BODY_MIX_FEMALE = BodyMix(1.0, 0.0)
BODY_MIX_CHILD = BodyMix(0.0, 1.0)

_DEFAULT_F0 = {"male": 110.0, "female": 205.0, "child": 260.0}


def default_f0(body: "BodyMix | str | Mapping[str, float]") -> float:
    """Default pitch per body (Hz): the presets', blended geometrically in between (engine.rs:24-35)."""
    mix = BodyMix.parse(body)
    p = mix.preset()
    if p is not None:
        return _DEFAULT_F0[p]
    return exp(mix.blend(lambda b: ln(_DEFAULT_F0[b])))
