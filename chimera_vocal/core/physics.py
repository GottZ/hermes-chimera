"""Physical constants for airway acoustics, CGS units throughout (physics.rs).

cm, g, s; pressure in dyn/cm^2 (= 0.1 Pa); volume velocity in cm^3/s.
"""

from __future__ import annotations

from typing import NamedTuple

RHO = 1.14e-3            # air density at body temperature, saturated (g/cm^3)
C_SOUND = 35000.0        # speed of sound in warm humid air (cm/s)
MU = 1.86e-4             # dynamic viscosity of air (poise)
NU = MU / RHO            # kinematic viscosity (cm^2/s)
RHO_C2 = RHO * C_SOUND * C_SOUND  # bulk modulus of air (dyn/cm^2)
CM_H2O = 980.665         # 1 cm H2O in dyn/cm^2
RE_CRIT = 1800.0         # critical Reynolds number for the onset of turbulence noise


class WallParams(NamedTuple):
    """Soft-tissue wall parameters per unit area (Ishizaka, French & Flanagan 1975)."""

    mass: float          # g/cm^2
    resistance: float    # dyn*s/cm^3
    stiffness: float     # dyn/cm^3


WALL_SOFT = WallParams(1.5, 1600.0, 3.0e5)
WALL_TRACHEA = WallParams(0.6, 1000.0, 1.0e6)   # cartilaginous trachea: stiffer and lighter
WALL_NASAL = WallParams(1.0, 3000.0, 1.0e6)     # nasal mucosa: lossy
