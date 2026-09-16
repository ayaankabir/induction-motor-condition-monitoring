"""Fault wrappers around the healthy plant.

Each fault must change equations or inputs. Spectral injection is not allowed.
Fault 04 (bearing outer race) is the documented exception in kind, not in
honesty: it adds a separate simulated vibration-sensor channel driven by
bearing kinematics and the simulated shaft speed; it does not inject anything
into the electrical traces.
"""

from imcm.faults.bearing import (
    impacts_per_shaft_revolution,
    outer_race_impact_times_s,
    outer_race_vibration,
    shaft_angle_rad,
)

__all__ = [
    "impacts_per_shaft_revolution",
    "outer_race_impact_times_s",
    "outer_race_vibration",
    "shaft_angle_rad",
]
