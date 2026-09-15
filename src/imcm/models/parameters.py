"""Motor parameter containers.

Values must be tagged by provenance. Do not treat literature examples as measured data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Literal

Provenance = Literal["literature_example", "user_supplied", "identified", "unknown"]
StatorConnection = Literal["star"]


@dataclass(frozen=True)
class InductionMotorParameters:
    """Lumped parameters for the fifth-order squirrel-cage model.

    All rotor electrical parameters are stator-referred. Units are SI.
    Rated power and rated speed are metadata only; they are not ODE states.
    """

    r_s: float
    r_r: float
    l_ls: float
    l_lr: float
    l_m: float
    n_poles: int
    inertia: float
    viscous_friction: float
    rated_frequency_hz: float
    rated_line_line_voltage_v: float
    provenance: Provenance
    name: str
    notes: str = ""
    rated_mechanical_power_w: float | None = None
    rated_speed_rpm: float | None = None
    stator_connection: StatorConnection = "star"

    def __post_init__(self) -> None:
        if self.n_poles % 2 != 0 or self.n_poles < 2:
            raise ValueError("n_poles must be an even integer >= 2")
        for field_name in (
            "r_s",
            "r_r",
            "l_ls",
            "l_lr",
            "l_m",
            "inertia",
            "viscous_friction",
            "rated_frequency_hz",
            "rated_line_line_voltage_v",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

    @property
    def l_s(self) -> float:
        return self.l_ls + self.l_m

    @property
    def l_r(self) -> float:
        return self.l_lr + self.l_m

    @property
    def pole_pairs(self) -> int:
        return self.n_poles // 2

    @property
    def synchronous_speed_rpm(self) -> float:
        return 120.0 * self.rated_frequency_hz / self.n_poles

    @property
    def assumed_rated_torque_nm(self) -> float | None:
        """Torque from assumed rated power and speed. Not a measured nameplate."""
        if self.rated_mechanical_power_w is None or self.rated_speed_rpm is None:
            return None
        omega_m = self.rated_speed_rpm * 2.0 * pi / 60.0
        if omega_m <= 0.0:
            raise ValueError("rated_speed_rpm must be positive")
        return self.rated_mechanical_power_w / omega_m


def illustrative_4kw_400v_50hz_4pole() -> InductionMotorParameters:
    """Assumed literature-example machine for development.

    Numbers match a widely published 4 kW, 400 V, 50 Hz, 4-pole SI example
    (MathWorks Asynchronous Machine default SI set and copies in teaching notes).
    They are **not** laboratory measurements from this project.
    """
    return InductionMotorParameters(
        r_s=1.405,
        r_r=1.395,
        l_ls=0.005839,
        l_lr=0.005839,
        l_m=0.1722,
        n_poles=4,
        inertia=0.0131,
        viscous_friction=0.002985,
        rated_frequency_hz=50.0,
        rated_line_line_voltage_v=400.0,
        provenance="literature_example",
        name="illustrative_4kW_400V_50Hz_4pole",
        notes=(
            "Illustrative squirrel-cage parameters consistent with a published "
            "4 kW, 400 V, 50 Hz, 1430 r/min, 4-pole SI example. Not measured "
            "on a physical motor in this repository. Provenance: literature_example."
        ),
        rated_mechanical_power_w=4000.0,
        rated_speed_rpm=1430.0,
        stator_connection="star",
    )
