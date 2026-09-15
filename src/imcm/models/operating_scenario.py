"""Locked first-milestone supply and load (not experimental conditions)."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Literal

ParkConvention = Literal["krause_classical_2_3"]
SupplyWaveform = Literal["balanced_sinusoidal"]
LoadType = Literal["constant_torque"]


PARK_CONVENTION: ParkConvention = "krause_classical_2_3"


@dataclass(frozen=True)
class StatorResistanceConfig:
    """Per-phase stator-resistance multipliers for controlled simulations.

    The disabled/default case is the approved balanced healthy plant.  Values
    are multipliers of ``InductionMotorParameters.r_s`` rather than new machine
    identification data.
    """

    enabled: bool = False
    multipliers_abc: tuple[float, float, float] = (1.0, 1.0, 1.0)
    label: str = "healthy_balanced"

    def __post_init__(self) -> None:
        if len(self.multipliers_abc) != 3 or any(value <= 0.0 for value in self.multipliers_abc):
            raise ValueError("stator-resistance multipliers must be three positive values")


@dataclass(frozen=True)
class SupplyConfig:
    """Ideal grid-like voltages for the first healthy-plant milestone."""

    frequency_hz: float
    line_line_rms_v: float
    waveform: SupplyWaveform
    notes: str

    @property
    def omega_e(self) -> float:
        return 2.0 * pi * self.frequency_hz

    @property
    def phase_peak_v(self) -> float:
        """Peak phase-to-neutral voltage for a balanced star machine."""
        return self.line_line_rms_v * sqrt(2.0) / sqrt(3.0)


@dataclass(frozen=True)
class LoadConfig:
    """Mechanical load. First milestone is constant torque only."""

    load_type: LoadType
    torque_nm: float
    notes: str


@dataclass(frozen=True)
class OperatingScenario:
    name: str
    supply: SupplyConfig
    load: LoadConfig
    park_convention: ParkConvention
    notes: str
    stator_resistance: StatorResistanceConfig = StatorResistanceConfig()


def first_milestone_scenario() -> OperatingScenario:
    return OperatingScenario(
        name="healthy_50hz_constant_torque",
        supply=SupplyConfig(
            frequency_hz=50.0,
            line_line_rms_v=400.0,
            waveform="balanced_sinusoidal",
            notes=(
                "Assumed balanced 50 Hz sinusoids. Inverter-fed PWM is out of "
                "scope for the first milestone. Not a recorded grid waveform."
            ),
        ),
        load=LoadConfig(
            load_type="constant_torque",
            torque_nm=15.0,
            notes=(
                "Assumed constant load torque (~0.56 of the assumed 4 kW rated "
                "torque at 1430 r/min). Not a measured dynamometer setting. "
                "Fan-type load is deferred."
            ),
        ),
        park_convention=PARK_CONVENTION,
        notes=(
            "First-milestone operating point for the healthy reduced-order plant. "
            "No faults. Simulation-only."
        ),
    )


def fault_01_phase_a_resistance_imbalance_scenario() -> OperatingScenario:
    """Controlled Fault 01: +10% phase-A stator resistance.

    Supply, load, and all other scenario values intentionally match the healthy
    first-milestone case.  This is a simulated high-resistance proxy, not a
    measured fault condition or an inter-turn-short model.
    """
    healthy = first_milestone_scenario()
    return OperatingScenario(
        name="fault_01_phase_a_stator_resistance_plus_10pct",
        supply=healthy.supply,
        load=healthy.load,
        park_convention=healthy.park_convention,
        notes=(
            "Controlled simulated stator-resistance imbalance: phase A is 1.10 "
            "times the illustrative per-phase resistance; phases B/C are unchanged."
        ),
        stator_resistance=StatorResistanceConfig(
            enabled=True,
            multipliers_abc=(1.10, 1.0, 1.0),
            label="fault_01_phase_a_plus_10pct",
        ),
    )
