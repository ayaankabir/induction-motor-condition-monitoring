"""Locked first-milestone supply and load (not experimental conditions)."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class VoltageUnbalanceConfig:
    """Per-phase supply-voltage multipliers for controlled simulations.

    The disabled/default case is the approved balanced healthy supply.  Values
    are multipliers of ``SupplyConfig.phase_peak_v`` rather than new machine
    identification data.  This is a supply-side confounder, not a winding
    fault and must never be labelled as stator damage.
    """

    enabled: bool = False
    multipliers_abc: tuple[float, float, float] = (1.0, 1.0, 1.0)
    label: str = "balanced"

    def __post_init__(self) -> None:
        if len(self.multipliers_abc) != 3 or any(
            value <= 0.0 for value in self.multipliers_abc
        ):
            raise ValueError("voltage multipliers must be three positive values")


@dataclass(frozen=True)
class SupplyConfig:
    """Ideal grid-like voltages for the first healthy-plant milestone."""

    frequency_hz: float
    line_line_rms_v: float
    waveform: SupplyWaveform
    notes: str
    voltage_unbalance: VoltageUnbalanceConfig = VoltageUnbalanceConfig()

    @property
    def omega_e(self) -> float:
        return 2.0 * pi * self.frequency_hz

    @property
    def phase_peak_v(self) -> float:
        """Peak phase-to-neutral voltage for a balanced star machine."""
        return self.line_line_rms_v * sqrt(2.0) / sqrt(3.0)

    @property
    def phase_peak_abc_v(self) -> tuple[float, float, float]:
        """Per-phase peak voltages after any controlled supply unbalance."""
        if not self.voltage_unbalance.enabled:
            peak = self.phase_peak_v
            return (peak, peak, peak)
        return tuple(
            self.phase_peak_v * factor
            for factor in self.voltage_unbalance.multipliers_abc
        )


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


def fault_02_increased_mechanical_load_scenario() -> OperatingScenario:
    """Controlled Condition 02: 50% higher constant mechanical load torque.

    This is an operating-condition change, not a confirmed internal motor
    fault. Supply, Park convention, motor parameters, and stator resistance
    remain identical to the approved healthy baseline. Only the constant
    mechanical load torque changes from 15.0 N m to 22.5 N m.
    """
    healthy = first_milestone_scenario()
    return OperatingScenario(
        name="fault_02_increased_mechanical_load_plus_50pct",
        supply=healthy.supply,
        load=LoadConfig(
            load_type="constant_torque",
            torque_nm=1.5 * healthy.load.torque_nm,
            notes=(
                "Controlled operating-condition change: 50% higher constant "
                "mechanical load torque than the healthy baseline. "
                "Simulation-only; not a confirmed internal motor fault."
            ),
        ),
        park_convention=healthy.park_convention,
        stator_resistance=StatorResistanceConfig(
            enabled=False,
            multipliers_abc=(1.0, 1.0, 1.0),
            label="healthy_balanced",
        ),
        notes=(
            "Controlled operating-condition change, not a confirmed internal "
            "motor fault; simulation-only."
        ),
    )


def fault_03_supply_voltage_unbalance_scenario(
    multipliers_abc: tuple[float, float, float] = (1.0, 1.0, 0.9),
    *,
    label: str | None = None,
) -> OperatingScenario:
    """Controlled Fault 03: unbalanced three-phase supply voltages.

    Only the applied three-phase supply voltages change (per-phase multipliers of
    the balanced phase peak).  Motor equations, motor parameters, stator
    resistance, load torque, Park convention, and supply frequency match the
    approved healthy baseline.  This is a supply-side confounder, not a winding
    fault and not a confirmed internal motor fault.

    The default (1.0, 1.0, 0.9) is a controlled 10% reduction in the phase-C
    supply voltage.  A 5% case is available by passing (1.0, 1.0, 0.95).
    """
    healthy = first_milestone_scenario()
    resolved_label = label or "fault_03_phase_c_minus_10pct"
    return OperatingScenario(
        name="fault_03_supply_voltage_unbalance",
        supply=replace(
            healthy.supply,
            notes=(
                "Controlled unbalanced supply: per-phase voltage multipliers "
                f"{multipliers_abc} on the same 50 Hz balanced-sinusoid model. "
                "Supply-side confounder, not a winding fault."
            ),
            voltage_unbalance=VoltageUnbalanceConfig(
                enabled=True,
                multipliers_abc=multipliers_abc,
                label=resolved_label,
            ),
        ),
        load=healthy.load,
        park_convention=healthy.park_convention,
        stator_resistance=StatorResistanceConfig(
            enabled=False,
            multipliers_abc=(1.0, 1.0, 1.0),
            label="healthy_balanced",
        ),
        notes=(
            "Controlled supply voltage unbalance (confounder). Simulation-only; "
            "not a confirmed internal motor fault and not winding damage."
        ),
    )
