"""Locked first-milestone supply and load (not experimental conditions)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, pi, radians, sqrt
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
class BearingFaultConfig:
    """Simulated rolling-element bearing outer-race fault (vibration channel).

    Enabling this does **not** change the motor ODEs, supply, load, stator
    resistance, or Park convention.  It configures a separate simulated
    vibration-sensor channel carrying an impulse train at the BPFO
    characteristic frequency (see ``imcm.faults.bearing``).  Bearing geometry
    values are published literature-example dimensions of a 6205-series
    deep-groove ball bearing, **not** a measured bearing in this project, and
    the result is a simulated condition-monitoring signature, not a confirmed
    real-machine diagnosis.
    """

    enabled: bool = False
    nb_balls: int = 9
    ball_diameter_m: float = 7.94e-3
    pitch_diameter_m: float = 39.04e-3
    contact_angle_deg: float = 0.0
    resonance_hz: float = 2000.0
    resonance_decay_s: float = 1.0e-3
    amplitude_m_s2: float = 1.0
    label: str = "healthy_bearing"

    def __post_init__(self) -> None:
        if self.nb_balls < 1:
            raise ValueError("nb_balls must be a positive integer")
        if self.ball_diameter_m <= 0.0 or self.pitch_diameter_m <= 0.0:
            raise ValueError("bearing diameters must be positive")
        if self.pitch_diameter_m <= self.ball_diameter_m:
            raise ValueError("pitch diameter must exceed the ball diameter")
        if not -90.0 <= self.contact_angle_deg <= 90.0:
            raise ValueError("contact angle must lie in [-90, 90] degrees")
        if self.resonance_hz <= 0.0 or self.resonance_decay_s <= 0.0:
            raise ValueError("resonance frequency and decay time must be positive")
        if self.amplitude_m_s2 < 0.0:
            raise ValueError("amplitude_m_s2 must be non-negative")

    @property
    def ball_pitch_ratio(self) -> float:
        """``Bd / Pd`` appearing in the bearing characteristic frequencies."""
        return self.ball_diameter_m / self.pitch_diameter_m

    def bpfo_hz(self, shaft_hz: float) -> float:
        """Ball-pass frequency, outer race: ``(Nb/2) f_r (1 - (Bd/Pd) cos(phi))``."""
        return (
            0.5
            * self.nb_balls
            * shaft_hz
            * (1.0 - self.ball_pitch_ratio * cos(radians(self.contact_angle_deg)))
        )

    def bpfi_hz(self, shaft_hz: float) -> float:
        """Ball-pass frequency, inner race: ``(Nb/2) f_r (1 + (Bd/Pd) cos(phi))``."""
        return (
            0.5
            * self.nb_balls
            * shaft_hz
            * (1.0 + self.ball_pitch_ratio * cos(radians(self.contact_angle_deg)))
        )


@dataclass(frozen=True)
class RotorAsymmetryConfig:
    """Rotor electrical asymmetry proxy for broken-bar-related signatures.

    The single stator-referred cage resistance ``R_r`` is split between two
    orthogonal rotor-frame axes, ``R_r (1 + severity)`` and
    ``R_r (1 - severity)``.  In the locked synchronous frame this appears as a
    rotor resistance modulated at twice the slip angle (see
    ``imcm.faults.rotor_asymmetry`` for the equations).  This is a
    **simulation-only proxy** for broken-bar-related behaviour: it is not a
    bar-resolved cage model, it is not calibrated to a number of broken bars,
    and nothing here is experimentally validated or a real-machine diagnosis.

    ``reference_slip`` optionally pins the constant slip used to advance the
    modulation phase; when ``None`` it is solved from the healthy
    T-equivalent-circuit torque balance at the scenario load.  The mean rotor
    resistance stays exactly ``R_r``, so the proxy isolates the asymmetry
    effect and does not model the small mean-resistance shift a real broken
    bar also produces.
    """

    enabled: bool = False
    severity: float = 0.0
    initial_phase_deg: float = 0.0
    reference_slip: float | None = None
    label: str = "healthy_rotor"

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity < 1.0:
            raise ValueError("severity must lie in [0, 1)")
        if self.reference_slip is not None and not 0.0 < self.reference_slip < 1.0:
            raise ValueError("reference_slip must lie in (0, 1) when provided")


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
    bearing_fault: BearingFaultConfig = BearingFaultConfig()
    rotor_asymmetry: RotorAsymmetryConfig = RotorAsymmetryConfig()


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


def fault_04_bearing_outer_race_scenario(
    *,
    severity_scale: float = 1.0,
) -> OperatingScenario:
    """Controlled Fault 04: rolling-element bearing outer-race defect.

    The motor ODEs, supply, load, motor parameters, stator resistance, and
    Park convention are **identical** to the approved healthy baseline; the
    electrical traces are therefore bit-for-bit healthy.  The fault exists as
    a separate **simulated vibration channel** configured here and generated
    by ``imcm.faults.bearing`` at the BPFO characteristic frequency.  Bearing
    geometry is a published 6205-series literature example, not a measured
    bearing, and the output is a simulated condition-monitoring signature,
    not a confirmed real-machine diagnosis.
    """
    healthy = first_milestone_scenario()
    if severity_scale <= 0.0:
        raise ValueError("severity_scale must be positive")
    return OperatingScenario(
        name="fault_04_bearing_outer_race_bpfo",
        supply=healthy.supply,
        load=healthy.load,
        park_convention=healthy.park_convention,
        stator_resistance=StatorResistanceConfig(
            enabled=False,
            multipliers_abc=(1.0, 1.0, 1.0),
            label="healthy_balanced",
        ),
        bearing_fault=BearingFaultConfig(
            enabled=True,
            amplitude_m_s2=severity_scale,
            label="fault_04_outer_race_bpfo",
        ),
        notes=(
            "Controlled simulated bearing outer-race fault expressed through a "
            "vibration channel at BPFO. Electrical motor model unchanged; "
            "simulation-only; not a confirmed real-machine diagnosis."
        ),
    )


def fault_05_rotor_asymmetry_scenario(
    *,
    severity: float = 0.10,
    label: str | None = None,
) -> OperatingScenario:
    """Controlled Fault 05: rotor electrical asymmetry (broken-bar proxy).

    Supply, load, motor parameters, stator resistance, and Park convention
    match the approved healthy baseline; only the rotor resistance
    representation changes, as a documented **simulation-only proxy**: a
    rotor-frame axis resistance split ``R_r (1 +/- severity)`` whose
    synchronous-frame modulation advances at twice the reference slip angle.
    This is not a physically complete broken rotor bar model, is not
    bar-resolved, and is not severity-calibrated; no experimental validation
    or real-machine diagnosis is claimed.
    """
    healthy = first_milestone_scenario()
    resolved_label = label or (
        f"fault_05_rotor_asymmetry_severity_{round(severity * 100)}pct"
    )
    return OperatingScenario(
        name="fault_05_rotor_electrical_asymmetry",
        supply=healthy.supply,
        load=healthy.load,
        park_convention=healthy.park_convention,
        stator_resistance=StatorResistanceConfig(
            enabled=False,
            multipliers_abc=(1.0, 1.0, 1.0),
            label="healthy_balanced",
        ),
        rotor_asymmetry=RotorAsymmetryConfig(
            enabled=True,
            severity=severity,
            label=resolved_label,
        ),
        notes=(
            "Controlled simulated rotor electrical asymmetry proxy for "
            "broken-bar-related behaviour. Simulation-only; not a physically "
            "complete broken rotor bar model; not experimentally validated."
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
