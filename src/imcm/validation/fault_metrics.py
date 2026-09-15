"""Matched healthy/fault metrics for controlled simulated resistance imbalance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imcm.models.fifth_order_dq import HealthySimulationResult

LATE_WINDOW_START_S = 0.8
LATE_WINDOW_STOP_S = 1.0

# Absolute normal-versus-tight solver agreement limits for Fault 01 metrics.
CONVERGENCE_TOLERANCES = {
    "current_unbalance_pct": 1.0e-5,
    "negative_sequence_a": 1.0e-6,
    "torque_ripple_nm": 1.0e-6,
    "final_speed_difference_rpm": 1.0e-5,
    "final_slip_difference": 1.0e-8,
}


@dataclass(frozen=True)
class StatorResistanceComparisonMetrics:
    phase_current_rms_healthy_a: tuple[float, float, float]
    phase_current_rms_fault_a: tuple[float, float, float]
    current_unbalance_healthy_pct: float
    current_unbalance_fault_pct: float
    negative_sequence_healthy_a: float
    negative_sequence_fault_a: float
    negative_sequence_healthy_pct: float
    negative_sequence_fault_pct: float
    torque_ripple_healthy_nm: float
    torque_ripple_fault_nm: float
    final_speed_difference_rpm: float
    final_slip_difference: float


def _window(
    result: HealthySimulationResult,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float = LATE_WINDOW_STOP_S,
) -> np.ndarray:
    """Return the shared half-open late window ``t_start <= t < t_stop``."""
    if t_stop <= t_start:
        raise ValueError("metric-window stop must be greater than start")
    mask = (result.t >= t_start) & (result.t < t_stop)
    if np.count_nonzero(mask) < 8:
        raise ValueError("metric window is too short")
    return mask


def phase_current_rms(
    result: HealthySimulationResult,
    *,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float = LATE_WINDOW_STOP_S,
) -> tuple[float, float, float]:
    mask = _window(result, t_start, t_stop)
    return tuple(float(np.sqrt(np.mean(phase[mask] ** 2))) for phase in result.i_abc)


def current_unbalance_pct(rms: tuple[float, float, float]) -> float:
    values = np.asarray(rms, dtype=float)
    return float(100.0 * (np.max(values) - np.min(values)) / np.mean(values))


def negative_sequence_current(
    result: HealthySimulationResult,
    *,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float = LATE_WINDOW_STOP_S,
) -> tuple[float, float]:
    """Return magnitude and percent of positive sequence from 50 Hz RMS phasors."""
    mask = _window(result, t_start, t_stop)
    t = result.t[mask]
    theta = result.scenario.supply.omega_e * t
    # sqrt(2)*mean(x exp(-j theta)) is an RMS phasor for an integer-cycle window.
    phasors = np.sqrt(2.0) * np.mean(result.i_abc[:, mask] * np.exp(-1j * theta), axis=1)
    a = np.exp(2j * np.pi / 3.0)
    i_1 = (phasors[0] + a * phasors[1] + a**2 * phasors[2]) / 3.0
    i_2 = (phasors[0] + a**2 * phasors[1] + a * phasors[2]) / 3.0
    magnitude = float(abs(i_2))
    percent = float(100.0 * magnitude / abs(i_1)) if abs(i_1) > 0.0 else float("nan")
    return magnitude, percent


def torque_ripple_rms(
    result: HealthySimulationResult,
    *,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float = LATE_WINDOW_STOP_S,
) -> float:
    mask = _window(result, t_start, t_stop)
    values = result.tau_e[mask]
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))


def compare_stator_resistance_cases(
    healthy: HealthySimulationResult,
    fault: HealthySimulationResult,
    *,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float = LATE_WINDOW_STOP_S,
) -> StatorResistanceComparisonMetrics:
    """Compare matched time grids; this is simulation evidence, not diagnosis."""
    if not np.array_equal(healthy.t, fault.t):
        raise ValueError("healthy and fault cases must share the same output time grid")
    h_rms = phase_current_rms(healthy, t_start=t_start, t_stop=t_stop)
    f_rms = phase_current_rms(fault, t_start=t_start, t_stop=t_stop)
    h_i2, h_i2_pct = negative_sequence_current(healthy, t_start=t_start, t_stop=t_stop)
    f_i2, f_i2_pct = negative_sequence_current(fault, t_start=t_start, t_stop=t_stop)
    return StatorResistanceComparisonMetrics(
        phase_current_rms_healthy_a=h_rms,
        phase_current_rms_fault_a=f_rms,
        current_unbalance_healthy_pct=current_unbalance_pct(h_rms),
        current_unbalance_fault_pct=current_unbalance_pct(f_rms),
        negative_sequence_healthy_a=h_i2,
        negative_sequence_fault_a=f_i2,
        negative_sequence_healthy_pct=h_i2_pct,
        negative_sequence_fault_pct=f_i2_pct,
        torque_ripple_healthy_nm=torque_ripple_rms(healthy, t_start=t_start, t_stop=t_stop),
        torque_ripple_fault_nm=torque_ripple_rms(fault, t_start=t_start, t_stop=t_stop),
        final_speed_difference_rpm=float(fault.speed_rpm[-1] - healthy.speed_rpm[-1]),
        final_slip_difference=float(fault.slip[-1] - healthy.slip[-1]),
    )
