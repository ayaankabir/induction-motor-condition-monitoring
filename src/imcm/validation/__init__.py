"""Physics and numerical sanity checks for simulated traces."""

from imcm.validation.equivalent_circuit import (
    EquivalentCircuitPoint,
    equivalent_circuit_point,
)
from imcm.validation.frequency import interpolated_rfft_peak_hz, mean_zero_crossing_frequency_hz
from imcm.validation.fault_metrics import (
    StatorResistanceComparisonMetrics,
    compare_stator_resistance_cases,
)
from imcm.validation.power_balance import power_balance_window
from imcm.validation.rotor_asymmetry_metrics import (
    RotorAsymmetryComparisonMetrics,
    compare_rotor_asymmetry_cases,
)

__all__ = [
    "EquivalentCircuitPoint",
    "equivalent_circuit_point",
    "interpolated_rfft_peak_hz",
    "mean_zero_crossing_frequency_hz",
    "StatorResistanceComparisonMetrics",
    "compare_stator_resistance_cases",
    "RotorAsymmetryComparisonMetrics",
    "compare_rotor_asymmetry_cases",
    "power_balance_window",
]
