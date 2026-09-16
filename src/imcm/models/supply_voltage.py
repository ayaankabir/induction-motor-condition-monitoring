"""Applied supply-voltage model for the fifth-order dq plant.

The balanced case is the approved control path and is reproduced exactly: a DC
q-axis voltage, zero d-axis voltage, and zero zero-sequence voltage.  The
unbalanced case scales each phase peak independently while keeping the locked
120-degree spacing, so the applied voltage contains positive- and
negative-sequence components.

The machine is a three-wire system: the zero-sequence voltage does not drive
current, so the plant responds only to the positive- and negative-sequence
components.  This is a supply-side confounder, not a winding fault.
"""

from __future__ import annotations

import numpy as np

from imcm.models.operating_scenario import OperatingScenario
from imcm.signals.park import (
    abc_to_qd0,
    balanced_phase_voltages,
    unbalanced_phase_voltages,
)


def _is_balanced(supply) -> bool:
    """True when the applied supply is the exact balanced control path.

    Both the disabled default and an explicitly enabled all-ones multiplier set
    are treated as balanced so they reproduce the healthy result exactly.
    """
    unbalance = supply.voltage_unbalance
    return (not unbalance.enabled) or unbalance.multipliers_abc == (1.0, 1.0, 1.0)


def supply_voltage_qd(
    t: float, scenario: OperatingScenario
) -> tuple[float, float, float]:
    """Return ``(v_qs, v_ds, v_0)`` at time ``t`` for the applied supply.

    The balanced branch returns the exact approved scalars so the healthy,
    Fault 01, and Fault 02 paths are bit-for-bit unchanged.
    """
    supply = scenario.supply
    if _is_balanced(supply):
        return supply.phase_peak_v, 0.0, 0.0
    theta = supply.omega_e * t
    v_abc = unbalanced_phase_voltages(
        t,
        supply.omega_e,
        supply.phase_peak_v,
        supply.voltage_unbalance.multipliers_abc,
    )
    v_qd0 = abc_to_qd0(v_abc, theta)
    return float(v_qd0[0]), float(v_qd0[1]), float(v_qd0[2])


def supply_voltage_traces(
    t: np.ndarray, scenario: OperatingScenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(v_abc, v_qs, v_ds, v_0)`` arrays on the output grid."""
    t = np.asarray(t, dtype=float)
    supply = scenario.supply
    theta = supply.omega_e * t
    if _is_balanced(supply):
        v_abc = balanced_phase_voltages(t, supply.omega_e, supply.phase_peak_v)
        v_qs = np.full_like(t, supply.phase_peak_v)
        v_ds = np.zeros_like(t)
        v_0 = np.zeros_like(t)
        return v_abc, v_qs, v_ds, v_0
    v_abc = unbalanced_phase_voltages(
        t,
        supply.omega_e,
        supply.phase_peak_v,
        supply.voltage_unbalance.multipliers_abc,
    )
    v_qd0 = abc_to_qd0(v_abc, theta)
    return v_abc, v_qd0[0], v_qd0[1], v_qd0[2]


def voltage_unbalance_factor_pct(
    t: np.ndarray, v_abc: np.ndarray, omega_e: float
) -> float:
    """Return ``|V2| / |V1| * 100`` from 50 Hz RMS phasors.

    Uses the standard symmetrical-component decomposition.  The zero-sequence
    component is not used, matching the three-wire machine behaviour.
    """
    t = np.asarray(t, dtype=float)
    v_abc = np.asarray(v_abc, dtype=float)
    theta = omega_e * t
    phasors = np.sqrt(2.0) * np.mean(v_abc * np.exp(-1j * theta), axis=1)
    a = np.exp(2j * np.pi / 3.0)
    v_1 = (phasors[0] + a * phasors[1] + a**2 * phasors[2]) / 3.0
    v_2 = (phasors[0] + a**2 * phasors[1] + a * phasors[2]) / 3.0
    if abs(v_1) == 0.0:
        return float("nan")
    return float(100.0 * abs(v_2) / abs(v_1))
