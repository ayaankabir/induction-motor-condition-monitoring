"""Krause instantaneous power identity for the linear machine.

P_in = (3/2)(v_qs i_qs + v_ds i_ds)
     = stator+rotor copper + T_e ω_m + dW_mag/dt

After transients, dW_mag/dt ≈ 0, so input ≈ copper + mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imcm.models.fifth_order_dq import HealthySimulationResult
from imcm.models.rotor_asymmetry import modulation_phase_rad, rotor_copper_loss_w
from imcm.models.stator_resistance import phase_resistances_abc, resistance_matrix_qd


@dataclass(frozen=True)
class PowerBalanceWindow:
    p_in_mean: float
    p_copper_mean: float
    p_mech_mean: float
    p_mag_dot_mean: float
    residual_mean: float
    residual_rel: float


def magnetic_energy(result: HealthySimulationResult) -> np.ndarray:
    """Co-energy of the linear coupled inductors (Krause 3/2 scaling)."""
    return 0.75 * (
        result.i_qs * result.lambda_qs
        + result.i_ds * result.lambda_ds
        + result.i_qr * result.lambda_qr
        + result.i_dr * result.lambda_dr
    )


def instantaneous_powers(result: HealthySimulationResult) -> dict[str, np.ndarray]:
    p_in = 1.5 * (result.v_qs * result.i_qs + result.v_ds * result.i_ds)
    r_abc = phase_resistances_abc(
        result.params.r_s,
        enabled=result.scenario.stator_resistance.enabled,
        multipliers_abc=result.scenario.stator_resistance.multipliers_abc,
    )
    if r_abc[0] == r_abc[1] == r_abc[2]:
        p_cu_stator = 1.5 * r_abc[0] * (result.i_qs**2 + result.i_ds**2)
    else:
        theta = result.scenario.supply.omega_e * result.t
        p_cu_stator = np.fromiter(
            (
                1.5
                * np.array((i_qs, i_ds))
                @ resistance_matrix_qd(angle, r_abc)
                @ np.array((i_qs, i_ds))
                for i_qs, i_ds, angle in zip(result.i_qs, result.i_ds, theta)
            ),
            dtype=float,
            count=result.t.size,
        )
    rotor = result.scenario.rotor_asymmetry
    if rotor.enabled and rotor.severity != 0.0:
        # Fault 05 proxy: rotor-frame axis resistance split.  The healthy
        # scalar path below is preserved exactly for disabled/zero severity.
        two_phi = modulation_phase_rad(result.t, rotor, result.scenario.supply.omega_e)
        p_cu_rotor = rotor_copper_loss_w(
            result.i_qr, result.i_dr, two_phi, rotor.severity, result.params.r_r
        )
    else:
        p_cu_rotor = 1.5 * result.params.r_r * (result.i_qr**2 + result.i_dr**2)
    p_cu = p_cu_stator + p_cu_rotor
    p_mech = result.tau_e * result.omega_m
    w_mag = magnetic_energy(result)
    p_mag = np.gradient(w_mag, result.t)
    residual = p_in - (p_cu + p_mech + p_mag)
    return {
        "p_in": p_in,
        "p_copper": p_cu,
        "p_mech": p_mech,
        "p_mag_dot": p_mag,
        "residual": residual,
        "w_mag": w_mag,
    }


def power_balance_window(
    result: HealthySimulationResult, *, t_start: float
) -> PowerBalanceWindow:
    mask = result.t >= t_start
    if np.count_nonzero(mask) < 8:
        raise ValueError("power-balance window is too short")
    powers = instantaneous_powers(result)
    p_in = float(np.mean(powers["p_in"][mask]))
    p_cu = float(np.mean(powers["p_copper"][mask]))
    p_mech = float(np.mean(powers["p_mech"][mask]))
    p_mag = float(np.mean(powers["p_mag_dot"][mask]))
    residual = float(np.mean(powers["residual"][mask]))
    scale = max(abs(p_in), 1.0)
    return PowerBalanceWindow(
        p_in_mean=p_in,
        p_copper_mean=p_cu,
        p_mech_mean=p_mech,
        p_mag_dot_mean=p_mag,
        residual_mean=residual,
        residual_rel=abs(residual) / scale,
    )
