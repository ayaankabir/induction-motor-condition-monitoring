"""Krause instantaneous power identity for the linear machine.

P_in = (3/2)(v_qs i_qs + v_ds i_ds)
     = stator+rotor copper + T_e ω_m + dW_mag/dt

After transients, dW_mag/dt ≈ 0, so input ≈ copper + mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imcm.models.fifth_order_dq import HealthySimulationResult


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
    p_cu = 1.5 * (
        result.params.r_s * (result.i_qs**2 + result.i_ds**2)
        + result.params.r_r * (result.i_qr**2 + result.i_dr**2)
    )
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
