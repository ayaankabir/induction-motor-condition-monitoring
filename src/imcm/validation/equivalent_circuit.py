"""Steady-state T-equivalent circuit (RMS phasors).

This is the algebraic counterpart of the linear fifth-order model at constant
speed. It is used as a sanity check, not as a measured nameplate calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

from imcm.models.parameters import InductionMotorParameters


@dataclass(frozen=True)
class EquivalentCircuitPoint:
    slip: float
    stator_current_rms_a: float
    rotor_current_rms_a: float
    electromagnetic_torque_nm: float
    input_power_w: float
    airgap_power_w: float


def equivalent_circuit_point(
    params: InductionMotorParameters,
    slip: float,
    *,
    line_line_rms_v: float,
    frequency_hz: float,
) -> EquivalentCircuitPoint:
    """IEEE T-circuit at one slip. Rotor branch uses \(R_r/s\)."""
    if abs(slip) < 1.0e-12:
        raise ValueError("slip too small for R_r/s branch")
    omega_e = 2.0 * pi * frequency_hz
    v_ph = line_line_rms_v / sqrt(3.0)
    xs = omega_e * params.l_ls
    xm = omega_e * params.l_m
    xr = omega_e * params.l_lr
    z_r = params.r_r / slip + 1j * xr
    z_m = 1j * xm
    z_par = (z_m * z_r) / (z_m + z_r)
    z_in = params.r_s + 1j * xs + z_par
    i_s = v_ph / z_in
    i_r = i_s * z_m / (z_m + z_r)
    p_ag = 3.0 * (abs(i_r) ** 2) * (params.r_r / slip)
    omega_sync_mech = omega_e / params.pole_pairs
    tau_e = p_ag / omega_sync_mech
    p_in = 3.0 * (v_ph * i_s.conjugate()).real
    return EquivalentCircuitPoint(
        slip=slip,
        stator_current_rms_a=float(abs(i_s)),
        rotor_current_rms_a=float(abs(i_r)),
        electromagnetic_torque_nm=float(tau_e),
        input_power_w=float(p_in),
        airgap_power_w=float(p_ag),
    )
