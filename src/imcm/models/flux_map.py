"""Linear flux–current map for the fifth-order two-axis model.

Algebra only. This module does not integrate ODEs.
"""

from __future__ import annotations

from imcm.models.parameters import InductionMotorParameters


def inductance_delta(params: InductionMotorParameters) -> float:
    delta = params.l_s * params.l_r - params.l_m**2
    if delta <= 0.0:
        raise ValueError("Ls Lr - Lm^2 must be positive")
    return delta


def currents_from_fluxes(
    lambda_qs: float,
    lambda_ds: float,
    lambda_qr: float,
    lambda_dr: float,
    params: InductionMotorParameters,
) -> tuple[float, float, float, float]:
    """Return (i_qs, i_ds, i_qr, i_dr)."""
    delta = inductance_delta(params)
    i_qs = (params.l_r * lambda_qs - params.l_m * lambda_qr) / delta
    i_ds = (params.l_r * lambda_ds - params.l_m * lambda_dr) / delta
    i_qr = (params.l_s * lambda_qr - params.l_m * lambda_qs) / delta
    i_dr = (params.l_s * lambda_dr - params.l_m * lambda_ds) / delta
    return i_qs, i_ds, i_qr, i_dr


def fluxes_from_currents(
    i_qs: float,
    i_ds: float,
    i_qr: float,
    i_dr: float,
    params: InductionMotorParameters,
) -> tuple[float, float, float, float]:
    """Return (lambda_qs, lambda_ds, lambda_qr, lambda_dr)."""
    lambda_qs = params.l_s * i_qs + params.l_m * i_qr
    lambda_ds = params.l_s * i_ds + params.l_m * i_dr
    lambda_qr = params.l_r * i_qr + params.l_m * i_qs
    lambda_dr = params.l_r * i_dr + params.l_m * i_ds
    return lambda_qs, lambda_ds, lambda_qr, lambda_dr


def electromagnetic_torque_flux_current(
    lambda_qs: float,
    lambda_ds: float,
    i_qs: float,
    i_ds: float,
    n_poles: int,
) -> float:
    """Krause torque: (3/2)*(P/2)*(λds iqs − λqs ids)."""
    return 1.5 * (n_poles / 2.0) * (lambda_ds * i_qs - lambda_qs * i_ds)


def electromagnetic_torque_currents(
    i_qs: float,
    i_ds: float,
    i_qr: float,
    i_dr: float,
    params: InductionMotorParameters,
) -> float:
    """Equivalent linear-magnetics form: (3/2)*(P/2)*Lm*(iqs idr − ids iqr)."""
    return 1.5 * (params.n_poles / 2.0) * params.l_m * (i_qs * i_dr - i_ds * i_qr)
