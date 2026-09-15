"""Fifth-order dq squirrel-cage plant (healthy, Krause synchronous frame).

Equations: docs/modeling-plan.md §4. This is a reduced-order simulation, not a
digital twin and not experimental data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from imcm.models.flux_map import (
    currents_from_fluxes,
    electromagnetic_torque_flux_current,
)
from imcm.models.operating_scenario import OperatingScenario, first_milestone_scenario
from imcm.models.parameters import InductionMotorParameters, illustrative_4kw_400v_50hz_4pole
from imcm.models.stator_resistance import phase_resistances_abc, resistive_drop_qd
from imcm.signals.park import CONVENTION_NAME, balanced_phase_voltages, qd0_to_abc

STATE_NAMES = ("lambda_qs", "lambda_ds", "lambda_qr", "lambda_dr", "omega_m")

DEFAULT_MAX_STEP_S = 1.0e-4
DEFAULT_RTOL = 1.0e-6
DEFAULT_ATOL = 1.0e-8
DEFAULT_T_END_S = 1.0
DEFAULT_OUTPUT_DT_S = 1.0e-4


@dataclass
class HealthySimulationResult:
    """Time-domain traces from the healthy fifth-order model.

    Provenance of the waveforms is always ``simulated``. Motor parameters used to
    produce them keep their own provenance tag (typically ``literature_example``).
    """

    t: np.ndarray
    lambda_qs: np.ndarray
    lambda_ds: np.ndarray
    lambda_qr: np.ndarray
    lambda_dr: np.ndarray
    omega_m: np.ndarray
    i_qs: np.ndarray
    i_ds: np.ndarray
    i_qr: np.ndarray
    i_dr: np.ndarray
    tau_e: np.ndarray
    tau_l: np.ndarray
    v_qs: np.ndarray
    v_ds: np.ndarray
    v_abc: np.ndarray
    i_abc: np.ndarray
    params: InductionMotorParameters
    scenario: OperatingScenario
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def omega_r(self) -> np.ndarray:
        return self.params.pole_pairs * self.omega_m

    @property
    def slip(self) -> np.ndarray:
        omega_e = self.scenario.supply.omega_e
        return (omega_e - self.omega_r) / omega_e

    @property
    def speed_rpm(self) -> np.ndarray:
        return self.omega_m * 60.0 / (2.0 * np.pi)


def rest_initial_state() -> np.ndarray:
    """Start-from-rest: unfluxed machine, zero mechanical speed."""
    return np.zeros(5, dtype=float)


def healthy_state_derivative(
    t: float,
    x: np.ndarray,
    params: InductionMotorParameters,
    scenario: OperatingScenario,
) -> np.ndarray:
    """Right-hand side of the fifth-order healthy plant.

    ``t`` supplies the synchronous angle for an active phase-resistance
    imbalance; it is otherwise kept for ``solve_ivp`` compatibility.
    """
    lam_qs, lam_ds, lam_qr, lam_dr, omega_m = x
    omega = scenario.supply.omega_e
    v_qs = scenario.supply.phase_peak_v
    v_ds = 0.0
    omega_r = params.pole_pairs * omega_m
    i_qs, i_ds, i_qr, i_dr = currents_from_fluxes(
        lam_qs, lam_ds, lam_qr, lam_dr, params
    )
    r_abc = phase_resistances_abc(
        params.r_s,
        enabled=scenario.stator_resistance.enabled,
        multipliers_abc=scenario.stator_resistance.multipliers_abc,
    )
    if r_abc[0] == r_abc[1] == r_abc[2]:
        # Preserve the approved healthy scalar path exactly for balanced values.
        v_rqs, v_rds = r_abc[0] * i_qs, r_abc[0] * i_ds
    else:
        v_rqs, v_rds = resistive_drop_qd(i_qs, i_ds, omega * t, r_abc)
    p_lam_qs = v_qs - v_rqs - omega * lam_ds
    p_lam_ds = v_ds - v_rds + omega * lam_qs
    p_lam_qr = -params.r_r * i_qr - (omega - omega_r) * lam_dr
    p_lam_dr = -params.r_r * i_dr + (omega - omega_r) * lam_qr
    tau_e = electromagnetic_torque_flux_current(
        lam_qs, lam_ds, i_qs, i_ds, params.n_poles
    )
    if scenario.load.load_type != "constant_torque":
        raise NotImplementedError("Only constant load torque is implemented")
    tau_l = scenario.load.torque_nm
    p_omega_m = (tau_e - tau_l - params.viscous_friction * omega_m) / params.inertia
    return np.array([p_lam_qs, p_lam_ds, p_lam_qr, p_lam_dr, p_omega_m], dtype=float)


def _reconstruct(
    t: np.ndarray,
    y: np.ndarray,
    params: InductionMotorParameters,
    scenario: OperatingScenario,
) -> dict[str, np.ndarray]:
    lam_qs, lam_ds, lam_qr, lam_dr, omega_m = y
    delta = params.l_s * params.l_r - params.l_m**2
    i_qs = (params.l_r * lam_qs - params.l_m * lam_qr) / delta
    i_ds = (params.l_r * lam_ds - params.l_m * lam_dr) / delta
    i_qr = (params.l_s * lam_qr - params.l_m * lam_qs) / delta
    i_dr = (params.l_s * lam_dr - params.l_m * lam_ds) / delta
    tau_e = 1.5 * (params.n_poles / 2.0) * (lam_ds * i_qs - lam_qs * i_ds)
    tau_l = np.full_like(t, scenario.load.torque_nm)
    v_qs = np.full_like(t, scenario.supply.phase_peak_v)
    v_ds = np.zeros_like(t)
    v_0 = np.zeros_like(t)
    theta = scenario.supply.omega_e * t
    v_abc = balanced_phase_voltages(t, scenario.supply.omega_e, scenario.supply.phase_peak_v)
    i_qd0 = np.stack((i_qs, i_ds, np.zeros_like(i_qs)), axis=0)
    i_abc = qd0_to_abc(i_qd0, theta)
    return {
        "lambda_qs": lam_qs,
        "lambda_ds": lam_ds,
        "lambda_qr": lam_qr,
        "lambda_dr": lam_dr,
        "omega_m": omega_m,
        "i_qs": i_qs,
        "i_ds": i_ds,
        "i_qr": i_qr,
        "i_dr": i_dr,
        "tau_e": tau_e,
        "tau_l": tau_l,
        "v_qs": v_qs,
        "v_ds": v_ds,
        "v_0": v_0,
        "v_abc": v_abc,
        "i_abc": i_abc,
    }


def simulate_healthy(
    params: InductionMotorParameters | None = None,
    scenario: OperatingScenario | None = None,
    *,
    t_end: float = DEFAULT_T_END_S,
    max_step: float = DEFAULT_MAX_STEP_S,
    output_dt: float = DEFAULT_OUTPUT_DT_S,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    y0: np.ndarray | None = None,
) -> HealthySimulationResult:
    """Integrate the healthy plant from rest (unless ``y0`` is given).

    Parameters are the illustrative literature-example machine unless overridden.
    Results are simulated; they are not laboratory measurements.
    """
    if params is None:
        params = illustrative_4kw_400v_50hz_4pole()
    if scenario is None:
        scenario = first_milestone_scenario()
    if scenario.park_convention != CONVENTION_NAME:
        raise ValueError(
            f"Scenario Park convention {scenario.park_convention!r} does not match "
            f"locked convention {CONVENTION_NAME!r}"
        )
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if max_step <= 0.0 or output_dt <= 0.0:
        raise ValueError("max_step and output_dt must be positive")

    y0_vec = rest_initial_state() if y0 is None else np.asarray(y0, dtype=float)
    if y0_vec.shape != (5,):
        raise ValueError("y0 must have shape (5,)")

    n_out = int(np.floor(t_end / output_dt)) + 1
    t_eval = np.linspace(0.0, t_end, n_out)

    def fun(t: float, x: np.ndarray) -> np.ndarray:
        return healthy_state_derivative(t, x, params, scenario)

    sol = solve_ivp(
        fun,
        (0.0, t_end),
        y0_vec,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"RK45 integration failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("RK45 produced non-finite states (numerical instability)")

    rec = _reconstruct(sol.t, sol.y, params, scenario)
    metadata: dict[str, Any] = {
        "label": "healthy" if not scenario.stator_resistance.enabled else scenario.stator_resistance.label,
        "provenance": "simulated",
        "parameter_name": params.name,
        "parameter_provenance": params.provenance,
        "scenario_name": scenario.name,
        "park_convention": CONVENTION_NAME,
        "solver": "RK45",
        "max_step_s": max_step,
        "rtol": rtol,
        "atol": atol,
        "t_end_s": t_end,
        "output_dt_s": output_dt,
        "initial_condition": "rest" if y0 is None else "user_supplied",
        "nfev": int(sol.nfev),
        "njev": int(sol.njev) if sol.njev is not None else 0,
        "n_eval": int(sol.t.size),
        "message": sol.message,
        "supply_frequency_hz": scenario.supply.frequency_hz,
        "load_type": scenario.load.load_type,
        "load_torque_nm": scenario.load.torque_nm,
        "stator_resistance_case": scenario.stator_resistance.label,
        "stator_resistances_abc_ohm": phase_resistances_abc(
            params.r_s,
            enabled=scenario.stator_resistance.enabled,
            multipliers_abc=scenario.stator_resistance.multipliers_abc,
        ),
        "inverter": False,
        "faults": scenario.stator_resistance.enabled,
        "experimental_validation": False,
    }
    return HealthySimulationResult(
        t=sol.t,
        lambda_qs=rec["lambda_qs"],
        lambda_ds=rec["lambda_ds"],
        lambda_qr=rec["lambda_qr"],
        lambda_dr=rec["lambda_dr"],
        omega_m=rec["omega_m"],
        i_qs=rec["i_qs"],
        i_ds=rec["i_ds"],
        i_qr=rec["i_qr"],
        i_dr=rec["i_dr"],
        tau_e=rec["tau_e"],
        tau_l=rec["tau_l"],
        v_qs=rec["v_qs"],
        v_ds=rec["v_ds"],
        v_abc=rec["v_abc"],
        i_abc=rec["i_abc"],
        params=params,
        scenario=scenario,
        metadata=metadata,
    )
