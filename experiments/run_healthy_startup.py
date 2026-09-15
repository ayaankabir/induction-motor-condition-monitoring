"""Run the first-milestone healthy DOL start-up and write traces plus figures.

Usage (from the repository root, with the package installed):

    python experiments/run_healthy_startup.py

Outputs go to results/ and are labeled simulated / literature-example.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.reporting.healthy_plots import save_healthy_startup_figures
from imcm.validation.equivalent_circuit import equivalent_circuit_point
from imcm.validation.power_balance import power_balance_window


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    result = simulate_healthy(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4)
    t_win = 0.8
    mask = result.t >= t_win
    slip = float(np.mean(result.slip[mask]))
    tau_e = float(np.mean(result.tau_e[mask]))
    omega_m = float(np.mean(result.omega_m[mask]))
    i_rms = float(np.sqrt(np.mean(result.i_abc[0, mask] ** 2)))
    pb = power_balance_window(result, t_start=t_win)
    ec = equivalent_circuit_point(
        result.params,
        slip,
        line_line_rms_v=result.scenario.supply.line_line_rms_v,
        frequency_hz=result.scenario.supply.frequency_hz,
    )
    tau_friction = result.params.viscous_friction * omega_m

    summary = {
        **result.metadata,
        "steady_window_t_start_s": t_win,
        "mean_slip": slip,
        "mean_speed_rpm": float(np.mean(result.speed_rpm[mask])),
        "mean_tau_e_nm": tau_e,
        "tau_l_nm": result.scenario.load.torque_nm,
        "mean_B_omega_nm": tau_friction,
        "torque_balance_nm": tau_e - (result.scenario.load.torque_nm + tau_friction),
        "phase_a_current_rms_a": i_rms,
        "power_balance_residual_rel": pb.residual_rel,
        "equivalent_circuit_tau_e_nm": ec.electromagnetic_torque_nm,
        "equivalent_circuit_is_rms_a": ec.stator_current_rms_a,
        "parameter_notes": result.params.notes,
        "honesty": (
            "Waveforms are ODE simulation. Parameters are a published "
            "illustrative set, not measurements. No experimental validation."
        ),
    }

    np.savez_compressed(
        out / "healthy_startup.npz",
        t=result.t,
        i_a=result.i_abc[0],
        i_b=result.i_abc[1],
        i_c=result.i_abc[2],
        v_a=result.v_abc[0],
        v_b=result.v_abc[1],
        v_c=result.v_abc[2],
        omega_m=result.omega_m,
        tau_e=result.tau_e,
        tau_l=result.tau_l,
        slip=result.slip,
        i_qs=result.i_qs,
        i_ds=result.i_ds,
        lambda_qs=result.lambda_qs,
        lambda_ds=result.lambda_ds,
        lambda_qr=result.lambda_qr,
        lambda_dr=result.lambda_dr,
    )
    (out / "healthy_startup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    paths = save_healthy_startup_figures(result, out)
    print("Simulation finished (RK45).")
    print(json.dumps(summary, indent=2))
    print("Wrote:")
    for path in [out / "healthy_startup.npz", out / "healthy_startup_summary.json", *paths]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
