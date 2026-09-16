"""Run matched healthy and Condition 02 (increased mechanical load) cases.

Condition 02 is a controlled operating-condition change: the constant load
torque is raised from 15.0 N m to 22.5 N m (+50%).  It is not a confirmed
internal motor fault.  Supply, motor parameters, Park convention, and stator
resistance are unchanged from the healthy baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    fault_02_increased_mechanical_load_scenario,
    first_milestone_scenario,
)
from imcm.validation.power_balance import power_balance_window


def _late_mean(values: np.ndarray, t: np.ndarray, t_start: float = 0.8) -> float:
    return float(np.mean(values[t >= t_start]))


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    healthy = simulate_healthy(scenario=first_milestone_scenario(), **kwargs)
    condition = simulate_healthy(scenario=fault_02_increased_mechanical_load_scenario(), **kwargs)

    healthy_pb = power_balance_window(healthy, t_start=0.8)
    condition_pb = power_balance_window(condition, t_start=0.8)

    summary = {
        "label": "simulated_fault_02_increased_mechanical_load_comparison",
        "condition_kind": "operating_condition_change_not_internal_fault",
        "parameter_provenance": healthy.params.provenance,
        "healthy_metadata": healthy.metadata,
        "condition_metadata": condition.metadata,
        "healthy_load_torque_nm": healthy.scenario.load.torque_nm,
        "condition_load_torque_nm": condition.scenario.load.torque_nm,
        "load_increase_percent": 100.0
        * (condition.scenario.load.torque_nm - healthy.scenario.load.torque_nm)
        / healthy.scenario.load.torque_nm,
        "healthy_final_speed_rpm": float(healthy.speed_rpm[-1]),
        "condition_final_speed_rpm": float(condition.speed_rpm[-1]),
        "final_speed_difference_rpm": float(condition.speed_rpm[-1] - healthy.speed_rpm[-1]),
        "healthy_final_slip": float(healthy.slip[-1]),
        "condition_final_slip": float(condition.slip[-1]),
        "final_slip_difference": float(condition.slip[-1] - healthy.slip[-1]),
        "healthy_late_mean_torque_nm": _late_mean(healthy.tau_e, healthy.t),
        "condition_late_mean_torque_nm": _late_mean(condition.tau_e, condition.t),
        "healthy_late_mean_i_qs_a": _late_mean(healthy.i_qs, healthy.t),
        "condition_late_mean_i_qs_a": _late_mean(condition.i_qs, condition.t),
        "healthy_power_balance_residual_rel": healthy_pb.residual_rel,
        "condition_power_balance_residual_rel": condition_pb.residual_rel,
        "note": (
            "Controlled operating-condition change: 50% higher constant mechanical "
            "load torque (15.0 -> 22.5 N m). Simulation-only; not a confirmed "
            "internal motor fault and not experimental validation."
        ),
    }
    np.savez_compressed(
        out / "fault_02_increased_mechanical_load.npz",
        t=healthy.t,
        healthy_i_abc=healthy.i_abc,
        condition_i_abc=condition.i_abc,
        healthy_i_qs=healthy.i_qs,
        healthy_i_ds=healthy.i_ds,
        condition_i_qs=condition.i_qs,
        condition_i_ds=condition.i_ds,
        healthy_tau_e=healthy.tau_e,
        condition_tau_e=condition.tau_e,
        healthy_speed_rpm=healthy.speed_rpm,
        condition_speed_rpm=condition.speed_rpm,
        healthy_slip=healthy.slip,
        condition_slip=condition.slip,
    )
    (out / "fault_02_increased_mechanical_load_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=False)
    for index, name in enumerate(("a", "b", "c")):
        axes[0].plot(healthy.t, healthy.i_abc[index], ls="--", alpha=0.75, label=f"healthy i_{name}")
        axes[0].plot(condition.t, condition.i_abc[index], label=f"condition i_{name}")
    axes[0].set_title("Phase currents: healthy vs +50% load")
    axes[0].set_ylabel("A")
    axes[0].legend(ncol=3, fontsize=7)
    for values, label, ls in (
        (healthy.i_qs, "healthy $i_{qs}$", "--"),
        (healthy.i_ds, "healthy $i_{ds}$", "--"),
        (condition.i_qs, "condition $i_{qs}$", "-"),
        (condition.i_ds, "condition $i_{ds}$", "-"),
    ):
        axes[1].plot(healthy.t, values, ls=ls, label=label)
    axes[1].set_title("Synchronous-frame currents")
    axes[1].set_ylabel("A")
    axes[1].legend(ncol=2, fontsize=8)
    for ax, h, c, title, ylabel in (
        (axes[2], healthy.tau_e, condition.tau_e, "Electromagnetic torque", "N m"),
        (axes[3], healthy.speed_rpm, condition.speed_rpm, "Rotor speed", "r/min"),
        (axes[4], healthy.slip, condition.slip, "Slip", "p.u."),
    ):
        ax.plot(healthy.t, h, ls="--", label="healthy")
        ax.plot(condition.t, c, label="condition 02: +50% load")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    figure = out / "fault_02_increased_mechanical_load_comparison.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()