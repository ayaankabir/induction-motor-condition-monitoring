"""Run matched healthy and Fault 03 (supply-voltage unbalance) cases.

Fault 03 is a controlled supply-side confounder: the applied three-phase supply
voltages are per-phase scaled (default phase C at 0.9 of the balanced phase
peak) while keeping the locked 120-degree spacing.  Motor equations, motor
parameters, stator resistance, load torque, Park convention, and supply
frequency match the approved healthy baseline.  It is not a winding fault and
not a confirmed internal motor fault.  All traces are simulated.
"""

from __future__ import annotations

from dataclasses import asdict
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
    fault_03_supply_voltage_unbalance_scenario,
    first_milestone_scenario,
)
from imcm.validation.fault_metrics import (
    compare_supply_unbalance_cases,
    supply_voltage_unbalance_pct,
)
from imcm.validation.power_balance import power_balance_window


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    healthy = simulate_healthy(scenario=first_milestone_scenario(), **kwargs)
    fault = simulate_healthy(scenario=fault_03_supply_voltage_unbalance_scenario(), **kwargs)
    metrics = compare_supply_unbalance_cases(healthy, fault)
    summary = {
        "label": "simulated_fault_03_supply_voltage_unbalance_comparison",
        "condition_kind": "supply_voltage_confounder_not_internal_fault",
        "parameter_provenance": healthy.params.provenance,
        "healthy_metadata": healthy.metadata,
        "fault_metadata": fault.metadata,
        "supply_phase_peak_abc_v": fault.scenario.supply.phase_peak_abc_v,
        "supply_voltage_multipliers_abc": fault.scenario.supply.voltage_unbalance.multipliers_abc,
        "metrics": asdict(metrics),
        "healthy_supply_voltage_unbalance_pct": supply_voltage_unbalance_pct(healthy),
        "fault_supply_voltage_unbalance_pct": supply_voltage_unbalance_pct(fault),
        "healthy_power_balance_residual_rel": power_balance_window(healthy, t_start=0.8).residual_rel,
        "fault_power_balance_residual_rel": power_balance_window(fault, t_start=0.8).residual_rel,
        "note": (
            "Controlled supply-voltage unbalance (phase C at 0.9 p.u.). "
            "Supply-side confounder; simulation-only; not a confirmed internal "
            "motor fault and not winding damage."
        ),
    }
    np.savez_compressed(
        out / "fault_03_supply_voltage_unbalance.npz",
        t=healthy.t,
        healthy_v_abc=healthy.v_abc,
        fault_v_abc=fault.v_abc,
        healthy_v_qs=healthy.v_qs,
        healthy_v_ds=healthy.v_ds,
        fault_v_qs=fault.v_qs,
        fault_v_ds=fault.v_ds,
        healthy_i_abc=healthy.i_abc,
        fault_i_abc=fault.i_abc,
        healthy_i_qs=healthy.i_qs,
        healthy_i_ds=healthy.i_ds,
        fault_i_qs=fault.i_qs,
        fault_i_ds=fault.i_ds,
        healthy_tau_e=healthy.tau_e,
        fault_tau_e=fault.tau_e,
        healthy_speed_rpm=healthy.speed_rpm,
        fault_speed_rpm=fault.speed_rpm,
        healthy_slip=healthy.slip,
        fault_slip=fault.slip,
    )
    (out / "fault_03_supply_voltage_unbalance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=False)
    for index, name in enumerate(("a", "b", "c")):
        axes[0].plot(healthy.t, healthy.v_abc[index], ls="--", alpha=0.75, label=f"healthy v_{name}")
        axes[0].plot(fault.t, fault.v_abc[index], label=f"unbalanced v_{name}")
    axes[0].set_title("Applied supply voltages: balanced vs phase-C 0.9 p.u.")
    axes[0].set_ylabel("V")
    axes[0].legend(ncol=3, fontsize=7)
    for index, name in enumerate(("a", "b", "c")):
        axes[1].plot(healthy.t, healthy.i_abc[index], ls="--", alpha=0.75, label=f"healthy i_{name}")
        axes[1].plot(fault.t, fault.i_abc[index], label=f"unbalanced i_{name}")
    axes[1].set_title("Phase currents under unbalanced supply")
    axes[1].set_ylabel("A")
    axes[1].legend(ncol=3, fontsize=7)
    for values, label, ls in (
        (healthy.i_qs, "healthy $i_{qs}$", "--"),
        (healthy.i_ds, "healthy $i_{ds}$", "--"),
        (fault.i_qs, "unbalanced $i_{qs}$", "-"),
        (fault.i_ds, "unbalanced $i_{ds}$", "-"),
    ):
        axes[2].plot(healthy.t, values, ls=ls, label=label)
    axes[2].set_title("Synchronous-frame currents")
    axes[2].set_ylabel("A")
    axes[2].legend(ncol=2, fontsize=8)
    axes[3].plot(healthy.t, healthy.tau_e, ls="--", label="healthy")
    axes[3].plot(fault.t, fault.tau_e, label="fault 03: supply unbalance")
    axes[3].set_title("Electromagnetic torque (2nd-harmonic ripple)")
    axes[3].set_ylabel("N m")
    axes[3].legend(fontsize=8)
    axes[4].plot(healthy.t, healthy.speed_rpm, ls="--", label="healthy")
    axes[4].plot(fault.t, fault.speed_rpm, label="fault 03: supply unbalance")
    axes[4].set_title("Rotor speed")
    axes[4].set_ylabel("r/min")
    axes[4].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    figure = out / "fault_03_supply_voltage_unbalance_comparison.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
