"""Run matched healthy and Fault 01 simulated stator-resistance cases."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
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
    fault_01_phase_a_resistance_imbalance_scenario,
    first_milestone_scenario,
)
from imcm.reporting.provenance import ProvenanceRecord
from imcm.reporting.run_config import RunConfig, emit_run_artifacts
from imcm.validation.fault_metrics import compare_stator_resistance_cases
from imcm.validation.power_balance import power_balance_window


def parameter_hash(params) -> str:
    """Return a deterministic SHA-256 hash of the motor parameters."""
    payload = asdict(params)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        t_end=1.0,
        max_step=1.0e-4,
        output_dt=1.0e-4,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    healthy = simulate_healthy(
        scenario=first_milestone_scenario(),
        **kwargs,
    )
    fault = simulate_healthy(
        scenario=fault_01_phase_a_resistance_imbalance_scenario(),
        **kwargs,
    )

    metrics = compare_stator_resistance_cases(healthy, fault)

    summary = {
        "label": "simulated_fault_01_comparison",
        "parameter_provenance": healthy.params.provenance,
        "healthy_metadata": healthy.metadata,
        "fault_metadata": fault.metadata,
        "metrics": asdict(metrics),
        "healthy_power_balance_residual_rel": (
            power_balance_window(healthy, t_start=0.8).residual_rel
        ),
        "fault_power_balance_residual_rel": (
            power_balance_window(fault, t_start=0.8).residual_rel
        ),
        "note": (
            "Controlled simulated high-resistance phase-A proxy. Not experimental "
            "validation and not an inter-turn-short model."
        ),
    }

    np.savez_compressed(
        out / "fault_01_stator_resistance_imbalance.npz",
        t=healthy.t,
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

    summary_path = out / "fault_01_stator_resistance_imbalance_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    late = healthy.t >= 0.8

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=False)

    for index, name in enumerate(("a", "b", "c")):
        axes[0].plot(
            healthy.t[late],
            healthy.i_abc[index, late],
            ls="--",
            alpha=0.75,
            label=f"healthy i_{name}",
        )
        axes[0].plot(
            fault.t[late],
            fault.i_abc[index, late],
            label=f"fault i_{name}",
        )

    axes[0].set_title("Phase currents: late matched window")
    axes[0].set_ylabel("A")
    axes[0].legend(ncol=3, fontsize=7)

    for values, label, ls in (
        (healthy.i_qs, "healthy $i_{qs}$", "--"),
        (healthy.i_ds, "healthy $i_{ds}$", "--"),
        (fault.i_qs, "fault $i_{qs}$", "-"),
        (fault.i_ds, "fault $i_{ds}$", "-"),
    ):
        axes[1].plot(
            healthy.t[late],
            values[late],
            ls=ls,
            label=label,
        )

    axes[1].set_title("Synchronous-frame currents: late matched window")
    axes[1].set_ylabel("A")
    axes[1].legend(ncol=2, fontsize=8)

    for ax, h, f, title, ylabel in (
        (
            axes[2],
            healthy.tau_e,
            fault.tau_e,
            "Electromagnetic torque",
            "N m",
        ),
        (
            axes[3],
            healthy.speed_rpm,
            fault.speed_rpm,
            "Rotor speed",
            "r/min",
        ),
        (
            axes[4],
            healthy.slip,
            fault.slip,
            "Slip",
            "p.u.",
        ),
    ):
        ax.plot(healthy.t, h, ls="--", label="healthy")
        ax.plot(
            fault.t,
            f,
            label="fault 01: phase A +10% R",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    figure = out / "fault_01_stator_resistance_imbalance_comparison.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)

    output_paths = [
        out / "fault_01_stator_resistance_imbalance.npz",
        summary_path,
        figure,
    ]

    manifest_path = (
        out / "fault_01_stator_resistance_imbalance_provenance.json"
    )
    run_config_path = (
        out / "fault_01_stator_resistance_imbalance_run_config.json"
    )

    output_rel = [str(path.relative_to(ROOT)) for path in output_paths]
    output_rel.append(str(manifest_path.relative_to(ROOT)))

    solver_settings = {
        "method": "RK45",
        "t_end": kwargs["t_end"],
        "max_step": kwargs["max_step"],
        "output_dt": kwargs["output_dt"],
        "rtol": kwargs["rtol"],
        "atol": kwargs["atol"],
    }

    provenance = ProvenanceRecord.create(
        dataset_label="simulated",
        parameter_hash=parameter_hash(fault.params),
        parameter_source=fault.params.provenance,
        solver_settings=solver_settings,
        initial_condition={
            "description": (
                "start from rest; matched healthy and Fault 01 "
                "stator-resistance-imbalance simulations"
            ),
        },
        output_files=output_rel,
    )

    run_config = RunConfig.create(
        condition_id="fault_01",
        condition_kind="stator_resistance_imbalance_proxy_not_turn_fault",
        scenario_name=fault.scenario.name,
        park_convention=fault.scenario.park_convention,
        motor_parameters=asdict(fault.params),
        solver_settings=solver_settings,
        initial_condition={
            "state": "rest",
            "description": "start from rest; matched healthy and fault runs",
        },
        analysis_windows=[
            {"name": "late_matched", "start_s": 0.8, "end_s": kwargs["t_end"]},
        ],
        condition_parameters={
            "type": "stator_resistance_imbalance",
            "stator_resistance_case": fault.scenario.stator_resistance.label,
            "stator_resistance_multipliers_abc": list(
                fault.scenario.stator_resistance.multipliers_abc
            ),
            "stator_resistances_abc_ohm": list(
                fault.metadata["stator_resistances_abc_ohm"]
            ),
        },
        outputs=output_rel,
        provenance_manifest={"path": str(manifest_path.relative_to(ROOT))},
    )

    emit_run_artifacts(
        run_config,
        provenance,
        run_config_path=run_config_path,
        manifest_path=manifest_path,
        root=ROOT,
    )

    print(json.dumps(summary, indent=2))
    print("Wrote:")
    for path in [*output_paths, manifest_path, run_config_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()