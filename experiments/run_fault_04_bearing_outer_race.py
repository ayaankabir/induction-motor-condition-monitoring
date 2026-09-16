"""Run matched healthy and Fault 04 (bearing outer-race, BPFO) cases.

Fault 04 is a simulated rolling-element bearing outer-race fault expressed as
a **vibration-sensor channel**, not a motor-model fault.  The electrical plant
(supply, load, parameters, stator resistance, Park convention) is identical to
the approved healthy baseline, so the stator currents are bit-for-bit healthy;
the fault signature lives on the simulated accelerometer channel at the BPFO
characteristic frequency.  Bearing geometry is a published 6205-series
literature example, not a measured bearing.  All outputs are simulated; this
is not a confirmed real-machine diagnosis and there is no experimental
validation in this repository.
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

from imcm.faults.bearing import outer_race_vibration
from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    fault_04_bearing_outer_race_scenario,
    first_milestone_scenario,
)
from imcm.validation.bearing_metrics import compare_bearing_cases
from imcm.validation.power_balance import power_balance_window


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    healthy = simulate_healthy(scenario=first_milestone_scenario(), **kwargs)
    fault = simulate_healthy(scenario=fault_04_bearing_outer_race_scenario(), **kwargs)

    # Simulated accelerometer channels on the shared output grid.  The healthy
    # channel is identically zero: the reduced-order model has no baseline
    # vibration source (no slotting, no noise) by construction.
    healthy_vibration = outer_race_vibration(
        healthy.t, healthy.omega_m, healthy.scenario.bearing_fault,
        enabled=healthy.scenario.bearing_fault.enabled,
    )
    fault_vibration = outer_race_vibration(
        fault.t, fault.omega_m, fault.scenario.bearing_fault,
        enabled=fault.scenario.bearing_fault.enabled,
    )
    metrics = compare_bearing_cases(
        healthy.t, healthy_vibration, fault_vibration, healthy.omega_m,
        fault.scenario.bearing_fault,
    )
    electrical_unchanged = all(
        np.array_equal(getattr(healthy, name), getattr(fault, name))
        for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc", "v_abc")
    )
    summary = {
        "label": "simulated_fault_04_bearing_outer_race_bpfo_comparison",
        "condition_kind": "simulated_bearing_vibration_signature_not_motor_model_fault",
        "parameter_provenance": healthy.params.provenance,
        "bearing_geometry_provenance": "literature_example_6205_series",
        "healthy_metadata": healthy.metadata,
        "fault_metadata": fault.metadata,
        "bearing_config": asdict(fault.scenario.bearing_fault),
        "metrics": metrics.asdict(),
        "electrical_traces_identical_to_healthy": electrical_unchanged,
        "healthy_power_balance_residual_rel": power_balance_window(healthy, t_start=0.8).residual_rel,
        "fault_power_balance_residual_rel": power_balance_window(fault, t_start=0.8).residual_rel,
        "note": (
            "Simulated outer-race bearing fault on a separate vibration channel "
            "at BPFO. Electrical motor model unchanged; simulation-only; not a "
            "confirmed real-machine diagnosis."
        ),
    }
    np.savez_compressed(
        out / "fault_04_bearing_outer_race.npz",
        t=healthy.t,
        healthy_vibration_m_s2=healthy_vibration,
        fault_vibration_m_s2=fault_vibration,
        healthy_omega_m=healthy.omega_m,
        fault_omega_m=fault.omega_m,
        healthy_i_abc=healthy.i_abc,
        fault_i_abc=fault.i_abc,
    )
    (out / "fault_04_bearing_outer_race_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    axes[0].plot(fault.t, fault.speed_rpm, label="shaft speed (simulated)")
    axes[0].set_title("Simulated shaft speed (identical for healthy and Fault 04)")
    axes[0].set_ylabel("r/min")
    axes[0].legend(fontsize=8)
    axes[1].plot(fault.t, fault_vibration, lw=0.4, label="Fault 04 vibration")
    axes[1].plot(healthy.t, healthy_vibration, lw=1.2, label="healthy vibration (zero by construction)")
    axes[1].set_title("Simulated bearing vibration channel (start-up transient visible)")
    axes[1].set_ylabel("m/s$^2$")
    axes[1].legend(fontsize=8)
    from imcm.processing.envelope import envelope_spectrum

    fs = 1.0 / float(np.mean(np.diff(healthy.t[healthy.t >= 0.5])))
    spec = envelope_spectrum(
        fault_vibration[fault.t >= 0.5],
        fs,
        band=(
            fault.scenario.bearing_fault.resonance_hz - 500.0,
            fault.scenario.bearing_fault.resonance_hz + 500.0,
        ),
    )
    axes[2].plot(spec.freq_hz, spec.amplitude, lw=0.8)
    bpfo = metrics.bpfo_hz
    axes[2].axvline(bpfo, color="tab:red", ls="--", lw=1.0, label=f"BPFO = {bpfo:.1f} Hz")
    axes[2].axvline(2.0 * bpfo, color="tab:orange", ls=":", lw=1.0, label=f"2 x BPFO = {2 * bpfo:.1f} Hz")
    axes[2].set_xlim(0.0, 6.0 * bpfo)
    axes[2].set_title("Envelope spectrum of the Fault 04 vibration channel (0.5-1.0 s window)")
    axes[2].set_ylabel("Envelope amplitude (m/s$^2$)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure = out / "fault_04_bearing_outer_race_comparison.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()