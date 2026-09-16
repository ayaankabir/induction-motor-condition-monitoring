"""Run matched healthy and Fault 05 (rotor electrical asymmetry) cases.

Fault 05 is a documented **rotor electrical asymmetry proxy** for
broken-bar-related behaviour: the stator-referred cage resistance is split
between two rotor-frame axes, ``R_r (1 +/- severity)``, whose synchronous-frame
projection modulates at twice the reference slip angle.  This produces
stator-current components near the classic broken-bar sideband locations
``f_s (1 -/+ 2 s)``.  It is **not** a physically complete or bar-resolved
electromagnetic broken-bar model, it is not severity-calibrated, and nothing
here is experimentally validated or a real-machine diagnosis.  All traces are
simulated with the unchanged fifth-order dq plant; only the rotor resistance
representation differs from the healthy baseline.
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
    fault_05_rotor_asymmetry_scenario,
    first_milestone_scenario,
)
from imcm.validation.power_balance import power_balance_window
from imcm.validation.rotor_asymmetry_metrics import (
    compare_rotor_asymmetry_cases,
    modulation_frequency_hz,
    sideband_frequencies_hz,
)

# The proxy sidebands sit only 2*s*f_s ~ 2.4 Hz from the 50 Hz carrier, so the
# matched runs use a 3 s window (bin spacing 1/3 Hz) to resolve them.  Both
# cases share identical solver settings; only the scenario differs.
T_END_S = 3.0
METRIC_WINDOW_START_S = 2.0


def main() -> None:
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(t_end=T_END_S, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    healthy = simulate_healthy(scenario=first_milestone_scenario(), **kwargs)
    fault = simulate_healthy(scenario=fault_05_rotor_asymmetry_scenario(), **kwargs)
    metrics = compare_rotor_asymmetry_cases(
        healthy, fault, t_start=METRIC_WINDOW_START_S
    )
    summary = {
        "label": "simulated_fault_05_rotor_asymmetry_proxy_comparison",
        "condition_kind": "rotor_electrical_asymmetry_proxy_not_complete_broken_bar_model",
        "parameter_provenance": healthy.params.provenance,
        "healthy_metadata": healthy.metadata,
        "fault_metadata": fault.metadata,
        "rotor_asymmetry_config": asdict(fault.scenario.rotor_asymmetry),
        "metrics": asdict(metrics),
        "healthy_power_balance_residual_rel": power_balance_window(
            healthy, t_start=METRIC_WINDOW_START_S
        ).residual_rel,
        "fault_power_balance_residual_rel": power_balance_window(
            fault, t_start=METRIC_WINDOW_START_S
        ).residual_rel,
        "note": (
            "Simulated rotor electrical asymmetry proxy for broken-bar-related "
            "behaviour. Not a physically complete or bar-resolved broken-bar "
            "model; not severity-calibrated; simulation-only; no experimental "
            "validation and no real-machine diagnosis."
        ),
    }
    np.savez_compressed(
        out / "fault_05_rotor_asymmetry.npz",
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
    (out / "fault_05_rotor_asymmetry_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Comparison figure: late-window phase-a spectrum around the fundamental
    # (sidebands visible), synchronous-frame modulation tone, torque, speed.
    mask = healthy.t >= METRIC_WINDOW_START_S
    t_win = healthy.t[mask]
    fs_hz = 1.0 / float(np.mean(np.diff(t_win)))

    def amp(x: np.ndarray) -> np.ndarray:
        window = np.hanning(x.size)
        return 2.0 * np.abs(np.fft.rfft((x - np.mean(x)) * window)) / np.sum(window)

    freq = np.fft.rfftfreq(t_win.size, d=1.0 / fs_hz)
    f_lo, f_hi = sideband_frequencies_hz(
        metrics.fundamental_hz, metrics.reference_slip
    )
    f_mod = modulation_frequency_hz(metrics.fundamental_hz, metrics.reference_slip)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=False)
    axes[0].plot(freq, amp(healthy.i_abc[0][mask]), ls="--", lw=0.8, label="healthy $i_a$")
    axes[0].plot(freq, amp(fault.i_abc[0][mask]), lw=0.8, label="Fault 05 $i_a$")
    axes[0].axvline(f_lo, color="tab:red", ls=":", lw=1.0, label=f"$f_s(1-2s)$ = {f_lo:.2f} Hz")
    axes[0].axvline(f_hi, color="tab:orange", ls=":", lw=1.0, label=f"$f_s(1+2s)$ = {f_hi:.2f} Hz")
    axes[0].set_xlim(0.0, 2.5 * metrics.fundamental_hz)
    axes[0].set_ylim(0.0, 1.5 * metrics.phase_a_fundamental_fault_a)
    axes[0].set_title(
        "Phase-a spectrum, late window (proxy sidebands near $f_s(1\\pm2s)$; "
        "simulation-only, not a bar-resolved model)"
    )
    axes[0].set_ylabel("Amplitude (A)")
    axes[0].legend(fontsize=8)
    axes[1].plot(freq, amp(healthy.i_qs[mask]), ls="--", lw=0.8, label="healthy $i_{qs}$")
    axes[1].plot(freq, amp(fault.i_qs[mask]), lw=0.8, label="Fault 05 $i_{qs}$")
    axes[1].axvline(f_mod, color="tab:red", ls=":", lw=1.0, label=f"$2sf_s$ = {f_mod:.2f} Hz")
    axes[1].set_xlim(0.0, 6.0 * f_mod)
    axes[1].set_title("Synchronous-frame $i_{qs}$ modulation tone at $2 s f_s$")
    axes[1].set_ylabel("Amplitude (A)")
    axes[1].legend(fontsize=8)
    axes[2].plot(healthy.t, healthy.tau_e, ls="--", lw=0.6, label="healthy")
    axes[2].plot(fault.t, fault.tau_e, lw=0.6, label="Fault 05")
    axes[2].set_title("Electromagnetic torque (slip-frequency ripple from the proxy)")
    axes[2].set_ylabel("N m")
    axes[2].legend(fontsize=8)
    axes[3].plot(healthy.t, healthy.speed_rpm, ls="--", label="healthy")
    axes[3].plot(fault.t, fault.speed_rpm, label="Fault 05")
    axes[3].set_title("Rotor speed (mean operating point preserved by the proxy)")
    axes[3].set_ylabel("r/min")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figure = out / "fault_05_rotor_asymmetry_comparison.png"
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()