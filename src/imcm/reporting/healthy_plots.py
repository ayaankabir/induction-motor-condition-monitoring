"""Static figures for the healthy start-up study. Not a dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imcm.models.fifth_order_dq import HealthySimulationResult

CAPTION = (
    "Simulated healthy reduced-order model. Literature-example parameters "
    "(not measured). Not experimental data."
)


def _caption(ax, extra: str = "") -> None:
    text = CAPTION if not extra else f"{CAPTION} {extra}"
    ax.figure.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8, wrap=True)


def save_healthy_startup_figures(
    result: HealthySimulationResult, out_dir: Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    t = result.t
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, result.speed_rpm, color="C0")
    ax.axhline(
        result.params.synchronous_speed_rpm,
        color="0.4",
        ls="--",
        lw=1,
        label="synchronous (assumed 50 Hz, 4 pole)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mechanical speed (r/min)")
    ax.set_title("Start from rest — rotor speed")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _caption(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out_dir / "healthy_speed.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, result.tau_e, label=r"$T_e$")
    ax.plot(t, result.tau_l, label=r"$T_L$ (assumed constant)", ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Electromagnetic and load torque")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _caption(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out_dir / "healthy_torque.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=False)
    n_start = int(np.searchsorted(t, min(0.08, t[-1])))
    axes[0].plot(t[:n_start], result.i_abc[0, :n_start], label=r"$i_a$")
    axes[0].plot(t[:n_start], result.i_abc[1, :n_start], label=r"$i_b$")
    axes[0].plot(t[:n_start], result.i_abc[2, :n_start], label=r"$i_c$")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Phase currents — DOL inrush window")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    n_end = int(np.searchsorted(t, max(t[-1] - 0.08, 0.0)))
    axes[1].plot(t[n_end:], result.i_abc[0, n_end:], label=r"$i_a$")
    axes[1].plot(t[n_end:], result.i_abc[1, n_end:], label=r"$i_b$")
    axes[1].plot(t[n_end:], result.i_abc[2, n_end:], label=r"$i_c$")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Phase currents — late window (near 50 Hz sinusoids)")
    axes[1].grid(True, alpha=0.3)
    _caption(axes[1])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out_dir / "healthy_currents.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, result.i_qs, label=r"$i_{qs}$")
    ax.plot(t, result.i_ds, label=r"$i_{ds}$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Synchronous-frame stator currents (Krause)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _caption(ax, "After start-up these should approach DC.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = out_dir / "healthy_i_qd.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, result.slip)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Slip (p.u.)")
    ax.set_title("Per-unit slip")
    ax.grid(True, alpha=0.3)
    _caption(ax, "Motoring with $T_L>0$ should settle at slip $>0$.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = out_dir / "healthy_slip.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    return paths
