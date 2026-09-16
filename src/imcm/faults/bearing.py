"""Simulated outer-race bearing-fault vibration signature (Fault 04).

Reduced-order, software-only condition-monitoring signature.  The mechanical
sensor channel is synthesised directly from bearing kinematics:

- A localised outer-race defect is struck once per rolling-element pass, i.e.
  at the ball-pass frequency outer race
  ``BPFO = (Nb/2) f_r (1 - (Bd/Pd) cos(phi))``.  The outer race is stationary,
  so a defect in the loaded zone produces an equal-amplitude impulse train
  with no shaft-rate modulation (the classic outer-race signature).
- The instantaneous shaft speed is the **simulated** ``omega_m(t)`` trace from
  the unchanged fifth-order dq plant, so the impact rate follows the real
  start-up speed sweep through shaft-angle integration.
- Each impact excites an assumed structural resonance (damped sinusoid) in a
  typical rolling-element bearing band, which is what envelope analysis
  demodulates.

Honesty: this is a **simulated vibration-sensor channel**, not a motor-model
fault.  The electrical plant equations are untouched, no stator-current
signature is claimed, bearing geometry is a literature-example (not measured),
and nothing here is a confirmed diagnosis of a physical machine.  There is no
experimental validation in this repository.
"""

from __future__ import annotations

from math import cos, floor, pi, radians

import numpy as np

from imcm.models.operating_scenario import BearingFaultConfig


def shaft_angle_rad(t: np.ndarray, omega_m: np.ndarray) -> np.ndarray:
    """Cumulative mechanical shaft angle from the simulated speed trace.

    Trapezoidal integration of ``omega_m`` (the plant's mechanical state), so
    the impact phase inherits the simulated start-up speed sweep exactly.
    """
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega_m, dtype=float)
    if t.ndim != 1 or omega.shape != t.shape:
        raise ValueError("t and omega_m must be one-dimensional with equal length")
    if t.size < 2:
        raise ValueError("need at least two time samples")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("t must be strictly increasing")
    increments = 0.5 * (omega[1:] + omega[:-1]) * dt
    return np.concatenate(([0.0], np.cumsum(increments)))


def impacts_per_shaft_revolution(config: BearingFaultConfig) -> float:
    """Dimensionless ``BPFO / f_r``: outer-race impacts per shaft revolution."""
    return 0.5 * config.nb_balls * (
        1.0 - config.ball_pitch_ratio * cos(radians(config.contact_angle_deg))
    )


def outer_race_impact_times_s(
    t: np.ndarray,
    omega_m: np.ndarray,
    config: BearingFaultConfig,
) -> np.ndarray:
    """Times at which rolling elements strike the outer-race defect.

    Impacts occur at equal increments of **shaft angle** (BPFO is a fixed
    multiple of shaft speed), located by inverting the cumulative shaft angle
    from the simulated speed trace.  Deterministic and noise-free.
    """
    theta = shaft_angle_rad(t, omega_m)
    per_rev = impacts_per_shaft_revolution(config)
    if per_rev <= 0.0:
        raise ValueError("outer-race impacts per revolution must be positive")
    step = 2.0 * pi / per_rev
    k_max = int(floor(theta[-1] / step))
    if k_max < 1:
        return np.empty(0, dtype=float)
    targets = np.arange(1, k_max + 1) * step
    # theta is non-decreasing, so np.interp linearly inverts theta(t).
    return np.interp(targets, theta, t)


def outer_race_vibration(
    t: np.ndarray,
    omega_m: np.ndarray,
    config: BearingFaultConfig,
    *,
    enabled: bool,
) -> np.ndarray:
    """Simulated acceleration channel (m/s^2) for the bearing case.

    ``enabled=False`` (or a zero amplitude) returns an identically zero
    channel: the reduced-order model contains **no** baseline vibration
    source (no slotting, no noise), so the healthy bearing channel is exactly
    zero by construction.  ``enabled=True`` superimposes the damped-resonance
    response of one impact per BPFO event.  Fully deterministic.
    """
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega_m, dtype=float)
    if t.ndim != 1 or omega.shape != t.shape:
        raise ValueError("t and omega_m must be one-dimensional with equal length")
    if not enabled or config.amplitude_m_s2 == 0.0:
        return np.zeros_like(t)
    dt = float(np.mean(np.diff(t)))
    if config.resonance_hz >= 0.5 / dt:
        raise ValueError(
            "bearing resonance_hz must stay below the Nyquist frequency of "
            "the simulation output grid"
        )
    impact_times = outer_race_impact_times_s(t, omega, config)
    vibration = np.zeros_like(t)
    omega_res = 2.0 * pi * config.resonance_hz
    tau = config.resonance_decay_s
    for t_k in impact_times:
        mask = t >= t_k
        local = t[mask] - t_k
        vibration[mask] += (
            config.amplitude_m_s2 * np.exp(-local / tau) * np.sin(omega_res * local)
        )
    return vibration