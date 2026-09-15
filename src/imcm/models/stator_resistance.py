"""Phase-specific stator resistance projected into the locked Krause qd frame."""

from __future__ import annotations

import numpy as np


def phase_resistances_abc(
    base_resistance_ohm: float,
    *,
    enabled: bool,
    multipliers_abc: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return physical per-phase resistances for a scenario configuration."""
    if base_resistance_ohm <= 0.0:
        raise ValueError("base stator resistance must be positive")
    if not enabled:
        return (base_resistance_ohm, base_resistance_ohm, base_resistance_ohm)
    values = tuple(base_resistance_ohm * factor for factor in multipliers_abc)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("phase resistances must be three positive values")
    return values


def resistance_matrix_qd(theta: float, resistances_abc: tuple[float, float, float]) -> np.ndarray:
    """Project ``diag(Ra, Rb, Rc)`` into the zero-sequence-free Krause qd space."""
    r_abc = np.asarray(resistances_abc, dtype=float)
    if r_abc.shape != (3,) or np.any(r_abc <= 0.0):
        raise ValueError("resistances_abc must contain three positive values")
    if r_abc[0] == r_abc[1] == r_abc[2]:
        return np.eye(2) * r_abc[0]
    shifts = np.array((0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0))
    c = np.cos(theta + shifts)
    s = np.sin(theta + shifts)
    # K_qd diag(Rabc) K_qd^{-1}, restricted to i0 = 0.
    return (2.0 / 3.0) * np.array(
        ((np.sum(r_abc * c * c), np.sum(r_abc * c * s)),
         (np.sum(r_abc * s * c), np.sum(r_abc * s * s)))
    )


def resistive_drop_qd(
    i_qs: float,
    i_ds: float,
    theta: float,
    resistances_abc: tuple[float, float, float],
) -> tuple[float, float]:
    """Return the phase-resistance voltage drop in the synchronous qd frame."""
    drop = resistance_matrix_qd(theta, resistances_abc) @ np.array((i_qs, i_ds))
    return float(drop[0]), float(drop[1])
