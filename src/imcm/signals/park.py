"""Classical Krause qd0 transforms (amplitude-invariant, factor 2/3).

This is not a power-invariant (orthonormal) transform. Torque and three-phase
power must use the matching 3/2 factors. See docs/modeling-plan.md.
"""

from __future__ import annotations

import numpy as np

CONVENTION_NAME = "krause_classical_2_3"


def abc_to_qd0(f_abc: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    """Transform phase quantities ``(a, b, c)`` to Krause ``(q, d, 0)``.

    Parameters
    ----------
    f_abc:
        Array of shape ``(3,)`` or ``(3, n)``.
    theta:
        Electrical angle of the q-axis relative to phase a (radians).
    """
    f_abc = np.asarray(f_abc, dtype=float)
    if f_abc.shape[0] != 3:
        raise ValueError("f_abc must have leading dimension 3 (a, b, c)")
    th = np.asarray(theta, dtype=float)
    two_pi_3 = 2.0 * np.pi / 3.0
    k = 2.0 / 3.0
    fa, fb, fc = f_abc[0], f_abc[1], f_abc[2]
    fq = k * (fa * np.cos(th) + fb * np.cos(th - two_pi_3) + fc * np.cos(th + two_pi_3))
    fd = k * (fa * np.sin(th) + fb * np.sin(th - two_pi_3) + fc * np.sin(th + two_pi_3))
    f0 = (fa + fb + fc) / 3.0
    return np.stack((fq, fd, f0), axis=0)


def qd0_to_abc(f_qd0: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    """Inverse Krause transform from ``(q, d, 0)`` to ``(a, b, c)``."""
    f_qd0 = np.asarray(f_qd0, dtype=float)
    if f_qd0.shape[0] != 3:
        raise ValueError("f_qd0 must have leading dimension 3 (q, d, 0)")
    th = np.asarray(theta, dtype=float)
    two_pi_3 = 2.0 * np.pi / 3.0
    fq, fd, f0 = f_qd0[0], f_qd0[1], f_qd0[2]
    fa = fq * np.cos(th) + fd * np.sin(th) + f0
    fb = fq * np.cos(th - two_pi_3) + fd * np.sin(th - two_pi_3) + f0
    fc = fq * np.cos(th + two_pi_3) + fd * np.sin(th + two_pi_3) + f0
    return np.stack((fa, fb, fc), axis=0)


def balanced_phase_voltages(t: np.ndarray | float, omega_e: float, phase_peak_v: float) -> np.ndarray:
    """Ideal balanced sinusoids: va = Vs cos(ωe t), 120° apart."""
    t = np.asarray(t, dtype=float)
    two_pi_3 = 2.0 * np.pi / 3.0
    va = phase_peak_v * np.cos(omega_e * t)
    vb = phase_peak_v * np.cos(omega_e * t - two_pi_3)
    vc = phase_peak_v * np.cos(omega_e * t + two_pi_3)
    return np.stack((va, vb, vc), axis=0)
