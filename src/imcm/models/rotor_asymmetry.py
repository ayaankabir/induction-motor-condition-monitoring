"""Rotor electrical asymmetry proxy (Fault 05): broken-bar-related signature.

**This is a simulation-only proxy, not a physically complete broken rotor bar
model.**  The locked fifth-order plant represents the cage by a single
stator-referred resistance ``r_r``; there are no individual bars, end-ring
segments, or bar-to-bar circuits, so a broken bar cannot be represented
geometrically.  Instead the fault is expressed as a *rotor electrical
asymmetry*: the cage resistance is split between two orthogonal rotor-frame
axes

    R_high = r_r (1 + severity)
    R_low  = r_r (1 - severity),        0 <= severity < 1

whose mean stays exactly ``r_r``.  The axes are fixed to the rotor frame,
which rotates at slip speed relative to the locked synchronous frame, so
projecting the split into the synchronous frame yields the symmetric matrix

    R_sync(2*phi) = r_r I + severity * r_r [[cos 2phi,     sin 2phi],
                                            [sin 2phi, -cos 2phi]]

with eigenvalues ``r_r (1 +/- severity)``.  Expanding ``R_sync @ i_r`` gives
the rotor-row voltage drops used by the plant ODE:

    drop_qr = r_r i_qr + severity r_r ( cos(2phi) i_qr + sin(2phi) i_dr )
    drop_dr = r_r i_dr + severity r_r ( sin(2phi) i_qr - cos(2phi) i_dr )

Because the projection of a rotor-fixed unbalance oscillates at **twice the
slip angle**, the impedance modulation runs at ``2 s f_s`` in the synchronous
frame, which corresponds to stator-current components near the classic
broken-bar sideband locations ``f_s (1 -/+ 2 s)``.

**Documented approximation.**  Advancing ``phi`` with the true slip-angle
integral ``int (omega_e - omega_r) dt`` would require a sixth ODE state and
break the locked fifth-order architecture.  The proxy therefore advances the
phase at the **constant reference slip rate** ``s_ref * omega_e``, where
``s_ref`` is solved from the healthy T-equivalent-circuit torque balance at
the scenario load (``reference_slip_for_load``).  In the late (nearly steady)
monitoring window the true slip approaches ``s_ref`` and the modulation
matches the physical slip-speed mechanism there; during the direct-on-line
start-up it does not.

**Limitations (honesty).**  This proxy is not bar-resolved and cannot claim
bar counts or a severity calibration against any physical machine.  It keeps
the *mean* rotor resistance at ``r_r``, deliberately isolating the asymmetry
effect and not modelling the small mean-resistance rise a real broken bar
also produces.  The modulation frequency is prescribed by construction, not
emergent from bar geometry.  No experimental validation exists in this
repository, and no real-machine diagnosis is claimed or implied.  See
``docs/fault_05_rotor_asymmetry.md`` for the full statement.
"""

from __future__ import annotations

from math import cos, sin, sqrt

import numpy as np

from imcm.models.operating_scenario import OperatingScenario, RotorAsymmetryConfig
from imcm.models.parameters import InductionMotorParameters

_BISECTION_ITERATIONS = 100
_BRACKET_S_HIGH = 0.5


def modulation_phase_rad(
    t: float | np.ndarray,
    config: RotorAsymmetryConfig,
    omega_e: float,
) -> float | np.ndarray:
    """Double modulation angle ``2 phi(t) = 2 (phi_0 + s_ref omega_e t)``.

    The phase advances at the constant reference slip rate because the locked
    fifth-order state vector has no slip-angle state (documented proxy
    approximation; see the module docstring).  Returns a float for scalar
    input and an ndarray for array input.
    """
    if omega_e <= 0.0:
        raise ValueError("supply electrical speed must be positive")
    s_ref = config.reference_slip
    if s_ref is None or not 0.0 < s_ref < 1.0:
        raise ValueError(
            "enabled rotor asymmetry requires reference_slip in (0, 1); resolve "
            "it with reference_slip_for_load() before integrating"
        )
    two_phi = 2.0 * (
        np.deg2rad(config.initial_phase_deg) + s_ref * omega_e * np.asarray(t, dtype=float)
    )
    return float(two_phi) if np.ndim(t) == 0 else two_phi


def rotor_asymmetry_drops(
    i_qr: float,
    i_dr: float,
    two_phi: float,
    severity: float,
    r_r_ohm: float,
) -> tuple[float, float]:
    """Rotor-row resistive drops ``(drop_qr, drop_dr)`` in the synchronous frame.

    Scalar inputs for use inside the ODE right-hand side.  With
    ``severity == 0`` the result reduces exactly to ``(r_r i_qr, r_r i_dr)``.
    Only the rotor rows change: the stator voltage equations are untouched.
    """
    if r_r_ohm <= 0.0:
        raise ValueError("rotor resistance must be positive")
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must lie in [0, 1)")
    c = cos(two_phi)
    s = sin(two_phi)
    drop_qr = r_r_ohm * i_qr + severity * r_r_ohm * (c * i_qr + s * i_dr)
    drop_dr = r_r_ohm * i_dr + severity * r_r_ohm * (s * i_qr - c * i_dr)
    return drop_qr, drop_dr


def rotor_resistance_matrix_sync(
    two_phi: float,
    severity: float,
    r_r_ohm: float,
) -> np.ndarray:
    """Synchronous-frame 2x2 rotor resistance matrix ``R_sync(2*phi)``.

    Symmetric with eigenvalues ``r_r (1 +/- severity)`` and trace ``2 r_r``:
    the mean rotor resistance is preserved by construction.
    """
    if r_r_ohm <= 0.0:
        raise ValueError("rotor resistance must be positive")
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must lie in [0, 1)")
    c = cos(two_phi)
    s = sin(two_phi)
    return r_r_ohm * np.array(
        ((1.0 + severity * c, severity * s),
         (severity * s, 1.0 - severity * c)),
        dtype=float,
    )


def rotor_copper_loss_w(
    i_qr: np.ndarray,
    i_dr: np.ndarray,
    two_phi: np.ndarray,
    severity: float,
    r_r_ohm: float,
) -> np.ndarray:
    """Instantaneous three-phase rotor copper loss ``(3/2) i_r^T R_sync i_r``.

    Array form for the power-balance identity; reduces exactly to the healthy
    expression ``(3/2) r_r (i_qr^2 + i_dr^2)`` when ``severity == 0``.
    """
    if r_r_ohm <= 0.0:
        raise ValueError("rotor resistance must be positive")
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must lie in [0, 1)")
    i_qr = np.asarray(i_qr, dtype=float)
    i_dr = np.asarray(i_dr, dtype=float)
    c = np.cos(two_phi)
    s = np.sin(two_phi)
    anisotropic = c * (i_qr**2 - i_dr**2) + 2.0 * s * i_qr * i_dr
    return 1.5 * r_r_ohm * (i_qr**2 + i_dr**2 + severity * anisotropic)


def _t_circuit_torque_nm(
    params: InductionMotorParameters,
    slip: float,
    *,
    line_line_rms_v: float,
    omega_e: float,
) -> float:
    """IEEE T-circuit electromagnetic torque at one slip (motoring region).

    Mirrors ``imcm.validation.equivalent_circuit.equivalent_circuit_point``;
    ``tests/test_fault_05_rotor_asymmetry.py`` cross-checks the two at the
    solved reference slip so the mirror cannot silently drift.
    """
    if slip <= 0.0:
        raise ValueError("slip must be positive")
    v_ph = line_line_rms_v / sqrt(3.0)
    xs = omega_e * params.l_ls
    xm = omega_e * params.l_m
    xr = omega_e * params.l_lr
    z_r = params.r_r / slip + 1j * xr
    z_m = 1j * xm
    z_par = (z_m * z_r) / (z_m + z_r)
    i_s = v_ph / (params.r_s + 1j * xs + z_par)
    i_r = i_s * z_m / (z_m + z_r)
    p_ag = 3.0 * (abs(i_r) ** 2) * (params.r_r / slip)
    return p_ag / (omega_e / params.pole_pairs)


def reference_slip_for_load(
    params: InductionMotorParameters,
    scenario: OperatingScenario,
) -> float:
    """Healthy steady-state slip from the T-circuit torque balance.

    Solves ``T_ec(s) = T_L + B (1 - s) omega_sync`` by bisection on the
    low-slip (stable motoring) branch.  This is the constant slip rate used
    to advance the proxy modulation phase; it is a documented approximation
    to the true time-varying slip, not a measured operating point.
    """
    if scenario.load.load_type != "constant_torque":
        raise NotImplementedError("reference slip requires constant load torque")
    omega_e = scenario.supply.omega_e
    omega_sync = omega_e / params.pole_pairs

    # Torque balance against load plus speed-dependent friction:
    def balance(s: float) -> float:
        tau = _t_circuit_torque_nm(
            params, s,
            line_line_rms_v=scenario.supply.line_line_rms_v,
            omega_e=omega_e,
        )
        return tau - (
            scenario.load.torque_nm + params.viscous_friction * (1.0 - s) * omega_sync
        )

    lo, hi = 1.0e-9, _BRACKET_S_HIGH
    if balance(hi) <= 0.0:
        raise RuntimeError("could not bracket the reference slip on the motoring branch")
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if balance(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)