"""Matched healthy/Fault-05 metrics for the rotor-asymmetry proxy.

All quantities are computed from **simulated** traces of the reduced-order
model.  The Fault 05 proxy is a documented rotor electrical asymmetry, not a
physically complete broken-bar model, so these metrics characterise the proxy
signature; they are not measurements and not a real-machine diagnosis.

Signature logic
---------------
The proxy modulates the synchronous-frame rotor resistance at
``2 s_ref f_s``, so the healthy DC synchronous-frame currents gain a tone at
that frequency, and the stator (phase) currents gain components near the
classic broken-bar sideband locations ``f_s (1 -/+ 2 s_ref)``.  Spectral
amplitudes use a Hann-windowed FFT (``2|X| / sum(w)`` normalisation, as in
``imcm.processing.envelope``) so a tone's peak bin reads its true amplitude.

Resolution caveat: resolving stator sidebands only ``2 s f_s ~ 2.4 Hz`` away
from the 50 Hz carrier requires a window long compared to ``1/(2 s f_s)``.
Use simulations of several seconds (``t_end >= 3 s``) for the sideband
metrics; short windows remain valid for the synchronous-frame tone after the
carrier has been removed by mean subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imcm.analysis.spectral_analysis import (
    compute_amplitude_spectrum,
    find_peak_amplitude,
)
from imcm.models.fifth_order_dq import HealthySimulationResult
from imcm.validation.fault_metrics import LATE_WINDOW_START_S, torque_ripple_rms

DEFAULT_SIDEBAND_HALF_WIDTH_HZ = 1.0


def modulation_frequency_hz(fundamental_hz: float, reference_slip: float) -> float:
    """Proxy modulation rate ``2 s_ref f_s`` of the synchronous-frame rotor resistance."""
    return 2.0 * reference_slip * fundamental_hz


def sideband_frequencies_hz(
    fundamental_hz: float, reference_slip: float
) -> tuple[float, float]:
    """Classic broken-bar sideband locations ``f_s (1 -/+ 2 s_ref)`` implied by the proxy."""
    offset = modulation_frequency_hz(fundamental_hz, reference_slip)
    return fundamental_hz - offset, fundamental_hz + offset


def spectrum_peak_amplitude(
    t: np.ndarray,
    x: np.ndarray,
    center_hz: float,
    half_width_hz: float,
) -> float:
    """Peak Hann-windowed FFT amplitude within ``+/- half_width_hz`` of a target.

    ``x`` is mean-removed before windowing (removes the DC carrier).  The
    ``2|X| / sum(w)`` normalisation makes an isolated tone's peak bin read its
    true amplitude.  Deterministic; no experimental analysis is claimed.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.shape != t.shape:
        raise ValueError("t and x must be one-dimensional with equal length")
    if t.size < 8:
        raise ValueError("spectral window is too short")
    if half_width_hz <= 0.0:
        raise ValueError("half_width_hz must be positive")
    dt = float(np.mean(np.diff(t)))
    fs_hz = 1.0 / dt
    if not (
        0.0 < center_hz - half_width_hz and center_hz + half_width_hz < 0.5 * fs_hz
    ):
        raise ValueError("search band must lie strictly inside (0, Nyquist)")
    spec = compute_amplitude_spectrum(x, fs_hz, remove_mean=True)
    return find_peak_amplitude(spec.freq_hz, spec.amplitude, center_hz, half_width_hz)


@dataclass(frozen=True)
class RotorAsymmetryComparisonMetrics:
    """Matched healthy/Fault-05 proxy metrics (simulation evidence only)."""

    severity: float
    reference_slip: float
    fundamental_hz: float
    modulation_frequency_hz: float
    lower_sideband_hz: float
    upper_sideband_hz: float
    sideband_half_width_hz: float
    phase_a_fundamental_healthy_a: float
    phase_a_fundamental_fault_a: float
    lower_sideband_healthy_a: float
    lower_sideband_fault_a: float
    upper_sideband_healthy_a: float
    upper_sideband_fault_a: float
    sync_modulation_healthy_a: float
    sync_modulation_fault_a: float
    torque_ripple_healthy_nm: float
    torque_ripple_fault_nm: float
    final_speed_difference_rpm: float
    final_slip_difference: float


def _window_mask(
    result: HealthySimulationResult, t_start: float, t_stop: float
) -> np.ndarray:
    if t_stop <= t_start:
        raise ValueError("metric-window stop must be greater than start")
    mask = (result.t >= t_start) & (result.t < t_stop)
    if np.count_nonzero(mask) < 8:
        raise ValueError("metric window is too short")
    return mask


def compare_rotor_asymmetry_cases(
    healthy: HealthySimulationResult,
    fault: HealthySimulationResult,
    *,
    t_start: float = LATE_WINDOW_START_S,
    t_stop: float | None = None,
    sideband_half_width_hz: float = DEFAULT_SIDEBAND_HALF_WIDTH_HZ,
) -> RotorAsymmetryComparisonMetrics:
    """Compare matched healthy and Fault 05 runs on a shared time grid.

    The stator sideband amplitudes need a window long compared to
    ``1/(2 s_ref f_s)``; pass ``t_stop`` from a several-second simulation for
    meaningful carrier leakage suppression.  Simulation evidence only.
    """
    if not np.array_equal(healthy.t, fault.t):
        raise ValueError("healthy and fault cases must share the same output time grid")
    config = fault.scenario.rotor_asymmetry
    if not config.enabled:
        raise ValueError("fault case must enable the rotor-asymmetry proxy")
    s_ref = config.reference_slip
    if s_ref is None:
        raise ValueError("reference slip must be resolved before metric extraction")
    fundamental_hz = fault.scenario.supply.frequency_hz
    if healthy.scenario.supply.frequency_hz != fundamental_hz:
        raise ValueError("healthy and fault cases must share the supply frequency")
    t_stop = float(fault.t[-1]) if t_stop is None else t_stop
    mask = _window_mask(fault, t_start, t_stop)
    t_win = fault.t[mask]
    f_mod = modulation_frequency_hz(fundamental_hz, s_ref)
    f_lo, f_hi = sideband_frequencies_hz(fundamental_hz, s_ref)

    phase_a_fundamental = (
        spectrum_peak_amplitude(
            t_win, healthy.i_abc[0][mask], fundamental_hz, sideband_half_width_hz
        ),
        spectrum_peak_amplitude(
            t_win, fault.i_abc[0][mask], fundamental_hz, sideband_half_width_hz
        ),
    )
    lower = (
        spectrum_peak_amplitude(
            t_win, healthy.i_abc[0][mask], f_lo, sideband_half_width_hz
        ),
        spectrum_peak_amplitude(t_win, fault.i_abc[0][mask], f_lo, sideband_half_width_hz),
    )
    upper = (
        spectrum_peak_amplitude(
            t_win, healthy.i_abc[0][mask], f_hi, sideband_half_width_hz
        ),
        spectrum_peak_amplitude(t_win, fault.i_abc[0][mask], f_hi, sideband_half_width_hz),
    )
    sync = (
        max(
            spectrum_peak_amplitude(t_win, healthy.i_qs[mask], f_mod, sideband_half_width_hz),
            spectrum_peak_amplitude(t_win, healthy.i_ds[mask], f_mod, sideband_half_width_hz),
        ),
        max(
            spectrum_peak_amplitude(t_win, fault.i_qs[mask], f_mod, sideband_half_width_hz),
            spectrum_peak_amplitude(t_win, fault.i_ds[mask], f_mod, sideband_half_width_hz),
        ),
    )
    return RotorAsymmetryComparisonMetrics(
        severity=config.severity,
        reference_slip=s_ref,
        fundamental_hz=fundamental_hz,
        modulation_frequency_hz=f_mod,
        lower_sideband_hz=f_lo,
        upper_sideband_hz=f_hi,
        sideband_half_width_hz=sideband_half_width_hz,
        phase_a_fundamental_healthy_a=phase_a_fundamental[0],
        phase_a_fundamental_fault_a=phase_a_fundamental[1],
        lower_sideband_healthy_a=lower[0],
        lower_sideband_fault_a=lower[1],
        upper_sideband_healthy_a=upper[0],
        upper_sideband_fault_a=upper[1],
        sync_modulation_healthy_a=sync[0],
        sync_modulation_fault_a=sync[1],
        torque_ripple_healthy_nm=torque_ripple_rms(
            healthy, t_start=t_start, t_stop=t_stop
        ),
        torque_ripple_fault_nm=torque_ripple_rms(fault, t_start=t_start, t_stop=t_stop),
        final_speed_difference_rpm=float(fault.speed_rpm[-1] - healthy.speed_rpm[-1]),
        final_slip_difference=float(fault.slip[-1] - healthy.slip[-1]),
    )