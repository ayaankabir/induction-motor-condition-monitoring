"""Envelope-spectrum condition indicators for the simulated bearing signature.

Fault 04 features are vibration-channel quantities only.  They compare the
simulated faulty bearing channel against the exactly-zero healthy channel of
the reduced-order model.  These are **simulation evidence**, not measurements
and not a diagnosis of a physical machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi

import numpy as np

from imcm.models.operating_scenario import BearingFaultConfig
from imcm.processing.envelope import (
    amplitude_near,
    envelope_spectrum,
    peak_frequency_near,
)

DEFAULT_ANALYSIS_START_S = 0.5
DEFAULT_BAND_HALF_WIDTH_HZ = 3.0
DEFAULT_RESONANCE_BAND_HZ = 500.0
MIN_WINDOW_SAMPLES = 32


@dataclass(frozen=True)
class BearingVibrationFeatures:
    """Envelope-analysis features of one simulated bearing channel."""

    bpfo_hz: float
    shaft_hz_mean: float
    envelope_bpfo_amplitude: float
    envelope_bpfo_2x_amplitude: float
    envelope_bpfo_peak_hz: float

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BearingComparisonMetrics:
    """Healthy-versus-fault envelope features for the same simulated run.

    ``envelope_bpfo_ratio`` is ``None`` when the healthy amplitude is exactly
    zero (the reduced-order healthy channel is identically zero), which keeps
    the JSON summary strictly serialisable.
    """

    bpfo_hz: float
    shaft_hz_mean: float
    envelope_bpfo_healthy: float
    envelope_bpfo_fault: float
    envelope_bpfo_2x_healthy: float
    envelope_bpfo_2x_fault: float
    bpfo_peak_healthy_hz: float
    bpfo_peak_fault_hz: float
    envelope_bpfo_ratio: float | None

    def asdict(self) -> dict:
        return asdict(self)


def bearing_vibration_features(
    t: np.ndarray,
    vibration: np.ndarray,
    omega_m: np.ndarray,
    config: BearingFaultConfig,
    *,
    t_start: float = DEFAULT_ANALYSIS_START_S,
    band_half_width_hz: float = DEFAULT_BAND_HALF_WIDTH_HZ,
) -> BearingVibrationFeatures:
    """Envelope features for one channel over ``t >= t_start``.

    The analytic BPFO is evaluated at the **mean simulated shaft speed** over
    the analysis window, which is nearly constant there; the measured envelope
    peak must land within a few bins of that value.
    """
    t = np.asarray(t, dtype=float)
    vibration = np.asarray(vibration, dtype=float)
    omega_m = np.asarray(omega_m, dtype=float)
    if t.ndim != 1 or vibration.shape != t.shape or omega_m.shape != t.shape:
        raise ValueError("t, vibration, and omega_m must share one shape")
    mask = t >= t_start
    if np.count_nonzero(mask) < MIN_WINDOW_SAMPLES:
        raise ValueError("bearing analysis window is too short")
    t_w = t[mask]
    vib_w = vibration[mask]
    omega_w = omega_m[mask]
    fs = 1.0 / float(np.mean(np.diff(t_w)))
    shaft_hz = float(np.mean(omega_w)) / (2.0 * pi)
    bpfo_hz = config.bpfo_hz(shaft_hz)
    spec = envelope_spectrum(
        vib_w,
        fs,
        band=(
            config.resonance_hz - DEFAULT_RESONANCE_BAND_HZ,
            config.resonance_hz + DEFAULT_RESONANCE_BAND_HZ,
        ),
    )
    return BearingVibrationFeatures(
        bpfo_hz=bpfo_hz,
        shaft_hz_mean=shaft_hz,
        envelope_bpfo_amplitude=amplitude_near(spec, bpfo_hz, band_half_width_hz),
        envelope_bpfo_2x_amplitude=amplitude_near(spec, 2.0 * bpfo_hz, band_half_width_hz),
        envelope_bpfo_peak_hz=peak_frequency_near(spec, bpfo_hz, band_half_width_hz),
    )


def compare_bearing_cases(
    t: np.ndarray,
    healthy_vibration: np.ndarray,
    fault_vibration: np.ndarray,
    omega_m: np.ndarray,
    config: BearingFaultConfig,
    *,
    t_start: float = DEFAULT_ANALYSIS_START_S,
    band_half_width_hz: float = DEFAULT_BAND_HALF_WIDTH_HZ,
) -> BearingComparisonMetrics:
    """Matched-window envelope comparison of the two simulated channels."""
    healthy = bearing_vibration_features(
        t, healthy_vibration, omega_m, config,
        t_start=t_start, band_half_width_hz=band_half_width_hz,
    )
    fault = bearing_vibration_features(
        t, fault_vibration, omega_m, config,
        t_start=t_start, band_half_width_hz=band_half_width_hz,
    )
    ratio = (
        fault.envelope_bpfo_amplitude / healthy.envelope_bpfo_amplitude
        if healthy.envelope_bpfo_amplitude > 0.0
        else None
    )
    return BearingComparisonMetrics(
        bpfo_hz=fault.bpfo_hz,
        shaft_hz_mean=fault.shaft_hz_mean,
        envelope_bpfo_healthy=healthy.envelope_bpfo_amplitude,
        envelope_bpfo_fault=fault.envelope_bpfo_amplitude,
        envelope_bpfo_2x_healthy=healthy.envelope_bpfo_2x_amplitude,
        envelope_bpfo_2x_fault=fault.envelope_bpfo_2x_amplitude,
        bpfo_peak_healthy_hz=healthy.envelope_bpfo_peak_hz,
        bpfo_peak_fault_hz=fault.envelope_bpfo_peak_hz,
        envelope_bpfo_ratio=ratio,
    )