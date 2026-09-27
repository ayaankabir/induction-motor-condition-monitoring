"""Envelope-spectrum tools for resonant band-pass fault signatures.

Backward-compatible re-exports and wrappers around the shared
`imcm.analysis.spectral_analysis` primitives.
"""

from __future__ import annotations

import numpy as np

from imcm.analysis.spectral_analysis import (
    DEFAULT_FILTER_ORDER,
    EnvelopeSpectrum,
    bandpass_filter,
    compute_envelope_spectrum,
    find_interpolated_peak_frequency,
    find_peak_amplitude,
)

__all__ = [
    "DEFAULT_FILTER_ORDER",
    "EnvelopeSpectrum",
    "bandpass",
    "envelope_spectrum",
    "amplitude_near",
    "peak_frequency_near",
]


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass (filtfilt keeps the signal causal-free)."""
    return bandpass_filter(x, fs, low_hz, high_hz, order=order)


def envelope_spectrum(
    x: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float],
    order: int = DEFAULT_FILTER_ORDER,
) -> EnvelopeSpectrum:
    """Envelope spectrum of ``x`` band-passed to ``band = (low, high)`` Hz.

    Demodulation chain: band-pass -> Hilbert envelope -> mean removal ->
    Hann window -> single-sided amplitude spectrum.  Deterministic.
    """
    return compute_envelope_spectrum(x, fs, band=band, order=order)


def amplitude_near(
    spec: EnvelopeSpectrum, center_hz: float, half_width_hz: float
) -> float:
    """Peak envelope-spectrum amplitude within ``+/- half_width_hz`` of a target."""
    return find_peak_amplitude(spec.freq_hz, spec.amplitude, center_hz, half_width_hz)


def peak_frequency_near(
    spec: EnvelopeSpectrum, center_hz: float, half_width_hz: float
) -> float:
    """Parabolically interpolated peak frequency near ``center_hz``."""
    return find_interpolated_peak_frequency(
        spec.freq_hz, spec.amplitude, center_hz, half_width_hz
    )
