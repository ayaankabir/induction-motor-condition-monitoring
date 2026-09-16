"""Envelope-spectrum tools for resonant band-pass fault signatures.

Generic, unit-testable signal-processing primitives used by the simulated
bearing-fault study (Fault 04): a zero-phase Butterworth band-pass, the
Hilbert-transform envelope, and a deterministic Hann-windowed amplitude
spectrum of that envelope.  Inputs are simulated signals; no experimental
analysis is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

DEFAULT_FILTER_ORDER = 4


@dataclass(frozen=True)
class EnvelopeSpectrum:
    """One-sided amplitude spectrum of a Hilbert envelope."""

    freq_hz: np.ndarray
    amplitude: np.ndarray


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass (filtfilt keeps the signal causal-free)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be one-dimensional with at least 8 samples")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if not 0.0 < low_hz < high_hz < 0.5 * fs:
        raise ValueError("require 0 < low_hz < high_hz < Nyquist")
    b, a = butter(order, (low_hz, high_hz), btype="bandpass", fs=fs)
    return filtfilt(b, a, x)


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
    x = np.asarray(x, dtype=float)
    x_bp = bandpass(x, fs, band[0], band[1], order=order)
    envelope = np.abs(hilbert(x_bp))
    envelope = envelope - np.mean(envelope)
    window = np.hanning(envelope.size)
    amplitude = 2.0 * np.abs(np.fft.rfft(envelope * window)) / np.sum(window)
    freq = np.fft.rfftfreq(envelope.size, d=1.0 / fs)
    return EnvelopeSpectrum(freq_hz=freq, amplitude=amplitude)


def _band_mask(spec: EnvelopeSpectrum, center_hz: float, half_width_hz: float) -> np.ndarray:
    if half_width_hz <= 0.0:
        raise ValueError("half_width_hz must be positive")
    mask = (spec.freq_hz >= center_hz - half_width_hz) & (
        spec.freq_hz <= center_hz + half_width_hz
    )
    if not np.any(mask):
        raise ValueError("no spectrum bins inside the requested band")
    return mask


def amplitude_near(
    spec: EnvelopeSpectrum, center_hz: float, half_width_hz: float
) -> float:
    """Peak envelope-spectrum amplitude within ``+/- half_width_hz`` of a target."""
    mask = _band_mask(spec, center_hz, half_width_hz)
    return float(np.max(spec.amplitude[mask]))


def peak_frequency_near(
    spec: EnvelopeSpectrum, center_hz: float, half_width_hz: float
) -> float:
    """Parabolically interpolated peak frequency near ``center_hz``."""
    mask = _band_mask(spec, center_hz, half_width_hz)
    local_index = int(np.argmax(spec.amplitude[mask]))
    k = int(np.nonzero(mask)[0][local_index])
    if not 1 <= k < spec.amplitude.size - 1:
        return float(spec.freq_hz[k])
    y0, y1, y2 = spec.amplitude[k - 1], spec.amplitude[k], spec.amplitude[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(spec.freq_hz[k])
    delta = 0.5 * (y0 - y2) / denom
    df = float(spec.freq_hz[1] - spec.freq_hz[0])
    return float(spec.freq_hz[k] + delta * df)