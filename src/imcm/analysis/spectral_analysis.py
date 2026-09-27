"""Generic signal processing and spectral analysis utilities.

Provides reusable, deterministic primitives for spectral analysis:
- Zero-phase Butterworth bandpass filtering
- Single-sided amplitude spectrum with Hann windowing and 2.0*|X|/sum(w) scaling
- Hilbert-transform envelope extraction and envelope amplitude spectrum
- Band-limited peak amplitude detection
- 3-point parabolic peak frequency interpolation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

DEFAULT_FILTER_ORDER: int = 4


@dataclass(frozen=True)
class AmplitudeSpectrum:
    """One-sided amplitude spectrum of a signal."""

    freq_hz: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class EnvelopeSpectrum:
    """One-sided amplitude spectrum of a Hilbert envelope."""

    freq_hz: np.ndarray
    amplitude: np.ndarray


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (filtfilt keeps the signal causal-free)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be one-dimensional with at least 8 samples")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if not 0.0 < low_hz < high_hz < 0.5 * fs:
        raise ValueError("require 0 < low_hz < high_hz < Nyquist")
    b, a = butter(order, (low_hz, high_hz), btype="bandpass", fs=fs)
    return filtfilt(b, a, x)


def compute_amplitude_spectrum(
    x: np.ndarray,
    fs: float,
    *,
    remove_mean: bool = True,
) -> AmplitudeSpectrum:
    """Compute one-sided Hann-windowed amplitude spectrum.

    Uses the project-standard normalisation:
        window = np.hanning(N)
        amplitude = 2.0 * np.abs(np.fft.rfft(x_proc * window)) / np.sum(window)
    so an isolated sinusoidal tone's peak bin reads its true amplitude.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be one-dimensional with at least 8 samples")
    if fs <= 0.0:
        raise ValueError("fs must be positive")

    x_proc = x - np.mean(x) if remove_mean else x
    window = np.hanning(x_proc.size)
    amplitude = 2.0 * np.abs(np.fft.rfft(x_proc * window)) / np.sum(window)
    freq_hz = np.fft.rfftfreq(x_proc.size, d=1.0 / fs)
    return AmplitudeSpectrum(freq_hz=freq_hz, amplitude=amplitude)


def extract_hilbert_envelope(
    x: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float],
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Band-pass filter the signal and extract its Hilbert envelope magnitude."""
    x_bp = bandpass_filter(x, fs, band[0], band[1], order=order)
    return np.abs(hilbert(x_bp))


def compute_envelope_spectrum(
    x: np.ndarray,
    fs: float,
    *,
    band: tuple[float, float],
    order: int = DEFAULT_FILTER_ORDER,
) -> EnvelopeSpectrum:
    """Envelope spectrum of ``x`` band-passed to ``band = (low, high)`` Hz.

    Demodulation chain: band-pass -> Hilbert envelope -> mean removal ->
    Hann window -> single-sided amplitude spectrum. Deterministic.
    """
    envelope = extract_hilbert_envelope(x, fs, band=band, order=order)
    spec = compute_amplitude_spectrum(envelope, fs, remove_mean=True)
    return EnvelopeSpectrum(freq_hz=spec.freq_hz, amplitude=spec.amplitude)


def find_band_mask(
    freq_hz: np.ndarray,
    center_hz: float,
    half_width_hz: float,
) -> np.ndarray:
    """Return boolean mask for frequency bins within ``center_hz +/- half_width_hz``."""
    if half_width_hz <= 0.0:
        raise ValueError("half_width_hz must be positive")
    mask = (freq_hz >= center_hz - half_width_hz) & (
        freq_hz <= center_hz + half_width_hz
    )
    if not np.any(mask):
        raise ValueError("no spectrum bins inside the requested band")
    return mask


def find_peak_amplitude(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    center_hz: float,
    half_width_hz: float,
) -> float:
    """Peak amplitude within ``+/- half_width_hz`` of a target frequency."""
    mask = find_band_mask(freq_hz, center_hz, half_width_hz)
    return float(np.max(amplitude[mask]))


def find_interpolated_peak_frequency(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    center_hz: float,
    half_width_hz: float,
) -> float:
    """Parabolically interpolated peak frequency near ``center_hz``."""
    mask = find_band_mask(freq_hz, center_hz, half_width_hz)
    local_index = int(np.argmax(amplitude[mask]))
    k = int(np.nonzero(mask)[0][local_index])
    if not 1 <= k < amplitude.size - 1:
        return float(freq_hz[k])
    y0, y1, y2 = amplitude[k - 1], amplitude[k], amplitude[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(freq_hz[k])
    delta = 0.5 * (y0 - y2) / denom
    df = float(freq_hz[1] - freq_hz[0])
    return float(freq_hz[k] + delta * df)
