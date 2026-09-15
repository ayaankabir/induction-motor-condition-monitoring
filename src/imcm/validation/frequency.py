"""Frequency estimates for simulated waveforms (not experimental analysis)."""

from __future__ import annotations

import numpy as np


def interpolated_rfft_peak_hz(t: np.ndarray, x: np.ndarray) -> float:
    """Parabolic interpolation of the positive-frequency DFT magnitude peak.

    Frequency resolution of the raw DFT is ``1 / (t[-1] - t[0])``. Interpolation
    only refines the peak *within about one bin*; it cannot move a 40 Hz tone
    onto 50 Hz.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    dt = float(np.mean(np.diff(t)))
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freq = np.fft.rfftfreq(x.size, dt)
    k = int(np.argmax(mag[1:]) + 1)
    if not (1 <= k < mag.size - 1):
        return float(freq[k])
    denom = mag[k - 1] - 2.0 * mag[k] + mag[k + 1]
    if denom == 0.0:
        return float(freq[k])
    delta = 0.5 * (mag[k - 1] - mag[k + 1]) / denom
    return float(freq[k] + delta * (freq[1] - freq[0]))


def mean_zero_crossing_frequency_hz(t: np.ndarray, x: np.ndarray) -> float:
    """Independent period estimate from interpolated zero crossings."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    crossings = []
    for i in range(len(x) - 1):
        if x[i] == 0.0:
            crossings.append(float(t[i]))
        elif x[i] * x[i + 1] < 0.0:
            w = x[i] / (x[i] - x[i + 1])
            crossings.append(float(t[i] + w * (t[i + 1] - t[i])))
    if len(crossings) < 4:
        raise ValueError("not enough zero crossings to estimate frequency")
    half_periods = np.diff(crossings)
    return float(1.0 / (2.0 * np.mean(half_periods)))
