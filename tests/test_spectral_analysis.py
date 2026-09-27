"""Unit tests for generic spectral analysis utilities in imcm.analysis.spectral_analysis."""

from __future__ import annotations

import numpy as np
import pytest

from imcm.analysis.spectral_analysis import (
    AmplitudeSpectrum,
    DEFAULT_FILTER_ORDER,
    EnvelopeSpectrum,
    bandpass_filter,
    compute_amplitude_spectrum,
    compute_envelope_spectrum,
    extract_hilbert_envelope,
    find_band_mask,
    find_interpolated_peak_frequency,
    find_peak_amplitude,
)
from imcm.processing.envelope import (
    amplitude_near,
    bandpass,
    envelope_spectrum,
    peak_frequency_near,
)


def test_amplitude_spectrum_single_tone_reads_true_amplitude() -> None:
    fs = 10_000.0
    duration = 2.0
    t = np.arange(int(fs * duration)) / fs
    target_hz = 47.0
    target_amp = 0.85
    tone = target_amp * np.sin(2.0 * np.pi * target_hz * t)

    spec = compute_amplitude_spectrum(tone, fs, remove_mean=True)
    peak_amp = find_peak_amplitude(spec.freq_hz, spec.amplitude, target_hz, half_width_hz=1.0)
    assert peak_amp == pytest.approx(target_amp, rel=0.01)

    peak_freq = find_interpolated_peak_frequency(
        spec.freq_hz, spec.amplitude, target_hz, half_width_hz=1.0
    )
    assert peak_freq == pytest.approx(target_hz, abs=0.2)


def test_amplitude_spectrum_dc_removal() -> None:
    fs = 1000.0
    t = np.arange(1000) / fs
    x = 10.0 + 1.0 * np.cos(2.0 * np.pi * 20.0 * t)

    spec_no_mean = compute_amplitude_spectrum(x, fs, remove_mean=True)
    assert spec_no_mean.amplitude[0] < 1e-4

    spec_with_mean = compute_amplitude_spectrum(x, fs, remove_mean=False)
    assert spec_with_mean.amplitude[0] > 5.0


def test_bandpass_filter_and_hilbert_envelope() -> None:
    fs = 10_000.0
    t = np.arange(10_000) / fs
    carrier_hz = 1500.0
    mod_hz = 85.0
    carrier = np.sin(2.0 * np.pi * carrier_hz * t)
    mod = 1.0 + 0.4 * np.sin(2.0 * np.pi * mod_hz * t)
    signal = carrier * mod

    env = extract_hilbert_envelope(signal, fs, band=(1000.0, 2000.0))
    assert env.shape == signal.shape
    assert np.mean(env) > 0.0

    env_spec = compute_envelope_spectrum(signal, fs, band=(1000.0, 2000.0))
    peak_amp = find_peak_amplitude(env_spec.freq_hz, env_spec.amplitude, mod_hz, 3.0)
    assert peak_amp > 0.1

    peak_freq = find_interpolated_peak_frequency(
        env_spec.freq_hz, env_spec.amplitude, mod_hz, 3.0
    )
    assert peak_freq == pytest.approx(mod_hz, abs=0.5)


def test_find_band_mask_and_peak_amplitude_errors() -> None:
    freqs = np.linspace(0, 500, 501)
    amps = np.ones_like(freqs)

    with pytest.raises(ValueError, match="half_width_hz must be positive"):
        find_band_mask(freqs, 100.0, -1.0)

    with pytest.raises(ValueError, match="no spectrum bins inside the requested band"):
        find_band_mask(freqs, 1000.0, 10.0)

    with pytest.raises(ValueError, match="no spectrum bins inside the requested band"):
        find_peak_amplitude(freqs, amps, 1000.0, 10.0)


def test_envelope_processing_backward_compatibility() -> None:
    fs = 10_000.0
    t = np.arange(10_000) / fs
    carrier_hz = 1500.0
    mod_hz = 85.0
    sig = np.sin(2.0 * np.pi * carrier_hz * t) * (1.0 + 0.5 * np.sin(2.0 * np.pi * mod_hz * t))

    # Old API through processing.envelope
    spec_old = envelope_spectrum(sig, fs, band=(1000.0, 2000.0))
    amp_old = amplitude_near(spec_old, mod_hz, 3.0)
    freq_old = peak_frequency_near(spec_old, mod_hz, 3.0)

    # Direct new API
    spec_new = compute_envelope_spectrum(sig, fs, band=(1000.0, 2000.0))
    amp_new = find_peak_amplitude(spec_new.freq_hz, spec_new.amplitude, mod_hz, 3.0)
    freq_new = find_interpolated_peak_frequency(spec_new.freq_hz, spec_new.amplitude, mod_hz, 3.0)

    assert np.array_equal(spec_old.freq_hz, spec_new.freq_hz)
    assert np.array_equal(spec_old.amplitude, spec_new.amplitude)
    assert amp_old == amp_new
    assert freq_old == freq_new


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="at least 8 samples"):
        compute_amplitude_spectrum(np.array([1.0, 2.0]), 100.0)

    with pytest.raises(ValueError, match="fs must be positive"):
        compute_amplitude_spectrum(np.ones(10), -10.0)

    with pytest.raises(ValueError, match="require 0 < low_hz < high_hz < Nyquist"):
        bandpass_filter(np.ones(10), 100.0, 60.0, 40.0)
