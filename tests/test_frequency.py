"""Synthetic checks that the frequency estimators reject the wrong tone."""

import numpy as np

from imcm.validation.frequency import interpolated_rfft_peak_hz, mean_zero_crossing_frequency_hz


def _tone(freq_hz: float, duration: float = 0.4, dt: float = 2.0e-4) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration, dt)
    return t, np.sin(2.0 * np.pi * freq_hz * t)


def test_interpolated_peak_finds_50_hz() -> None:
    t, x = _tone(50.0)
    assert abs(interpolated_rfft_peak_hz(t, x) - 50.0) < 0.2


def test_interpolated_peak_does_not_report_50_for_40_or_60_hz() -> None:
    """Same idea as the plant current test: |f - 50| must fail for a wrong tone.

    A 40 Hz or 60 Hz sinusoid must remain near its true frequency. Lengthening
    the window and interpolating must not pull those peaks onto 50 Hz.
    """
    for f in (40.0, 60.0):
        t, x = _tone(f)
        peak = interpolated_rfft_peak_hz(t, x)
        assert abs(peak - f) < 0.5
        assert abs(peak - 50.0) >= 1.0


def test_short_window_argmax_is_not_a_1_hz_measurement() -> None:
    """0.15 s DFT bins are ~6.7 Hz apart; raw argmax can land ~3 Hz off 50 Hz."""
    t, x = _tone(50.0, duration=0.15, dt=2.0e-4)
    dt = float(np.mean(np.diff(t)))
    mag = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(x.size)))
    freq = np.fft.rfftfreq(x.size, dt)
    raw = float(freq[np.argmax(mag[1:]) + 1])
    assert abs(raw - 50.0) > 1.0
    assert abs(interpolated_rfft_peak_hz(t, x) - 50.0) < 1.0


def test_zero_crossing_frequency_matches_tone() -> None:
    t, x = _tone(50.0)
    assert abs(mean_zero_crossing_frequency_hz(t, x) - 50.0) < 0.05
    t40, x40 = _tone(40.0)
    f40 = mean_zero_crossing_frequency_hz(t40, x40)
    assert abs(f40 - 40.0) < 0.05
    assert abs(f40 - 50.0) >= 1.0
