"""Signal processing and analysis tools."""

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

__all__ = [
    "AmplitudeSpectrum",
    "EnvelopeSpectrum",
    "DEFAULT_FILTER_ORDER",
    "bandpass_filter",
    "compute_amplitude_spectrum",
    "extract_hilbert_envelope",
    "compute_envelope_spectrum",
    "find_band_mask",
    "find_peak_amplitude",
    "find_interpolated_peak_frequency",
]
