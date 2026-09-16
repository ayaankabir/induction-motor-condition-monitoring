"""Focused tests for Fault 04: bearing outer-race fault (BPFO vibration).

The bearing fault is a simulated vibration-sensor channel only.  The motor
ODEs, supply, load, and Park convention must remain exactly healthy, and the
Fault 01/02/03 paths must be untouched.  Simulation-only; no experimental
claims.
"""

import numpy as np
import pytest

from imcm.faults.bearing import (
    impacts_per_shaft_revolution,
    outer_race_impact_times_s,
    outer_race_vibration,
    shaft_angle_rad,
)
from imcm.faults.catalog import PLANNED_FAULTS
from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    BearingFaultConfig,
    fault_03_supply_voltage_unbalance_scenario,
    fault_04_bearing_outer_race_scenario,
    first_milestone_scenario,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole
from imcm.processing.envelope import (
    EnvelopeSpectrum,
    amplitude_near,
    bandpass,
    envelope_spectrum,
    peak_frequency_near,
)
from imcm.validation.bearing_metrics import (
    bearing_vibration_features,
    compare_bearing_cases,
)


BPFO_PER_REV = 0.5 * 9 * (1.0 - 7.94e-3 / 39.04e-3)  # ~3.5868 for the default config


def _run(scenario, **kwargs):
    defaults = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4)
    defaults.update(kwargs)
    return simulate_healthy(scenario=scenario, **defaults)


# ---------------------------------------------------------------------------
# Scenario metadata and naming
# ---------------------------------------------------------------------------


def test_scenario_metadata_and_naming() -> None:
    scenario = fault_04_bearing_outer_race_scenario()
    assert scenario.name == "fault_04_bearing_outer_race_bpfo"
    assert scenario.bearing_fault.enabled is True
    assert scenario.bearing_fault.label == "fault_04_outer_race_bpfo"
    healthy = first_milestone_scenario()
    assert scenario.supply == healthy.supply
    assert scenario.load == healthy.load
    assert scenario.park_convention == healthy.park_convention
    assert scenario.stator_resistance == healthy.stator_resistance


def test_catalog_contains_fault_04_entry() -> None:
    entry = next(f for f in PLANNED_FAULTS if f.id == "bearing_outer_race_bpfo")
    assert "BPFO" in entry.mechanism
    assert "not a confirmed real-machine diagnosis" in entry.honesty_note


def test_metadata_labels_bearing_channel_simulated() -> None:
    result = _run(fault_04_bearing_outer_race_scenario())
    assert result.metadata["label"] == "healthy"  # electrical run is healthy
    assert result.metadata["bearing_fault_enabled"] is True
    assert result.metadata["bearing_fault_case"] == "fault_04_outer_race_bpfo"
    assert result.metadata["bearing_fault_channel"] == "simulated_vibration_m_s2"
    assert result.metadata["provenance"] == "simulated"
    assert result.metadata["experimental_validation"] is False
    assert result.metadata["faults"] is False
    assert result.metadata["supply_fault"] is False


# ---------------------------------------------------------------------------
# Bearing parameters and BPFO
# ---------------------------------------------------------------------------


def test_bearing_parameters_are_literature_6205_values() -> None:
    config = BearingFaultConfig()
    assert config.nb_balls == 9
    np.testing.assert_allclose(config.ball_diameter_m, 7.94e-3, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(config.pitch_diameter_m, 39.04e-3, rtol=0.0, atol=0.0)
    assert config.contact_angle_deg == 0.0
    params = illustrative_4kw_400v_50hz_4pole()
    assert params.provenance == "literature_example"


def test_bpfo_calculation_matches_closed_form() -> None:
    config = BearingFaultConfig()
    shaft_hz = 24.4  # ~1464 r/min steady speed of this machine
    expected = 0.5 * config.nb_balls * shaft_hz * (
        1.0 - (config.ball_diameter_m / config.pitch_diameter_m)
    )
    np.testing.assert_allclose(config.bpfo_hz(shaft_hz), expected, rtol=0.0, atol=1e-12)
    assert impacts_per_shaft_revolution(config) == pytest.approx(BPFO_PER_REV, rel=1e-12)


def test_bpfo_scales_linearly_with_shaft_frequency() -> None:
    config = BearingFaultConfig()
    np.testing.assert_allclose(
        config.bpfo_hz(10.0) / 10.0, config.bpfo_hz(37.5) / 37.5, rtol=1e-12
    )


def test_contact_angle_raises_bpfo_and_lowers_bpfi() -> None:
    """cos(phi) < 1 for phi > 0, so BPFO rises and BPFI falls."""
    config = BearingFaultConfig()
    shaft_hz = 24.4
    angled = BearingFaultConfig(contact_angle_deg=30.0)
    assert angled.bpfo_hz(shaft_hz) > config.bpfo_hz(shaft_hz)
    assert angled.bpfi_hz(shaft_hz) < config.bpfi_hz(shaft_hz)
    assert angled.bpfo_hz(shaft_hz) == pytest.approx(
        0.5 * 9 * shaft_hz * (1.0 - (7.94e-3 / 39.04e-3) * np.cos(np.deg2rad(30.0)))
    )
    assert angled.bpfi_hz(shaft_hz) == pytest.approx(
        0.5 * 9 * shaft_hz * (1.0 + (7.94e-3 / 39.04e-3) * np.cos(np.deg2rad(30.0)))
    )


def test_invalid_bearing_configurations_rejected() -> None:
    with pytest.raises(ValueError):
        BearingFaultConfig(nb_balls=0)
    with pytest.raises(ValueError):
        BearingFaultConfig(pitch_diameter_m=1.0, ball_diameter_m=2.0)
    with pytest.raises(ValueError):
        BearingFaultConfig(contact_angle_deg=120.0)
    with pytest.raises(ValueError):
        BearingFaultConfig(resonance_hz=-1.0)
    with pytest.raises(ValueError):
        fault_04_bearing_outer_race_scenario(severity_scale=0.0)


# ---------------------------------------------------------------------------
# Healthy channel has no bearing-fault component
# ---------------------------------------------------------------------------


def test_healthy_vibration_channel_is_exactly_zero() -> None:
    result = _run(first_milestone_scenario())
    config = result.scenario.bearing_fault
    assert config.enabled is False
    vibration = outer_race_vibration(result.t, result.omega_m, config, enabled=config.enabled)
    np.testing.assert_array_equal(vibration, np.zeros_like(result.t))


def test_healthy_case_envelope_features_are_zero() -> None:
    healthy = _run(first_milestone_scenario())
    config = fault_04_bearing_outer_race_scenario().bearing_fault
    vibration = outer_race_vibration(healthy.t, healthy.omega_m, config, enabled=False)
    features = bearing_vibration_features(healthy.t, vibration, healthy.omega_m, config)
    assert features.envelope_bpfo_amplitude == 0.0
    assert features.envelope_bpfo_2x_amplitude == 0.0


def test_impact_rate_matches_bpfo_from_simulated_speed() -> None:
    """Impact count over the late window matches BPFO at the mean speed."""
    fault = _run(fault_04_bearing_outer_race_scenario())
    config = fault.scenario.bearing_fault
    mask = fault.t >= 0.5
    impacts = outer_race_impact_times_s(fault.t, fault.omega_m, config)
    n_in_window = int(np.count_nonzero(impacts >= 0.5))
    mean_shaft_hz = float(np.mean(fault.omega_m[mask])) / (2.0 * np.pi)
    expected = BPFO_PER_REV * mean_shaft_hz * 0.5  # window length 0.5 s
    assert n_in_window == pytest.approx(expected, abs=2)  # +/- one period tolerance


# ---------------------------------------------------------------------------
# Faulty case contains the BPFO component
# ---------------------------------------------------------------------------


def test_faulty_vibration_is_finite_nonzero_and_impact_train() -> None:
    fault = _run(fault_04_bearing_outer_race_scenario())
    config = fault.scenario.bearing_fault
    vibration = outer_race_vibration(fault.t, fault.omega_m, config, enabled=True)
    assert np.all(np.isfinite(vibration))
    assert np.max(np.abs(vibration)) > 0.0
    assert np.count_nonzero(vibration) > 0
    impacts = outer_race_impact_times_s(fault.t, fault.omega_m, config)
    assert impacts.size >= 3
    assert np.all(np.diff(impacts) > 0.0)  # impacts stay ordered in time
    # Impacts are at equal shaft-angle increments.  Evaluate the shaft angle at
    # the late (nearly steady) impact times; the increments must equal
    # 2*pi / (BPFO per revolution) to sub-milliradian accuracy.
    theta = shaft_angle_rad(fault.t, fault.omega_m)
    late = impacts[impacts >= 0.8][:4]
    assert late.size == 4
    expected_lag = 2.0 * np.pi / impacts_per_shaft_revolution(config)
    theta_at_impacts = np.interp(late, fault.t, theta)
    np.testing.assert_allclose(
        np.diff(theta_at_impacts), expected_lag, rtol=0.0, atol=1e-5
    )


def test_electrical_traces_remain_bit_for_bit_healthy() -> None:
    healthy = _run(first_milestone_scenario())
    fault = _run(fault_04_bearing_outer_race_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc", "v_abc", "slip"):
        np.testing.assert_array_equal(
            getattr(fault, name), getattr(healthy, name),
            err_msg=f"bearing scenario changed electrical trace {name}",
        )


def test_envelope_peak_lands_on_bpfo() -> None:
    fault = _run(fault_04_bearing_outer_race_scenario())
    config = fault.scenario.bearing_fault
    vibration = outer_race_vibration(fault.t, fault.omega_m, config, enabled=True)
    features = bearing_vibration_features(fault.t, vibration, fault.omega_m, config)
    df = 1.0 / 0.5  # analysis window length in seconds
    assert features.envelope_bpfo_peak_hz == pytest.approx(features.bpfo_hz, abs=2.0 * df)
    assert features.envelope_bpfo_amplitude > 0.0
    assert features.envelope_bpfo_2x_amplitude > 0.0


def test_compare_bearing_cases_reports_clear_separation() -> None:
    healthy = _run(first_milestone_scenario())
    fault = _run(fault_04_bearing_outer_race_scenario())
    config = fault.scenario.bearing_fault
    healthy_vib = outer_race_vibration(
        healthy.t, healthy.omega_m, config, enabled=healthy.scenario.bearing_fault.enabled
    )
    fault_vib = outer_race_vibration(fault.t, fault.omega_m, config, enabled=True)
    metrics = compare_bearing_cases(
        healthy.t, healthy_vib, fault_vib, healthy.omega_m, config
    )
    assert metrics.envelope_bpfo_healthy == 0.0
    assert metrics.envelope_bpfo_fault > 0.0
    assert metrics.envelope_bpfo_ratio is None  # healthy channel is exactly zero
    assert metrics.bpfo_peak_fault_hz == pytest.approx(metrics.bpfo_hz, abs=4.0)


def test_fault_03_path_unchanged_by_bearing_addition() -> None:
    """Regression guard: bearing config must not alter the Fault 03 scenario."""
    f03 = fault_03_supply_voltage_unbalance_scenario()
    healthy = first_milestone_scenario()
    assert f03.name == "fault_03_supply_voltage_unbalance"
    assert f03.bearing_fault.enabled is False
    assert f03.load == healthy.load
    assert f03.park_convention == healthy.park_convention
    assert f03.stator_resistance == healthy.stator_resistance
    # The intended Fault 03 difference (supply unbalance) is still present.
    assert f03.supply.voltage_unbalance.enabled is True
    assert f03.supply.voltage_unbalance.multipliers_abc == (1.0, 1.0, 0.9)


# ---------------------------------------------------------------------------
# Envelope / frequency-domain feature behaviour
# ---------------------------------------------------------------------------


def test_envelope_spectrum_of_known_amplitude_modulated_tone() -> None:
    """A tone at 1500 Hz modulated at 89 Hz shows an 89 Hz envelope line."""
    fs = 10_000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    carrier = np.sin(2.0 * np.pi * 1500.0 * t)
    mod = 1.0 + 0.5 * np.sin(2.0 * np.pi * 89.0 * t)
    spec = envelope_spectrum(carrier * mod, fs, band=(1000.0, 2000.0))
    assert amplitude_near(spec, 89.0, 3.0) > 0.0
    assert peak_frequency_near(spec, 89.0, 3.0) == pytest.approx(89.0, abs=0.5)
    # A clean unmodulated tone has an empty envelope spectrum.
    spec_flat = envelope_spectrum(carrier, fs, band=(1000.0, 2000.0))
    assert amplitude_near(spec_flat, 89.0, 3.0) < 1e-6


def test_bandpass_and_spectrum_helpers_behave() -> None:
    fs = 10_000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    x = np.sin(2.0 * np.pi * 1800.0 * t) + 0.3 * np.sin(2.0 * np.pi * 200.0 * t)
    filtered = bandpass(x, fs, 1500.0, 2200.0)
    assert np.all(np.isfinite(filtered))
    # The 200 Hz tone must be strongly attenuated relative to 1800 Hz.
    low_fraction = float(np.mean(np.abs(filtered) < 0.1 * np.max(np.abs(filtered))))
    assert low_fraction < 0.5  # filtered keeps most of the 1800 Hz carrier
    spec = envelope_spectrum(filtered, fs, band=(1500.0, 2200.0))
    assert spec.freq_hz[0] == 0.0
    assert spec.amplitude.shape == spec.freq_hz.shape
    with pytest.raises(ValueError):
        bandpass(x, fs, 2200.0, 1500.0)
    with pytest.raises(ValueError):
        amplitude_near(EnvelopeSpectrum(spec.freq_hz, spec.amplitude), 1e6, 1.0)


# ---------------------------------------------------------------------------
# Finite outputs and determinism
# ---------------------------------------------------------------------------


def test_vibration_outputs_finite_and_deterministic() -> None:
    fault = _run(fault_04_bearing_outer_race_scenario())
    config = fault.scenario.bearing_fault
    v1 = outer_race_vibration(fault.t, fault.omega_m, config, enabled=True)
    v2 = outer_race_vibration(fault.t, fault.omega_m, config, enabled=True)
    np.testing.assert_array_equal(v1, v2)  # deterministic, no RNG anywhere
    assert np.all(np.isfinite(v1))
    features = bearing_vibration_features(fault.t, v1, fault.omega_m, config)
    for value in (
        features.bpfo_hz,
        features.shaft_hz_mean,
        features.envelope_bpfo_amplitude,
        features.envelope_bpfo_2x_amplitude,
        features.envelope_bpfo_peak_hz,
    ):
        assert np.isfinite(value)


def test_full_simulation_outputs_finite_for_fault_04() -> None:
    fault = _run(fault_04_bearing_outer_race_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc"):
        assert np.all(np.isfinite(getattr(fault, name))), name


def test_shaft_angle_and_speed_input_validation() -> None:
    t = np.linspace(0.0, 0.1, 101)
    omega = np.full_like(t, 10.0)
    with pytest.raises(ValueError):
        shaft_angle_rad(t, np.ones((2, len(t))))
    with pytest.raises(ValueError):
        shaft_angle_rad(t[:-1], omega)  # length mismatch
    non_monotonic = t.copy()
    non_monotonic[5], non_monotonic[4] = non_monotonic[4], non_monotonic[5]
    with pytest.raises(ValueError):
        shaft_angle_rad(non_monotonic, omega)
    with pytest.raises(ValueError):
        outer_race_vibration(t, omega, BearingFaultConfig(resonance_hz=9000.0), enabled=True)