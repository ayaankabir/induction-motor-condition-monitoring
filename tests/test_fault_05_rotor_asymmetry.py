"""Focused tests for Fault 05: rotor electrical asymmetry proxy.

This is a **simulation-only proxy** for broken-bar-related behaviour: a
rotor-frame axis resistance split ``R_r (1 +/- severity)`` projected into the
synchronous frame at twice the reference slip angle.  It is not a physically
complete or bar-resolved broken-bar model, not severity-calibrated, not
experimentally validated, and not a real-machine diagnosis.

The locked fifth-order architecture, the healthy baseline, and the Fault
01-04 paths must remain unchanged.
"""

import numpy as np
import pytest

from imcm.faults.catalog import PLANNED_FAULTS
from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    RotorAsymmetryConfig,
    fault_03_supply_voltage_unbalance_scenario,
    fault_04_bearing_outer_race_scenario,
    fault_05_rotor_asymmetry_scenario,
    first_milestone_scenario,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole
from imcm.models.rotor_asymmetry import (
    modulation_phase_rad,
    reference_slip_for_load,
    rotor_asymmetry_drops,
    rotor_copper_loss_w,
    rotor_resistance_matrix_sync,
)
from imcm.validation.equivalent_circuit import equivalent_circuit_point
from imcm.validation.fault_metrics import torque_ripple_rms
from imcm.validation.power_balance import power_balance_window
from imcm.validation.rotor_asymmetry_metrics import (
    compare_rotor_asymmetry_cases,
    modulation_frequency_hz,
    sideband_frequencies_hz,
    spectrum_peak_amplitude,
)


def _run(scenario, t_end=1.0, **kwargs):
    defaults = dict(max_step=1.0e-4, output_dt=2.0e-4)
    defaults.update(kwargs)
    return simulate_healthy(scenario=scenario, t_end=t_end, **defaults)


def _with_rotor_config(scenario, config):
    return type(scenario)(
        name=scenario.name,
        supply=scenario.supply,
        load=scenario.load,
        park_convention=scenario.park_convention,
        notes=scenario.notes,
        stator_resistance=scenario.stator_resistance,
        bearing_fault=scenario.bearing_fault,
        rotor_asymmetry=config,
    )


# ---------------------------------------------------------------------------
# Default parameter validation
# ---------------------------------------------------------------------------


def test_default_config_is_disabled_and_healthy_labelled() -> None:
    config = RotorAsymmetryConfig()
    assert config.enabled is False
    assert config.severity == 0.0
    assert config.reference_slip is None
    assert config.label == "healthy_rotor"


def test_fault_05_scenario_defaults_and_metadata() -> None:
    scenario = fault_05_rotor_asymmetry_scenario()
    assert scenario.name == "fault_05_rotor_electrical_asymmetry"
    assert scenario.rotor_asymmetry.enabled is True
    assert scenario.rotor_asymmetry.severity == pytest.approx(0.10)
    assert scenario.rotor_asymmetry.label == "fault_05_rotor_asymmetry_severity_10pct"
    healthy = first_milestone_scenario()
    assert scenario.supply == healthy.supply
    assert scenario.load == healthy.load
    assert scenario.park_convention == healthy.park_convention
    assert scenario.stator_resistance == healthy.stator_resistance
    assert scenario.bearing_fault == healthy.bearing_fault


@pytest.mark.parametrize(
    "kwargs",
    (
        {"severity": -0.01},
        {"severity": 1.0},
        {"severity": 1.5},
        {"reference_slip": 0.0},
        {"reference_slip": -0.1},
        {"reference_slip": 1.0},
    ),
)
def test_invalid_rotor_asymmetry_configurations_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        RotorAsymmetryConfig(enabled=True, **kwargs)


def test_invalid_scenario_severity_rejected() -> None:
    with pytest.raises(ValueError):
        fault_05_rotor_asymmetry_scenario(severity=1.0)


def test_modulation_phase_requires_resolved_reference_slip() -> None:
    config = RotorAsymmetryConfig(enabled=True, severity=0.1, reference_slip=None)
    with pytest.raises(ValueError):
        modulation_phase_rad(0.1, config, 2.0 * np.pi * 50.0)


def test_drops_and_matrix_reject_invalid_severity_and_resistance() -> None:
    with pytest.raises(ValueError):
        rotor_asymmetry_drops(1.0, 0.5, 0.3, -0.1, 1.395)
    with pytest.raises(ValueError):
        rotor_asymmetry_drops(1.0, 0.5, 0.3, 0.1, 0.0)
    with pytest.raises(ValueError):
        rotor_resistance_matrix_sync(0.3, 1.0, 1.395)
    with pytest.raises(ValueError):
        rotor_copper_loss_w(np.ones(4), np.zeros(4), np.zeros(4), 0.0, -1.0)


# ---------------------------------------------------------------------------
# Proxy mathematics
# ---------------------------------------------------------------------------


def test_rotor_resistance_matrix_eigenvalues_and_mean_preserved() -> None:
    r_r = 1.395
    severity = 0.1
    for two_phi in np.linspace(0.0, 2.0 * np.pi, 13):
        matrix = rotor_resistance_matrix_sync(float(two_phi), severity, r_r)
        eigenvalues = np.linalg.eigvalsh(matrix)
        np.testing.assert_allclose(
            sorted(eigenvalues),
            sorted((r_r * (1.0 - severity), r_r * (1.0 + severity))),
            atol=1e-12,
        )
        np.testing.assert_allclose(np.trace(matrix), 2.0 * r_r, atol=1e-12)


def test_drops_reduce_to_healthy_when_severity_zero() -> None:
    np.testing.assert_allclose(
        rotor_asymmetry_drops(2.1, -0.7, 1.234, 0.0, 1.395),
        (1.395 * 2.1, 1.395 * -0.7),
        rtol=0.0,
        atol=1e-12,
    )


def test_drops_match_resistance_matrix_application() -> None:
    i_qr, i_dr, two_phi, severity, r_r = 1.7, -0.9, 0.83, 0.15, 1.395
    drop_q, drop_d = rotor_asymmetry_drops(i_qr, i_dr, two_phi, severity, r_r)
    vector = rotor_resistance_matrix_sync(two_phi, severity, r_r) @ np.array((i_qr, i_dr))
    np.testing.assert_allclose((drop_q, drop_d), vector, rtol=0.0, atol=1e-12)


def test_copper_loss_matches_quadratic_form_and_healthy_reduction() -> None:
    severity, r_r = 0.2, 1.395
    i_qr = np.array((1.0, 2.0, -0.5))
    i_dr = np.array((0.4, -1.0, 0.8))
    two_phi = np.array((0.0, 0.7, 1.9))
    loss = rotor_copper_loss_w(i_qr, i_dr, two_phi, severity, r_r)
    direct = np.array(
        [
            1.5
            * np.array((q, d))
            @ rotor_resistance_matrix_sync(float(p), severity, r_r)
            @ np.array((q, d))
            for q, d, p in zip(i_qr, i_dr, two_phi)
        ]
    )
    np.testing.assert_allclose(loss, direct, rtol=0.0, atol=1e-12)
    healthy = 1.5 * r_r * (i_qr**2 + i_dr**2)
    zero = rotor_copper_loss_w(i_qr, i_dr, two_phi, 0.0, r_r)
    np.testing.assert_allclose(zero, healthy, rtol=0.0, atol=1e-12)


def test_modulation_phase_advances_at_twice_reference_slip_rate() -> None:
    omega_e = 2.0 * np.pi * 50.0
    config = RotorAsymmetryConfig(enabled=True, severity=0.1, reference_slip=0.024)
    t = np.array((0.0, 0.5, 1.0))
    two_phi = modulation_phase_rad(t, config, omega_e)
    np.testing.assert_allclose(two_phi, 2.0 * 0.024 * omega_e * t, rtol=0.0, atol=1e-12)
    scalar = modulation_phase_rad(0.5, config, omega_e)
    assert isinstance(scalar, float)
    assert scalar == two_phi[1]


def test_reference_slip_matches_equivalent_circuit_torque_balance() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    scenario = first_milestone_scenario()
    s_ref = reference_slip_for_load(params, scenario)
    assert 0.0 < s_ref < 0.1
    point = equivalent_circuit_point(
        params,
        s_ref,
        line_line_rms_v=scenario.supply.line_line_rms_v,
        frequency_hz=scenario.supply.frequency_hz,
    )
    shaft_hz = (1.0 - s_ref) * scenario.supply.frequency_hz
    omega_m = 2.0 * np.pi * shaft_hz / params.pole_pairs
    friction = params.viscous_friction * omega_m
    np.testing.assert_allclose(
        point.electromagnetic_torque_nm,
        scenario.load.torque_nm + friction,
        rtol=1e-6,
    )


def test_sideband_and_modulation_frequency_helpers() -> None:
    f_lo, f_hi = sideband_frequencies_hz(50.0, 0.024)
    np.testing.assert_allclose(
        (f_lo, f_hi), (50.0 - 2.4, 50.0 + 2.4), rtol=0.0, atol=1e-12
    )
    assert modulation_frequency_hz(50.0, 0.024) == pytest.approx(2.4)


def test_spectrum_peak_amplitude_reads_true_tone_amplitude() -> None:
    fs = 10_000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)  # 1 Hz bin spacing; 47 Hz sits on a bin
    tone = 0.37 * np.sin(2.0 * np.pi * 47.0 * t)
    assert spectrum_peak_amplitude(t, tone, 47.0, 1.0) == pytest.approx(0.37, rel=0.02)
    with pytest.raises(ValueError):
        spectrum_peak_amplitude(t, tone, 47.6, -1.0)
    with pytest.raises(ValueError):
        spectrum_peak_amplitude(t, tone, 9500.0, 900.0)
    with pytest.raises(ValueError):
        spectrum_peak_amplitude(t[:4], tone[:4], 47.6, 1.0)


# ---------------------------------------------------------------------------
# Fault activation and healthy-versus-fault behaviour
# ---------------------------------------------------------------------------


def test_fault_activation_changes_synchronous_frame_currents() -> None:
    healthy = _run(first_milestone_scenario())
    fault = _run(fault_05_rotor_asymmetry_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc"):
        assert np.all(np.isfinite(getattr(fault, name))), name
    # The proxy must actually act on the plant: identical inputs, different states.
    assert not np.array_equal(fault.i_qs, healthy.i_qs)
    assert not np.array_equal(fault.i_qr, healthy.i_qr)
    assert not np.array_equal(fault.tau_e, healthy.tau_e)


def test_fault_signature_appears_and_healthy_case_lacks_it() -> None:
    healthy = _run(first_milestone_scenario(), t_end=3.0, output_dt=1.0e-4)
    fault = _run(fault_05_rotor_asymmetry_scenario(), t_end=3.0, output_dt=1.0e-4)
    metrics = compare_rotor_asymmetry_cases(healthy, fault, t_start=2.0)
    s_ref = metrics.reference_slip
    f_mod = modulation_frequency_hz(metrics.fundamental_hz, s_ref)
    # Synchronous-frame tone at 2 s f_s: strong in the fault, absent in healthy.
    assert metrics.sync_modulation_fault_a > 10.0 * max(
        metrics.sync_modulation_healthy_a, 1e-9
    )
    assert metrics.sync_modulation_fault_a > 0.01
    # Stator sidebands near f_s(1 -/+ 2s) rise above the healthy leakage floor.
    assert metrics.lower_sideband_fault_a > metrics.lower_sideband_healthy_a
    assert metrics.upper_sideband_fault_a > metrics.upper_sideband_healthy_a
    # Sanity: the fundamental itself is essentially unchanged.
    np.testing.assert_allclose(
        metrics.phase_a_fundamental_fault_a,
        metrics.phase_a_fundamental_healthy_a,
        rtol=0.05,
    )
    # Torque ripple at the modulation frequency appears in the fault case.
    assert torque_ripple_rms(fault, t_start=2.0, t_stop=3.0) > torque_ripple_rms(
        healthy, t_start=2.0, t_stop=3.0
    )
    assert f_mod == pytest.approx(2.0 * s_ref * 50.0)


def test_mean_operating_point_is_preserved_by_the_proxy() -> None:
    """The mean rotor resistance stays R_r, so the mean speed/slip barely move."""
    healthy = _run(first_milestone_scenario(), t_end=3.0, output_dt=1.0e-4)
    fault = _run(fault_05_rotor_asymmetry_scenario(), t_end=3.0, output_dt=1.0e-4)
    mask = fault.t >= 2.0
    np.testing.assert_allclose(
        float(np.mean(fault.omega_m[mask])),
        float(np.mean(healthy.omega_m[mask])),
        rtol=0.0,
        atol=2.0,  # rad/s; the fault slightly raises mean slip, as a real broken bar would
    )
    # The mean operating point shifts only slightly (slip +~0.0017 at severity
    # 0.10): the proxy adds a small mean braking torque via its anisotropic
    # loss term, qualitatively matching real broken-bar behaviour.
    assert abs(float(fault.slip[-1] - healthy.slip[-1])) < 3.0e-3


# ---------------------------------------------------------------------------
# Determinism and no unintended healthy-path changes
# ---------------------------------------------------------------------------


def test_fault_output_is_deterministic() -> None:
    first = _run(fault_05_rotor_asymmetry_scenario())
    second = _run(fault_05_rotor_asymmetry_scenario())
    for name in ("t", "omega_m", "i_qs", "i_ds", "i_qr", "i_dr", "tau_e", "i_abc"):
        np.testing.assert_array_equal(
            getattr(first, name), getattr(second, name), err_msg=f"nondeterministic {name}"
        )


def test_disabled_and_zero_severity_reproduce_healthy_exactly() -> None:
    healthy = _run(first_milestone_scenario())
    for config in (
        RotorAsymmetryConfig(enabled=False, severity=0.3, label="disabled_control"),
        RotorAsymmetryConfig(enabled=True, severity=0.0, label="zero_severity_control"),
    ):
        control = _run(
            _with_rotor_config(fault_05_rotor_asymmetry_scenario(), config)
        )
        for name in ("omega_m", "i_qs", "i_ds", "i_qr", "i_dr", "tau_e", "i_abc"):
            np.testing.assert_array_equal(
                getattr(control, name),
                getattr(healthy, name),
                err_msg=f"{config.label} changed {name}",
            )


def test_healthy_metadata_and_provenance_unchanged() -> None:
    healthy = _run(first_milestone_scenario())
    assert healthy.metadata["label"] == "healthy"
    assert healthy.metadata["faults"] is False
    assert healthy.metadata["rotor_asymmetry_enabled"] is False
    assert healthy.metadata["rotor_asymmetry_case"] == "healthy_rotor"
    assert healthy.metadata["provenance"] == "simulated"
    assert healthy.metadata["experimental_validation"] is False


def test_fault_metadata_labels_the_proxy_simulated() -> None:
    fault = _run(fault_05_rotor_asymmetry_scenario())
    assert fault.metadata["label"] == "fault_05_rotor_asymmetry_severity_10pct"
    assert fault.metadata["faults"] is True
    assert fault.metadata["rotor_asymmetry_enabled"] is True
    assert fault.metadata["rotor_asymmetry_severity"] == pytest.approx(0.10)
    assert 0.0 < fault.metadata["rotor_asymmetry_reference_slip"] < 0.1
    assert fault.metadata["provenance"] == "simulated"
    assert fault.metadata["experimental_validation"] is False
    assert fault.metadata["supply_fault"] is False
    assert fault.metadata["bearing_fault_enabled"] is False


def test_power_balance_holds_under_the_proxy() -> None:
    fault = _run(fault_05_rotor_asymmetry_scenario(), output_dt=1.0e-4)
    pb = power_balance_window(fault, t_start=0.8)
    assert pb.residual_rel < 1e-6
    assert abs(pb.p_mag_dot_mean) < 1.0


def test_fault_01_to_04_paths_unchanged_by_rotor_addition() -> None:
    """Regression guard: the rotor config must not alter earlier fault scenarios."""
    healthy = first_milestone_scenario()
    default = type(healthy)(
        name="x",
        supply=healthy.supply,
        load=healthy.load,
        park_convention=healthy.park_convention,
        notes="",
    )
    assert default.rotor_asymmetry == RotorAsymmetryConfig()
    f03 = fault_03_supply_voltage_unbalance_scenario()
    assert f03.rotor_asymmetry.enabled is False
    assert f03.supply.voltage_unbalance.multipliers_abc == (1.0, 1.0, 0.9)
    f04 = fault_04_bearing_outer_race_scenario()
    assert f04.rotor_asymmetry.enabled is False
    assert f04.bearing_fault.enabled is True


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_catalog_contains_fault_05_proxy_entry() -> None:
    entries = [f for f in PLANNED_FAULTS if f.id == "rotor_asymmetry_proxy"]
    assert len(entries) == 1  # exactly one registered Fault 05 entry
    entry = entries[0]
    assert entry.phase == 5
    assert "R_r(1 +/- severity)" in entry.mechanism
    assert "f_s(1 -/+ 2s)" in entry.mechanism
    lowered = entry.honesty_note.lower()
    assert "simulation-only proxy" in lowered
    assert "not a physically complete" in lowered
    assert "bar-resolved" in lowered
    assert "not experimentally validated" in lowered
    assert "real-machine diagnosis" in lowered


def test_scenario_honesty_language_in_notes() -> None:
    scenario = fault_05_rotor_asymmetry_scenario()
    text = scenario.notes.lower()
    assert "proxy" in text
    assert "not a physically complete broken rotor bar model" in text
    assert "not experimentally validated" in text