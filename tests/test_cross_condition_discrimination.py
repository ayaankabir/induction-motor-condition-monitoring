"""Cross-condition discrimination tests.

Validates that the condition-monitoring pipeline correctly separates true internal
faults from external operating confounders (such as load steps and supply unbalance),
demonstrating the core scientific premise of multi-channel diagnostic discrimination.
"""

from __future__ import annotations

import numpy as np
import pytest

from imcm.faults.bearing import outer_race_vibration
from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    fault_01_phase_a_resistance_imbalance_scenario,
    fault_02_increased_mechanical_load_scenario,
    fault_03_supply_voltage_unbalance_scenario,
    fault_04_bearing_outer_race_scenario,
    fault_05_rotor_asymmetry_scenario,
    first_milestone_scenario,
)
from imcm.validation.bearing_metrics import bearing_vibration_features
from imcm.validation.fault_metrics import (
    negative_sequence_current,
    phase_current_rms,
    supply_voltage_unbalance_pct,
)


@pytest.fixture(scope="module")
def sim_healthy():
    return simulate_healthy(scenario=first_milestone_scenario(), t_end=1.0)


@pytest.fixture(scope="module")
def sim_cond02_load():
    return simulate_healthy(scenario=fault_02_increased_mechanical_load_scenario(), t_end=1.0)


@pytest.fixture(scope="module")
def sim_fault01_stator_r():
    return simulate_healthy(scenario=fault_01_phase_a_resistance_imbalance_scenario(), t_end=1.0)


@pytest.fixture(scope="module")
def sim_fault03_supply_unbalance():
    return simulate_healthy(scenario=fault_03_supply_voltage_unbalance_scenario(), t_end=1.0)


def test_load_increase_alters_magnitude_without_inducing_negative_sequence(
    sim_healthy, sim_cond02_load
) -> None:
    """Condition 02 (+50% load) must increase current magnitude but keep I2 ~ 0."""
    rms_h = phase_current_rms(sim_healthy)
    rms_load = phase_current_rms(sim_cond02_load)

    # Phase current RMS must increase with load
    assert np.mean(rms_load) > np.mean(rms_h) * 1.25

    # Negative sequence current must remain near numerical zero for both
    i2_h, _ = negative_sequence_current(sim_healthy)
    i2_load, _ = negative_sequence_current(sim_cond02_load)

    assert i2_h < 1.0e-5
    assert i2_load < 1.0e-5

    # Current unbalance must remain negligible
    unbalance_load = 100.0 * (max(rms_load) - min(rms_load)) / np.mean(rms_load)
    assert unbalance_load < 0.01


def test_supply_unbalance_vs_stator_resistance_confounder_discrimination(
    sim_healthy, sim_fault01_stator_r, sim_fault03_supply_unbalance
) -> None:
    """Separating Stator R from Supply Unbalance requires measuring terminal voltages.

    Both conditions produce a 50 Hz negative-sequence current (I2 > 0), but only
    Supply Unbalance has an external negative-sequence voltage (V2 > 0).
    """
    # Negative sequence current in both cases
    i2_f01, _ = negative_sequence_current(sim_fault01_stator_r)
    i2_f03, _ = negative_sequence_current(sim_fault03_supply_unbalance)

    assert i2_f01 > 0.01  # Internal asymmetry creates I2
    assert i2_f03 > 0.5   # External supply unbalance creates large I2

    # Terminal voltage unbalance factor (VUF)
    vuf_h = supply_voltage_unbalance_pct(sim_healthy)
    vuf_f01 = supply_voltage_unbalance_pct(sim_fault01_stator_r)
    vuf_f03 = supply_voltage_unbalance_pct(sim_fault03_supply_unbalance)

    assert vuf_h < 1.0e-6
    assert vuf_f01 < 1.0e-6  # Stator fault does NOT change supply voltage
    assert vuf_f03 > 3.0     # Supply fault has large VUF (~3.45%)

    # The ratio of I2 to V2 clearly separates the causes:
    # Stator fault has high I2 with V2 ~ 0; Supply fault has matching V2 and I2
    assert vuf_f01 == pytest.approx(0.0, abs=1e-5)
    assert vuf_f03 > 3.0


def test_bearing_fault_alters_only_vibration_channel_not_electrical(sim_healthy) -> None:
    """Fault 04 alters only the auxiliary vibration channel; electrical ODE is untouched."""
    scenario_b = fault_04_bearing_outer_race_scenario()
    sim_b = simulate_healthy(scenario=scenario_b, t_end=1.0)

    # Electrical arrays are bit-for-bit identical
    np.testing.assert_array_equal(sim_b.i_abc, sim_healthy.i_abc)
    np.testing.assert_array_equal(sim_b.v_abc, sim_healthy.v_abc)
    np.testing.assert_array_equal(sim_b.omega_m, sim_healthy.omega_m)

    # Vibration channel is distinct
    vib_h = outer_race_vibration(sim_healthy.t, sim_healthy.omega_m, scenario_b.bearing_fault, enabled=False)
    vib_fault = outer_race_vibration(sim_b.t, sim_b.omega_m, scenario_b.bearing_fault, enabled=True)

    assert np.all(vib_h == 0.0)
    assert np.max(np.abs(vib_fault)) > 0.5

    # Envelope features detect BPFO only on the fault vibration channel
    feat_h = bearing_vibration_features(sim_healthy.t, vib_h, sim_healthy.omega_m, scenario_b.bearing_fault)
    feat_fault = bearing_vibration_features(sim_b.t, vib_fault, sim_b.omega_m, scenario_b.bearing_fault)

    assert feat_h.envelope_bpfo_amplitude == 0.0
    assert feat_fault.envelope_bpfo_amplitude > 0.05
    assert abs(feat_fault.envelope_bpfo_peak_hz - feat_fault.bpfo_hz) < 1.0


def test_rotor_asymmetry_proxy_does_not_create_fundamental_negative_sequence(sim_healthy) -> None:
    """Fault 05 produces sub-synchronous 2*s*f_s modulation, not steady 50 Hz unbalance (I2)."""
    scenario_r = fault_05_rotor_asymmetry_scenario(severity=0.10)
    sim_r = simulate_healthy(scenario=scenario_r, t_end=1.0)

    # Fundamental 50 Hz negative sequence current should remain negligible
    i2_r, _ = negative_sequence_current(sim_r)
    assert i2_r < 0.005  # Negligible compared to stator unbalance (0.026 A) or supply unbalance (1.38 A)
