"""Focused tests for Condition 02: increased mechanical load.

Controlled operating-condition change, not a confirmed internal motor fault.
"""

import numpy as np

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    PARK_CONVENTION,
    fault_02_increased_mechanical_load_scenario,
    first_milestone_scenario,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole
from imcm.validation.power_balance import power_balance_window

HEALTHY_TORQUE_NM = 15.0
FAULT_02_TORQUE_NM = 22.5


def _run(scenario):
    return simulate_healthy(scenario=scenario, t_end=1.0, max_step=1.0e-4, output_dt=2.0e-4)


def test_scenario_exists_with_expected_name() -> None:
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.name == "fault_02_increased_mechanical_load_plus_50pct"


def test_supply_matches_healthy_baseline() -> None:
    healthy = first_milestone_scenario()
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.supply == healthy.supply
    assert scenario.supply.frequency_hz == 50.0
    assert scenario.supply.line_line_rms_v == 400.0
    assert scenario.supply.waveform == "balanced_sinusoidal"


def test_park_convention_unchanged() -> None:
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.park_convention == PARK_CONVENTION
    assert scenario.park_convention == "krause_classical_2_3"


def test_motor_parameters_unchanged() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    healthy = _run(first_milestone_scenario())
    condition = _run(fault_02_increased_mechanical_load_scenario())
    assert healthy.params == params
    assert condition.params == params
    assert condition.params.provenance == "literature_example"


def test_stator_resistance_remains_healthy_and_balanced() -> None:
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.stator_resistance.enabled is False
    assert scenario.stator_resistance.multipliers_abc == (1.0, 1.0, 1.0)
    assert scenario.stator_resistance.label == "healthy_balanced"


def test_load_type_remains_constant_torque() -> None:
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.load.load_type == "constant_torque"


def test_healthy_and_condition_torque_values() -> None:
    healthy = first_milestone_scenario()
    scenario = fault_02_increased_mechanical_load_scenario()
    assert healthy.load.torque_nm == HEALTHY_TORQUE_NM
    assert scenario.load.torque_nm == FAULT_02_TORQUE_NM


def test_torque_increase_is_exactly_fifty_percent() -> None:
    healthy = first_milestone_scenario()
    scenario = fault_02_increased_mechanical_load_scenario()
    assert scenario.load.torque_nm == 1.5 * healthy.load.torque_nm
    increase = (scenario.load.torque_nm - healthy.load.torque_nm) / healthy.load.torque_nm
    assert increase == 0.5


def test_labelled_operating_condition_not_internal_fault() -> None:
    scenario = fault_02_increased_mechanical_load_scenario()
    text = (scenario.notes + " " + scenario.load.notes).lower()
    assert "operating-condition change" in text
    assert "not a confirmed internal motor fault" in text


def test_metadata_reports_increased_load() -> None:
    condition = _run(fault_02_increased_mechanical_load_scenario())
    assert condition.metadata["load_torque_nm"] == FAULT_02_TORQUE_NM
    assert condition.metadata["supply_frequency_hz"] == 50.0
    assert condition.metadata["provenance"] == "simulated"
    assert condition.metadata["experimental_validation"] is False


def test_finite_and_balanced_three_wire() -> None:
    condition = _run(fault_02_increased_mechanical_load_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc"):
        assert np.all(np.isfinite(getattr(condition, name))), name
    np.testing.assert_allclose(np.sum(condition.i_abc, axis=0), 0.0, rtol=0.0, atol=1e-12)


def test_higher_load_reduces_speed_and_increases_slip() -> None:
    healthy = _run(first_milestone_scenario())
    condition = _run(fault_02_increased_mechanical_load_scenario())
    assert condition.speed_rpm[-1] < healthy.speed_rpm[-1]
    assert condition.slip[-1] > healthy.slip[-1]


def test_steady_torque_tracks_higher_load() -> None:
    condition = _run(fault_02_increased_mechanical_load_scenario())
    mean_torque = float(np.mean(condition.tau_e[condition.t >= 0.8]))
    assert mean_torque > HEALTHY_TORQUE_NM
    assert abs(mean_torque - FAULT_02_TORQUE_NM) < 1.0


def test_power_balance_holds() -> None:
    condition = _run(fault_02_increased_mechanical_load_scenario())
    pb = power_balance_window(condition, t_start=0.8)
    assert pb.residual_rel < 1e-6
    assert abs(pb.p_mag_dot_mean) < 1.0