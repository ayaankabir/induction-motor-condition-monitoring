"""Focused tests for Fault 03: supply-voltage unbalance.

Controlled supply-side confounder, not a confirmed internal motor fault and not
winding damage.  The balanced-supply control path must remain unchanged.
"""

from dataclasses import replace

import numpy as np

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    VoltageUnbalanceConfig,
    fault_03_supply_voltage_unbalance_scenario,
    first_milestone_scenario,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole
from imcm.models.supply_voltage import (
    supply_voltage_qd,
    supply_voltage_traces,
    voltage_unbalance_factor_pct,
)
from imcm.signals.park import abc_to_qd0, unbalanced_phase_voltages
from imcm.validation.fault_metrics import (
    CONVERGENCE_TOLERANCES,
    compare_supply_unbalance_cases,
)
from imcm.validation.power_balance import power_balance_window

PHASE_PEAK_V = 400.0 * np.sqrt(2.0) / np.sqrt(3.0)


def _run(scenario, **kwargs):
    defaults = dict(t_end=1.0, max_step=1.0e-4, output_dt=2.0e-4)
    defaults.update(kwargs)
    return simulate_healthy(scenario=scenario, **defaults)


def test_scenario_exists_with_expected_name() -> None:
    scenario = fault_03_supply_voltage_unbalance_scenario()
    assert scenario.name == "fault_03_supply_voltage_unbalance"
    assert scenario.supply.voltage_unbalance.enabled is True
    assert scenario.supply.voltage_unbalance.multipliers_abc == (1.0, 1.0, 0.9)
    assert scenario.supply.voltage_unbalance.label == "fault_03_phase_c_minus_10pct"


def test_balanced_supply_reproduces_healthy_result() -> None:
    healthy_scenario = first_milestone_scenario()
    balanced_scenario = replace(
        healthy_scenario,
        supply=replace(
            healthy_scenario.supply,
            voltage_unbalance=VoltageUnbalanceConfig(
                enabled=True, multipliers_abc=(1.0, 1.0, 1.0), label="balanced_control"
            ),
        ),
    )
    healthy = _run(healthy_scenario)
    balanced = _run(balanced_scenario)
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc", "v_abc"):
        np.testing.assert_allclose(
            getattr(balanced, name), getattr(healthy, name), rtol=0.0, atol=1e-12
        )


def test_direct_abc_voltage_reconstruction() -> None:
    scenario = fault_03_supply_voltage_unbalance_scenario()
    result = _run(scenario)
    theta = scenario.supply.omega_e * result.t
    expected = unbalanced_phase_voltages(
        result.t,
        scenario.supply.omega_e,
        scenario.supply.phase_peak_v,
        scenario.supply.voltage_unbalance.multipliers_abc,
    )
    np.testing.assert_allclose(result.v_abc, expected, rtol=0.0, atol=1e-12)
    v_qd0 = abc_to_qd0(result.v_abc, theta)
    np.testing.assert_allclose(v_qd0[0], result.v_qs, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(v_qd0[1], result.v_ds, rtol=0.0, atol=1e-9)
    # The applied unbalanced set has a nonzero zero-sequence voltage, but the
    # three-wire machine only responds to v_qs and v_ds (i0 is identically 0).
    np.testing.assert_allclose(
        v_qd0[2], (result.v_abc[0] + result.v_abc[1] + result.v_abc[2]) / 3.0,
        rtol=0.0, atol=1e-12,
    )


def test_nonzero_voltage_unbalance_factor_for_unbalanced_case() -> None:
    result = _run(fault_03_supply_voltage_unbalance_scenario())
    mask = (result.t >= 0.8) & (result.t < 1.0)
    vuf = voltage_unbalance_factor_pct(
        result.t[mask], result.v_abc[:, mask], result.scenario.supply.omega_e
    )
    assert vuf > 1.0
    # Analytical |V2|/|V1| for (1, 1, 0.9) is 0.1/sqrt(29) ~= 1.857% of the
    # positive-sequence magnitude; the standard VUF is 2x that ~= 3.448%.
    np.testing.assert_allclose(vuf, 3.448275862, rtol=1e-6)


def test_zero_voltage_unbalance_factor_for_balanced_case() -> None:
    result = _run(first_milestone_scenario())
    mask = (result.t >= 0.8) & (result.t < 1.0)
    vuf = voltage_unbalance_factor_pct(
        result.t[mask], result.v_abc[:, mask], result.scenario.supply.omega_e
    )
    assert vuf < 1e-9


def test_phase_voltage_magnitudes_and_120_degree_relationships() -> None:
    scenario = fault_03_supply_voltage_unbalance_scenario()
    peaks = scenario.supply.phase_peak_abc_v
    np.testing.assert_allclose(peaks, (PHASE_PEAK_V, PHASE_PEAK_V, 0.9 * PHASE_PEAK_V), rtol=1e-12)
    result = _run(scenario)
    mask = (result.t >= 0.8) & (result.t < 1.0)
    t = result.t[mask]
    theta = scenario.supply.omega_e * t
    phasors = np.sqrt(2.0) * np.mean(result.v_abc[:, mask] * np.exp(-1j * theta), axis=1)
    np.testing.assert_allclose(np.abs(phasors), np.asarray(peaks) / np.sqrt(2.0), rtol=1e-6)
    angle_b = np.angle(phasors[1]) - np.angle(phasors[0])
    angle_c = np.angle(phasors[2]) - np.angle(phasors[0])
    np.testing.assert_allclose(angle_b, -2.0 * np.pi / 3.0, atol=1e-6)
    np.testing.assert_allclose(angle_c, 2.0 * np.pi / 3.0, atol=1e-6)


def test_nonzero_negative_sequence_current_under_unbalanced_supply() -> None:
    healthy = _run(first_milestone_scenario())
    unbalance = _run(fault_03_supply_voltage_unbalance_scenario())
    metrics = compare_supply_unbalance_cases(healthy, unbalance)
    assert metrics.negative_sequence_unbalance_a > 1.0
    assert metrics.negative_sequence_unbalance_a > metrics.negative_sequence_healthy_a + 1.0
    assert metrics.current_unbalance_unbalance_pct > metrics.current_unbalance_healthy_pct + 1.0
    assert metrics.supply_voltage_unbalance_unbalance_pct > 1.0
    assert metrics.supply_voltage_unbalance_healthy_pct < 1e-9


def test_power_balance_residual() -> None:
    # output_dt=1e-4 keeps the np.gradient truncation below the 1e-6 limit; the
    # residual is a discretization artifact and converges as dt^2.
    unbalance = _run(fault_03_supply_voltage_unbalance_scenario(), output_dt=1.0e-4)
    pb = power_balance_window(unbalance, t_start=0.8)
    assert pb.residual_rel < 1e-6
    assert abs(pb.p_mag_dot_mean) < 1.0


def test_metrics_converge_with_tighter_rk45_settings() -> None:
    normal_kwargs = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    tight_kwargs = dict(t_end=1.0, max_step=1.0e-5, output_dt=1.0e-4, rtol=1.0e-9, atol=1.0e-11)
    normal = compare_supply_unbalance_cases(
        simulate_healthy(scenario=first_milestone_scenario(), **normal_kwargs),
        simulate_healthy(scenario=fault_03_supply_voltage_unbalance_scenario(), **normal_kwargs),
    )
    tight = compare_supply_unbalance_cases(
        simulate_healthy(scenario=first_milestone_scenario(), **tight_kwargs),
        simulate_healthy(scenario=fault_03_supply_voltage_unbalance_scenario(), **tight_kwargs),
    )
    assert abs(normal.negative_sequence_unbalance_a - tight.negative_sequence_unbalance_a) < CONVERGENCE_TOLERANCES["negative_sequence_a"]
    assert abs(normal.negative_sequence_unbalance_pct - tight.negative_sequence_unbalance_pct) < CONVERGENCE_TOLERANCES["current_unbalance_pct"]
    assert abs(normal.supply_voltage_unbalance_unbalance_pct - tight.supply_voltage_unbalance_unbalance_pct) < CONVERGENCE_TOLERANCES["voltage_unbalance_pct"]
    assert abs(normal.torque_ripple_unbalance_nm - tight.torque_ripple_unbalance_nm) < CONVERGENCE_TOLERANCES["torque_ripple_nm"]
    assert abs(normal.final_speed_difference_rpm - tight.final_speed_difference_rpm) < CONVERGENCE_TOLERANCES["final_speed_difference_rpm"]
    assert abs(normal.final_slip_difference - tight.final_slip_difference) < CONVERGENCE_TOLERANCES["final_slip_difference"]


def test_metadata_and_simulation_only_labelling() -> None:
    unbalance = _run(fault_03_supply_voltage_unbalance_scenario())
    assert unbalance.metadata["label"] == "fault_03_phase_c_minus_10pct"
    assert unbalance.metadata["supply_voltage_case"] == "fault_03_phase_c_minus_10pct"
    assert unbalance.metadata["supply_voltage_multipliers_abc"] == (1.0, 1.0, 0.9)
    assert unbalance.metadata["supply_fault"] is True
    assert unbalance.metadata["faults"] is False
    assert unbalance.metadata["provenance"] == "simulated"
    assert unbalance.metadata["experimental_validation"] is False
    assert unbalance.metadata["supply_unbalance"]["enabled"] is True
    np.testing.assert_allclose(
        unbalance.metadata["supply_phase_peak_abc_v"],
        (PHASE_PEAK_V, PHASE_PEAK_V, 0.9 * PHASE_PEAK_V),
        rtol=1e-12,
    )


def test_motor_parameters_and_stator_resistance_unchanged() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    scenario = fault_03_supply_voltage_unbalance_scenario()
    assert scenario.stator_resistance.enabled is False
    assert scenario.stator_resistance.multipliers_abc == (1.0, 1.0, 1.0)
    assert scenario.load == first_milestone_scenario().load
    result = _run(scenario)
    assert result.params == params
    assert result.params.provenance == "literature_example"


def test_labelled_supply_confounder_not_winding_fault() -> None:
    scenario = fault_03_supply_voltage_unbalance_scenario()
    text = (scenario.notes + " " + scenario.supply.notes).lower()
    assert "supply" in text
    assert "not a confirmed internal motor fault" in text
    assert "winding" in text


def test_finite_and_balanced_three_wire() -> None:
    unbalance = _run(fault_03_supply_voltage_unbalance_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc", "v_abc"):
        assert np.all(np.isfinite(getattr(unbalance, name))), name
    np.testing.assert_allclose(np.sum(unbalance.i_abc, axis=0), 0.0, rtol=0.0, atol=1e-12)


def test_supply_voltage_qd_balanced_control_path_is_exact() -> None:
    scenario = first_milestone_scenario()
    v_qs, v_ds, v_0 = supply_voltage_qd(0.37, scenario)
    assert v_qs == scenario.supply.phase_peak_v
    assert v_ds == 0.0
    assert v_0 == 0.0
    v_abc, v_qs_arr, v_ds_arr, v_0_arr = supply_voltage_traces(
        np.linspace(0.0, 0.01, 5), scenario
    )
    assert np.all(v_qs_arr == scenario.supply.phase_peak_v)
    assert np.all(v_ds_arr == 0.0)
    assert np.all(v_0_arr == 0.0)
    assert v_abc.shape == (3, 5)
