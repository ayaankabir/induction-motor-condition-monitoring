from dataclasses import replace

import numpy as np

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.models.operating_scenario import (
    StatorResistanceConfig,
    fault_01_phase_a_resistance_imbalance_scenario,
    first_milestone_scenario,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole
from imcm.models.stator_resistance import phase_resistances_abc, resistance_matrix_qd
from imcm.signals.park import abc_to_qd0, qd0_to_abc
from imcm.validation.fault_metrics import CONVERGENCE_TOLERANCES, compare_stator_resistance_cases
from imcm.validation.power_balance import power_balance_window


def _run(scenario):
    return simulate_healthy(scenario=scenario, t_end=1.0, max_step=1.0e-4, output_dt=2.0e-4)


def test_fault_01_phase_resistances_are_explicit() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    scenario = fault_01_phase_a_resistance_imbalance_scenario()
    values = phase_resistances_abc(
        params.r_s,
        enabled=scenario.stator_resistance.enabled,
        multipliers_abc=scenario.stator_resistance.multipliers_abc,
    )
    np.testing.assert_allclose(values, (1.5455, 1.405, 1.405), rtol=0.0, atol=1e-12)
    assert values[0] != values[1] == values[2]


def test_equal_phase_resistances_reproduce_healthy_result() -> None:
    healthy_scenario = first_milestone_scenario()
    equal_scenario = replace(
        healthy_scenario,
        stator_resistance=StatorResistanceConfig(
            enabled=True, multipliers_abc=(1.0, 1.0, 1.0), label="equal_resistances"
        ),
    )
    healthy = _run(healthy_scenario)
    equal = _run(equal_scenario)
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc"):
        np.testing.assert_allclose(getattr(equal, name), getattr(healthy, name), rtol=0.0, atol=1e-12)


def test_projected_resistance_matches_direct_abc_transform() -> None:
    resistances = (1.5455, 1.405, 1.405)
    theta = 0.73
    i_qd0 = np.array((4.2, -1.7, 0.0))
    i_abc = qd0_to_abc(i_qd0, theta)
    direct_qd = abc_to_qd0(np.asarray(resistances) * i_abc, theta)[:2]
    projected_qd = resistance_matrix_qd(theta, resistances) @ i_qd0[:2]
    np.testing.assert_allclose(projected_qd, direct_qd, rtol=0.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(resistance_matrix_qd(theta, resistances)) > 0.0)


def test_fault_01_is_finite_balanced_and_changes_current_metric() -> None:
    healthy = _run(first_milestone_scenario())
    fault = _run(fault_01_phase_a_resistance_imbalance_scenario())
    for name in ("omega_m", "i_qs", "i_ds", "tau_e", "i_abc"):
        assert np.all(np.isfinite(getattr(fault, name))), name
    np.testing.assert_allclose(np.sum(fault.i_abc, axis=0), 0.0, rtol=0.0, atol=1e-12)
    metrics = compare_stator_resistance_cases(healthy, fault)
    assert metrics.current_unbalance_fault_pct > metrics.current_unbalance_healthy_pct + 0.1
    assert metrics.negative_sequence_fault_a > metrics.negative_sequence_healthy_a + 1.0e-3
    assert fault.metadata["faults"] is True


def test_fault_01_power_balance_uses_phase_specific_resistance() -> None:
    fault = _run(fault_01_phase_a_resistance_imbalance_scenario())
    pb = power_balance_window(fault, t_start=0.8)
    assert pb.residual_rel < 1e-6
    assert abs(pb.p_mag_dot_mean) < 1.0


def test_fault_01_metrics_converge_with_tighter_rk45_settings() -> None:
    normal_kwargs = dict(t_end=1.0, max_step=1.0e-4, output_dt=1.0e-4, rtol=1.0e-6, atol=1.0e-8)
    tight_kwargs = dict(t_end=1.0, max_step=1.0e-5, output_dt=1.0e-4, rtol=1.0e-9, atol=1.0e-11)
    normal = compare_stator_resistance_cases(
        simulate_healthy(scenario=first_milestone_scenario(), **normal_kwargs),
        simulate_healthy(scenario=fault_01_phase_a_resistance_imbalance_scenario(), **normal_kwargs),
    )
    tight = compare_stator_resistance_cases(
        simulate_healthy(scenario=first_milestone_scenario(), **tight_kwargs),
        simulate_healthy(scenario=fault_01_phase_a_resistance_imbalance_scenario(), **tight_kwargs),
    )
    assert abs(normal.current_unbalance_fault_pct - tight.current_unbalance_fault_pct) < CONVERGENCE_TOLERANCES["current_unbalance_pct"]
    assert abs(normal.negative_sequence_fault_a - tight.negative_sequence_fault_a) < CONVERGENCE_TOLERANCES["negative_sequence_a"]
    assert abs(normal.torque_ripple_fault_nm - tight.torque_ripple_fault_nm) < CONVERGENCE_TOLERANCES["torque_ripple_nm"]
    assert abs(normal.final_speed_difference_rpm - tight.final_speed_difference_rpm) < CONVERGENCE_TOLERANCES["final_speed_difference_rpm"]
    assert abs(normal.final_slip_difference - tight.final_slip_difference) < CONVERGENCE_TOLERANCES["final_slip_difference"]
