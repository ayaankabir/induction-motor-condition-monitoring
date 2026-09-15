import numpy as np

from imcm.models.operating_scenario import PARK_CONVENTION, first_milestone_scenario
from imcm.signals.park import abc_to_qd0, balanced_phase_voltages, qd0_to_abc


def test_park_round_trip_with_zero_sequence() -> None:
    rng = np.random.default_rng(0)
    f_abc = rng.normal(size=3)
    theta = 0.7
    f_qd0 = abc_to_qd0(f_abc, theta)
    recovered = qd0_to_abc(f_qd0, theta)
    np.testing.assert_allclose(recovered, f_abc, rtol=0.0, atol=1e-12)


def test_balanced_supply_is_dc_on_q_axis() -> None:
    scenario = first_milestone_scenario()
    assert scenario.park_convention == PARK_CONVENTION
    supply = scenario.supply
    t = np.linspace(0.0, 0.04, 200)
    v_abc = balanced_phase_voltages(t, supply.omega_e, supply.phase_peak_v)
    theta = supply.omega_e * t
    v_qd0 = abc_to_qd0(v_abc, theta)
    np.testing.assert_allclose(v_qd0[0], supply.phase_peak_v, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(v_qd0[1], 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(v_qd0[2], 0.0, rtol=0.0, atol=1e-10)
