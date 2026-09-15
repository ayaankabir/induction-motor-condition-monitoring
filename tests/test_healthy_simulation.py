import numpy as np
import pytest

from imcm.models.fifth_order_dq import rest_initial_state, simulate_healthy
from imcm.models.operating_scenario import PARK_CONVENTION, first_milestone_scenario
from imcm.signals.park import abc_to_qd0, qd0_to_abc
from imcm.validation.equivalent_circuit import equivalent_circuit_point
from imcm.validation.frequency import interpolated_rfft_peak_hz, mean_zero_crossing_frequency_hz
from imcm.validation.power_balance import power_balance_window


STEADY_T = 0.8
FREQ_WINDOW_S = 0.4


@pytest.fixture(scope="module")
def startup():
    return simulate_healthy(t_end=1.0, max_step=1.0e-4, output_dt=2.0e-4)


def _steady(startup):
    return startup.t >= STEADY_T


def test_rest_initial_state_is_five_zeros() -> None:
    y0 = rest_initial_state()
    assert y0.shape == (5,)
    assert np.all(y0 == 0.0)


def test_rk45_finishes_with_finite_states(startup) -> None:
    assert startup.metadata["solver"] == "RK45"
    assert startup.metadata["initial_condition"] == "rest"
    assert startup.metadata["provenance"] == "simulated"
    assert startup.metadata["parameter_provenance"] == "literature_example"
    assert startup.metadata["experimental_validation"] is False
    assert startup.metadata["faults"] is False
    assert startup.metadata["park_convention"] == PARK_CONVENTION
    names = (
        "t",
        "lambda_qs",
        "lambda_ds",
        "lambda_qr",
        "lambda_dr",
        "omega_m",
        "i_qs",
        "i_ds",
        "i_qr",
        "i_dr",
        "tau_e",
        "tau_l",
        "v_qs",
        "v_ds",
    )
    for name in names:
        assert np.all(np.isfinite(getattr(startup, name))), name
    assert np.all(np.isfinite(startup.i_abc))
    assert np.all(np.isfinite(startup.v_abc))


def test_starts_from_rest(startup) -> None:
    np.testing.assert_allclose(startup.omega_m[0], 0.0, atol=1e-12)
    for name in ("lambda_qs", "lambda_ds", "lambda_qr", "lambda_dr"):
        np.testing.assert_allclose(getattr(startup, name)[0], 0.0, atol=1e-12)


def test_phase_currents_sum_to_zero(startup) -> None:
    """Isolated-neutral Krause reconstruction with i0=0 implies ia+ib+ic=0."""
    residual = startup.i_abc[0] + startup.i_abc[1] + startup.i_abc[2]
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-12)


def test_park_roundtrip_on_simulated_currents(startup) -> None:
    """Same θ=ωe t and same 2/3 transform as the voltage identity."""
    theta = startup.scenario.supply.omega_e * startup.t
    i_qd0 = abc_to_qd0(startup.i_abc, theta)
    np.testing.assert_allclose(i_qd0[0], startup.i_qs, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(i_qd0[1], startup.i_ds, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(i_qd0[2], 0.0, rtol=0.0, atol=1e-10)
    recovered = qd0_to_abc(i_qd0, theta)
    np.testing.assert_allclose(recovered, startup.i_abc, rtol=0.0, atol=1e-10)


def test_applied_abc_voltages_are_synchronous_q_axis_dc(startup) -> None:
    theta = startup.scenario.supply.omega_e * startup.t
    v_qd0 = abc_to_qd0(startup.v_abc, theta)
    np.testing.assert_allclose(v_qd0[0], startup.v_qs, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(v_qd0[1], startup.v_ds, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(startup.v_qs, startup.scenario.supply.phase_peak_v, atol=1e-12)
    np.testing.assert_allclose(startup.v_ds, 0.0, atol=1e-12)


def test_motoring_slip_and_torque_balance(startup) -> None:
    mask = _steady(startup)
    slip = float(np.mean(startup.slip[mask]))
    tau_e = float(np.mean(startup.tau_e[mask]))
    omega_m = float(np.mean(startup.omega_m[mask]))
    tau_l = startup.scenario.load.torque_nm
    tau_b = startup.params.viscous_friction * omega_m
    assert slip > 0.0
    assert 0.01 < slip < 0.08
    assert np.mean(startup.speed_rpm[mask]) < startup.params.synchronous_speed_rpm
    np.testing.assert_allclose(tau_e, tau_l + tau_b, rtol=1e-4, atol=1e-3)


def test_synchronous_frame_currents_settle(startup) -> None:
    mask = _steady(startup)
    i_qs = startup.i_qs[mask]
    i_ds = startup.i_ds[mask]
    assert np.std(i_qs) < 0.02 * max(abs(np.mean(i_qs)), 1.0)
    assert np.std(i_ds) < 0.02 * max(abs(np.mean(i_ds)), 1.0)


def test_steady_power_balance(startup) -> None:
    pb = power_balance_window(startup, t_start=STEADY_T)
    assert pb.residual_rel < 1e-6
    assert abs(pb.p_mag_dot_mean) < 1.0


def test_equivalent_circuit_agrees_at_same_slip(startup) -> None:
    mask = _steady(startup)
    slip = float(np.mean(startup.slip[mask]))
    tau_e = float(np.mean(startup.tau_e[mask]))
    i_rms = float(np.sqrt(np.mean(startup.i_abc[0, mask] ** 2)))
    ec = equivalent_circuit_point(
        startup.params,
        slip,
        line_line_rms_v=startup.scenario.supply.line_line_rms_v,
        frequency_hz=startup.scenario.supply.frequency_hz,
    )
    np.testing.assert_allclose(tau_e, ec.electromagnetic_torque_nm, rtol=1e-3, atol=0.05)
    np.testing.assert_allclose(i_rms, ec.stator_current_rms_a, rtol=1e-3, atol=0.05)


def test_late_currents_are_near_50_hz(startup) -> None:
    """FFT peak and zero-crossing period must both sit at the supply frequency.

    Window length is 0.4 s so the raw DFT bin spacing is 2.5 Hz. Interpolation
    only refines within a bin; see tests/test_frequency.py for 40/60 Hz rejects.
    """
    t_start = startup.t[-1] - FREQ_WINDOW_S
    mask = startup.t >= t_start
    t = startup.t[mask]
    ia = startup.i_abc[0, mask]
    fft_peak = interpolated_rfft_peak_hz(t, ia)
    zc_peak = mean_zero_crossing_frequency_hz(t, ia)
    assert abs(fft_peak - 50.0) < 0.5
    assert abs(zc_peak - 50.0) < 0.5
    scenario = first_milestone_scenario()
    assert scenario.supply.frequency_hz == 50.0
