import numpy as np

from imcm.models.flux_map import (
    currents_from_fluxes,
    electromagnetic_torque_currents,
    electromagnetic_torque_flux_current,
    fluxes_from_currents,
)
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole


def test_flux_current_round_trip() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    i_qs, i_ds, i_qr, i_dr = 4.0, -1.5, 2.0, 0.25
    fluxes = fluxes_from_currents(i_qs, i_ds, i_qr, i_dr, params)
    recovered = currents_from_fluxes(*fluxes, params)
    np.testing.assert_allclose(recovered, (i_qs, i_ds, i_qr, i_dr), rtol=0.0, atol=1e-12)


def test_torque_forms_agree_for_linear_magnetics() -> None:
    params = illustrative_4kw_400v_50hz_4pole()
    i_qs, i_ds, i_qr, i_dr = 3.0, 1.0, -2.0, 0.5
    lam_qs, lam_ds, _, _ = fluxes_from_currents(i_qs, i_ds, i_qr, i_dr, params)
    t_lambda = electromagnetic_torque_flux_current(
        lam_qs, lam_ds, i_qs, i_ds, params.n_poles
    )
    t_currents = electromagnetic_torque_currents(i_qs, i_ds, i_qr, i_dr, params)
    np.testing.assert_allclose(t_lambda, t_currents, rtol=0.0, atol=1e-12)
