from imcm.models.parameters import InductionMotorParameters


def test_inductance_sums() -> None:
    p = InductionMotorParameters(
        r_s=1.0,
        r_r=1.0,
        l_ls=0.01,
        l_lr=0.01,
        l_m=0.2,
        n_poles=4,
        inertia=0.05,
        viscous_friction=0.0,
        rated_frequency_hz=50.0,
        rated_line_line_voltage_v=400.0,
        provenance="unknown",
        name="dummy_values_for_unit_test_only",
        notes="Arbitrary numbers to test the dataclass. Not a machine under study.",
    )
    assert abs(p.l_s - (p.l_ls + p.l_m)) < 1e-18
    assert p.pole_pairs == 2
