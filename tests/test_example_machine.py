from imcm.models.operating_scenario import first_milestone_scenario
from imcm.models.parameters import illustrative_4kw_400v_50hz_4pole


def test_example_machine_is_literature_not_measured() -> None:
    motor = illustrative_4kw_400v_50hz_4pole()
    assert motor.provenance == "literature_example"
    assert motor.rated_frequency_hz == 50.0
    assert motor.n_poles == 4
    assert motor.stator_connection == "star"
    assert motor.assumed_rated_torque_nm is not None
    assert 26.0 < motor.assumed_rated_torque_nm < 27.5


def test_first_milestone_load_is_constant_torque() -> None:
    scenario = first_milestone_scenario()
    assert scenario.load.load_type == "constant_torque"
    assert scenario.load.torque_nm == 15.0
    assert scenario.supply.frequency_hz == 50.0
    assert scenario.supply.waveform == "balanced_sinusoidal"
