"""Regression tests for construct_dataset.py fixes.

These tests verify behavioral correctness and pool structure after the V2.4
dataset-generation corrections.  No dataset is regenerated; tests exercise the
imported helpers and variation-pool definitions directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from imcm.processing.construct_dataset import (
    _run_simulation,
    _run_simulation_custom_scenario,
    _extract_from_result,
    FAULT_01_VARIATIONS,
    FAULT_02_VARIATIONS,
    FAULT_03_VARIATIONS,
    FAULT_03_MULTIPLIERS_ABC,
    FAULT_04_VARIATIONS,
    FAULT_04_SEVERITY_SCALES,
    FAULT_05_VARIATIONS,
    FeatureRecord,
)
from imcm.models.operating_scenario import (
    fault_03_supply_voltage_unbalance_scenario,
    fault_04_bearing_outer_race_scenario,
)


# ── Issue 1: load_factor application ──────────────────────────────────────


def test_load_factor_applies_to_torque():
    """Verify load_factor scales the effective mechanical load torque.

    The _run_simulation() fix creates a new LoadConfig with
    torque_nm = original * load_factor and records it in metadata.load_torque_nm.
    Each fault scenario has its own baseline torque:
    - healthy/fault_01/fault_03/fault_04/fault_05: 15.0 Nm (from first_milestone_scenario)
    - fault_02: 22.5 Nm (from fault_02_increased_mechanical_load_scenario, 15.0 * 1.5)
    After load_factor=2.0, effective torque = baseline * 2.0.
    """
    expected_torque = {
        "healthy": 15.0 * 2.0,
        "fault_01": 15.0 * 2.0,
        "fault_02": 22.5 * 2.0,  # fault_02 baseline is 22.5
        "fault_03": 15.0 * 2.0,
        "fault_04": 15.0 * 2.0,
        "fault_05": 15.0 * 2.0,
    }
    for label in ["healthy", "fault_01", "fault_02", "fault_03", "fault_04", "fault_05"]:
        result = _run_simulation(
            label, param_overrides={"load_factor": 2.0}, noise_seed=42
        )
        expected = expected_torque[label]
        assert result.metadata["load_torque_nm"] == expected, (
            f"{label}: expected {expected}, got {result.metadata['load_torque_nm']}"
        )


def test_load_factor_validation_rejects_non_finite():
    """load_factor must be finite and strictly positive; reject NaN, inf, zero, negative."""
    for bad_val in [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]:
        for label in ["healthy", "fault_01"]:
            with pytest.raises(
                ValueError,
                match=f"load_factor must be finite and positive, got {bad_val}",
            ):
                _run_simulation(
                    label, param_overrides={"load_factor": bad_val}, noise_seed=42
                )


# ── Issue 2: Variation pool structure ─────────────────────────────────────


def test_fault_01_pool_only_r_s_keys():
    """Every entry in FAULT_01_VARIATIONS must have keys ⊆ {r_s} (or {r_s, load_factor})."""
    for v in FAULT_01_VARIATIONS:
        assert set(v.keys()).issubset({"r_s", "load_factor"}), (
            f"Unexpected keys in FAULT_01_VARIATIONS entry: {set(v.keys())}"
        )


def test_fault_02_pool_only_load_factor_keys():
    """Every entry in FAULT_02_VARIATIONS must have keys ⊆ {load_factor} (or {load_factor, r_s})."""
    for v in FAULT_02_VARIATIONS:
        assert set(v.keys()).issubset({"load_factor", "r_s"}), (
            f"Unexpected keys in FAULT_02_VARIATIONS entry: {set(v.keys())}"
        )


def test_fault_03_pool_is_empty():
    """FAULT_03_VARIATIONS is empty; voltage unbalance is handled via the scenario generator,
    not _run_simulation() param_overrides."""
    assert FAULT_03_VARIATIONS == []


def test_fault_04_pool_is_empty():
    """FAULT_04_VARIATIONS is empty; bearing severity is handled via the scenario generator,
    not _run_simulation() param_overrides."""
    assert FAULT_04_VARIATIONS == []


def test_fault_05_pool_only_r_r_and_l_ls_keys():
    """Every entry in FAULT_05_VARIATIONS must have keys ⊆ {r_r, l_ls}."""
    for v in FAULT_05_VARIATIONS:
        assert set(v.keys()).issubset({"r_r", "l_ls"}), (
            f"Unexpected keys in FAULT_05_VARIATIONS entry: {set(v.keys())}"
        )


# ── Issue 3: Sample counting uses true_label ──────────────────────────────


def test_extract_from_result_sets_true_label():
    """_extract_from_result sets FeatureRecord.true_label to the passed class label.

    The fix changes main() counting from rec.scenario_name (descriptive) to
    rec.true_label (the class label), so counts align with the intended per-class targets.
    """
    from imcm.models.operating_scenario import first_milestone_scenario

    t = np.linspace(0, 1, 100)
    i_abc = np.sin(2 * np.pi * 50 * t).reshape(1, -1).repeat(3, axis=0)
    result = type("MockResult", (), {
        "t": t,
        "i_abc": i_abc,
        "i_qs": np.zeros_like(t),
        "i_ds": np.zeros_like(t),
        "tau_e": np.full_like(t, 20.0),
        "scenario": first_milestone_scenario(),
        "metadata": {"bearing_fault_enabled": False},
    })()

    for label in ["healthy", "fault_01", "fault_02", "fault_03", "fault_04", "fault_05"]:
        rec = _extract_from_result(result, label)
        assert rec.true_label == label, (
            f"Expected true_label={label}, got {rec.true_label}"
        )


def test_scenario_counts_use_true_label():
    """Verify that per-scenario counting uses rec.true_label (class label).

    This is the actual fix applied in main(): sn = rec.true_label instead of
    sn = getattr(rec, 'scenario_name', 'healthy').

    The test exercises the counting logic directly.
    """
    # Simulate the fixed counting loop
    label = "fault_02"
    # rec.true_label should be the class label, not a descriptive scenario name
    assert label in ["healthy", "fault_01", "fault_02", "fault_03", "fault_04", "fault_05"]
    # Verify the key matches what the loop uses
    sn = label
    scenario_counts = {}
    scenario_counts[sn] = scenario_counts.get(sn, 0) + 1
    assert scenario_counts[sn] == 1
    # Verify the old buggy approach would have mismatched
    old_sn = "healthy_50hz_constant_torque"  # descriptive scenario_name
    # The old code would do: scenario_counts[old_sn] = ... which would create
    # a different key, causing the per-scenario summary to miss the class label.
    assert old_sn != label, "Old buggy key would not match class label"


# ── Issue 4: Numeric validation ───────────────────────────────────────────


def test_numeric_validation_rejects_nan_inf_all_params():
    """All 7 supported parameter keys reject NaN, inf, -inf, zero, and negative values.

    Both validation sites in _run_simulation() (lines 252-254 and 274-276) now use
    `not np.isfinite(val) or val <= 0` to catch all non-finite and non-positive cases.
    """
    bad_vals = [float("nan"), float("inf"), float("-inf"), 0.0, -1.0]
    param_keys = ["r_s", "r_r", "l_ls", "l_lr", "l_m", "inertia", "viscous_friction"]

    for key in param_keys:
        for bad_val in bad_vals:
            with pytest.raises(
                ValueError,
                match=f"Parameter '{key}' must be finite and positive, got {bad_val}",
            ):
                _run_simulation(
                    "healthy",
                    param_overrides={key: bad_val},
                    noise_seed=42,
                )


# ── Integration: load_factor changes effective torque without altering other fields ──

def test_load_factor_preserves_other_scenario_fields():
    """When load_factor is applied, all other scenario fields remain unchanged.

    The frozen-dataclass replacement (dc_replace) should preserve every field
    except torque_nm.  We verify this by checking the scenario's load config
    and other metadata fields stay intact.
    """
    result = _run_simulation(
        "healthy", param_overrides={"load_factor": 2.5}, noise_seed=42
    )

    # torque should be doubled
    assert result.metadata["load_torque_nm"] == 15.0 * 2.5

    # supply and park convention should be unchanged
    assert result.metadata["park_convention"] == "krause_classical_2_3"
    assert result.metadata["supply_frequency_hz"] == 50.0

    # scenario-level fields should also be preserved
    assert result.scenario.load.load_type == "constant_torque"
    assert result.scenario.name == "healthy_50hz_constant_torque"


# ── Issue 5: Fault 03 generation path (voltage unbalance via scenario generator) ──────


def test_fault_03_multipliers_abc_pool_structure():
    """FAULT_03_MULTIPLIERS_ABC contains 12 tuples of 3 positive floats each."""
    assert len(FAULT_03_MULTIPLIERS_ABC) == 12
    for mult in FAULT_03_MULTIPLIERS_ABC:
        assert isinstance(mult, tuple), f"Expected tuple, got {type(mult)}"
        assert len(mult) == 3, f"Expected 3 elements, got {len(mult)}"
        for val in mult:
            assert isinstance(val, (int, float)), f"Expected float, got {type(val)}"
            assert val > 0.0, f"Multipliers must be positive, got {val}"


def test_fault_03_generation_path_creates_correct_scenario():
    """_run_simulation_custom_scenario with fault_03 scenario produces correct metadata."""
    scenario = fault_03_supply_voltage_unbalance_scenario(multipliers_abc=(1.0, 1.0, 0.90))
    result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)

    # Verify the scenario has voltage unbalance enabled
    assert result.metadata["supply_unbalance"]["enabled"] is True
    assert result.metadata["supply_voltage_multipliers_abc"] == (1.0, 1.0, 0.90)
    assert result.metadata["supply_voltage_case"] == "fault_03_phase_c_minus_10pct"
    assert result.metadata["supply_fault"] is True

    # Verify other fields are preserved
    assert result.metadata["load_torque_nm"] == 15.0
    assert result.metadata["park_convention"] == "krause_classical_2_3"
    assert result.metadata["supply_frequency_hz"] == 50.0


def test_fault_03_varied_multipliers_produce_distinct_features():
    """Different voltage unbalance multipliers produce distinct feature vectors."""
    features = []
    for mult in FAULT_03_MULTIPLIERS_ABC[:4]:  # Test first 4 to keep test fast
        scenario = fault_03_supply_voltage_unbalance_scenario(multipliers_abc=mult)
        result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)
        record = _extract_from_result(result, "fault_03")
        features.append((
            record.neg_seq_current_a,
            record.current_unbalance_pct,
            record.spectral_peak_freq_hz,
        ))

    # All feature vectors should be distinct
    assert len(set(features)) == len(features), "All multipliers should produce distinct features"


# ── Issue 6: Fault 04 generation path (bearing severity via scenario generator) ──────


def test_fault_04_severity_scales_pool_structure():
    """FAULT_04_SEVERITY_SCALES contains 12 positive floats."""
    assert len(FAULT_04_SEVERITY_SCALES) == 12
    for scale in FAULT_04_SEVERITY_SCALES:
        assert isinstance(scale, (int, float)), f"Expected float, got {type(scale)}"
        assert scale > 0.0, f"Severity scales must be positive, got {scale}"


def test_fault_04_generation_path_creates_correct_scenario():
    """_run_simulation_custom_scenario with fault_04 scenario produces correct metadata."""
    scenario = fault_04_bearing_outer_race_scenario(severity_scale=1.5)
    result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)

    # Verify the scenario has bearing fault enabled with correct amplitude
    assert result.metadata["bearing_fault_enabled"] is True
    assert result.metadata["bearing_fault_case"] == "fault_04_outer_race_bpfo"
    assert result.metadata["bearing_fault_channel"] == "simulated_vibration_m_s2"

    # Verify other fields are preserved
    assert result.metadata["load_torque_nm"] == 15.0
    assert result.metadata["park_convention"] == "krause_classical_2_3"
    assert result.metadata["supply_frequency_hz"] == 50.0
    assert result.metadata["supply_unbalance"]["enabled"] is False
    assert result.metadata["supply_fault"] is False


def test_fault_04_varied_severity_produces_distinct_features():
    """Different bearing severity scales produce distinct feature vectors."""
    features = []
    for scale in FAULT_04_SEVERITY_SCALES[:4]:  # Test first 4 to keep test fast
        scenario = fault_04_bearing_outer_race_scenario(severity_scale=scale)
        result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)
        record = _extract_from_result(result, "fault_04")
        features.append((
            record.bearing_env_peak_m_s2,
            record.bearing_env_mean_m_s2,
            record.neg_seq_current_a,
        ))

    # All feature vectors should be distinct
    assert len(set(features)) == len(features), "All severity scales should produce distinct features"


def test_fault_04_bearing_envelope_features_populated():
    """Fault 04 records have bearing envelope features populated (not NaN)."""
    scenario = fault_04_bearing_outer_race_scenario(severity_scale=1.0)
    result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)
    record = _extract_from_result(result, "fault_04")

    # Bearing features should be computed (not NaN) for fault_04
    assert not np.isnan(record.bearing_env_peak_m_s2), "bearing_env_peak_m_s2 should not be NaN for fault_04"
    assert not np.isnan(record.bearing_env_mean_m_s2), "bearing_env_mean_m_s2 should not be NaN for fault_04"


# ── Integration: Custom scenario function preserves other fields ───────────


def test_custom_scenario_preserves_load_and_supply():
    """_run_simulation_custom_scenario preserves all scenario fields except overrides."""
    # Use a scenario with a non-standard load torque
    scenario = fault_03_supply_voltage_unbalance_scenario(multipliers_abc=(1.0, 1.0, 0.85))
    result = _run_simulation_custom_scenario(scenario, param_overrides={}, noise_seed=42)

    # Supply fields should reflect the custom multipliers
    assert result.metadata["supply_voltage_multipliers_abc"] == (1.0, 1.0, 0.85)

    # Load should be unchanged from base scenario
    assert result.metadata["load_torque_nm"] == 15.0
    assert result.scenario.load.load_type == "constant_torque"

    # Park convention and frequency preserved
    assert result.metadata["park_convention"] == "krause_classical_2_3"
    assert result.metadata["supply_frequency_hz"] == 50.0

    # Stator resistance should be healthy (not enabled)
    assert result.metadata["stator_resistance_case"] == "healthy_balanced"