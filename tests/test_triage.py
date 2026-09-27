"""Tests for deterministic, source-data-only rule-based triage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from imcm.reporting.triage import (
    RULE_LABELS,
    TriageResult,
    evaluate_triage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CONDITION_FILES = {
    "healthy": "healthy_startup_summary.json",
    "fault-01": "fault_01_stator_resistance_imbalance_summary.json",
    "condition-02": "fault_02_increased_mechanical_load_summary.json",
    "fault-03": "fault_03_supply_voltage_unbalance_summary.json",
    "fault-04": "fault_04_bearing_outer_race_summary.json",
    "fault-05": "fault_05_rotor_asymmetry_summary.json",
}


def load_json(filename: str) -> dict[str, Any]:
    with (RESULTS_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def provenance_summary(**fields: Any) -> dict[str, Any]:
    return {
        "label": "synthetic_simulated_record",
        "parameter_provenance": "literature_example",
        "metrics": fields,
    }


def test_all_six_rule_ids_and_labels_are_stable() -> None:
    result = evaluate_triage(
        provenance_summary(
            current_unbalance_pct=1.0,
            negative_sequence_a=1.0,
            supply_voltage_unbalance_pct=1.0,
            final_slip=1.0,
        )
    )
    assert isinstance(result, TriageResult)
    assert {trace.rule_id for trace in result.rules} == {
        "R00_HEALTHY_BASELINE",
        "R01_STATOR_RESISTANCE_ASYMMETRY",
        "R02_LOAD_STEP",
        "R03_SUPPLY_UNBALANCE",
        "R04_BEARING_VIBRATION_SIGNATURE",
        "R05_ROTOR_ASYMMETRY_PROXY",
    }
    assert set(RULE_LABELS) == {
        "healthy",
        "fault-01",
        "condition-02",
        "fault-03",
        "fault-04",
        "fault-05",
    }


def test_stored_supported_records_are_classified() -> None:
    overview = load_json("fault_overview_summary.json")
    expected = {
        "healthy": "classified",
        "fault-03": "classified",
        "fault-04": "classified",
    }
    for condition_id, expected_status in expected.items():
        result = evaluate_triage(
            load_json(CONDITION_FILES[condition_id]), healthy_baseline=overview
        )
        assert result.status == expected_status, (condition_id, result.to_dict())
        assert result.condition_id == condition_id
        assert result.matched_rule_count == 1


def test_stored_missing_discriminators_fail_safely() -> None:
    overview = load_json("fault_overview_summary.json")
    expected = {
        "fault-01": {"supply_voltage_unbalance_pct"},
        "condition-02": {"current_unbalance_pct", "negative_sequence_a"},
        "fault-05": {"negative_sequence_a"},
    }
    for condition_id, expected_missing in expected.items():
        result = evaluate_triage(
            load_json(CONDITION_FILES[condition_id]), healthy_baseline=overview
        )
        assert result.status == "indeterminate", (condition_id, result.to_dict())
        assert result.condition_id is None
        assert expected_missing <= set(result.missing_metrics), (
            condition_id,
            result.missing_metrics,
        )
        assert not result.triggered_rule_ids


def test_complete_synthetic_fault_01_matches_without_fallback() -> None:
    result = evaluate_triage(
        provenance_summary(
            negative_sequence_a=0.02,
            supply_voltage_unbalance_pct=0.0,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "fault-01"
    assert result.triggered_rule_ids == ("R01_STATOR_RESISTANCE_ASYMMETRY",)


def test_complete_synthetic_condition_02_matches_without_fallback() -> None:
    result = evaluate_triage(
        provenance_summary(
            condition_final_slip=0.04,
            healthy_final_slip=0.02,
            condition_late_mean_i_qs_a=8.0,
            healthy_late_mean_i_qs_a=5.0,
            final_speed_difference_rpm=-10.0,
            current_unbalance_pct=0.0,
            negative_sequence_a=0.0,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "condition-02"
    assert result.triggered_rule_ids == ("R02_LOAD_STEP",)


def test_complete_synthetic_fault_03_matches_without_fallback() -> None:
    result = evaluate_triage(
        provenance_summary(
            supply_voltage_unbalance_pct=3.5,
            negative_sequence_a=1.0,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "fault-03"


def test_complete_synthetic_fault_04_matches_without_fallback() -> None:
    result = evaluate_triage(
        provenance_summary(
            electrical_traces_identical_to_healthy=True,
            envelope_bpfo_healthy=0.0,
            envelope_bpfo_fault=0.2,
            envelope_bpfo_2x_healthy=0.0,
            envelope_bpfo_2x_fault=0.1,
            bpfo_hz=87.5,
            bpfo_peak_fault_hz=87.6,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "fault-04"


def test_complete_synthetic_fault_05_matches_without_fallback() -> None:
    result = evaluate_triage(
        provenance_summary(
            negative_sequence_a=0.001,
            fundamental_hz=50.0,
            reference_slip=0.02,
            lower_sideband_hz=48.0,
            lower_sideband_healthy_a=0.001,
            lower_sideband_fault_a=0.2,
            upper_sideband_healthy_a=0.001,
            upper_sideband_fault_a=0.2,
            sync_modulation_healthy_a=0.0,
            sync_modulation_fault_a=0.3,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "fault-05"


def test_missing_metric_does_not_guess_a_condition() -> None:
    result = evaluate_triage(
        provenance_summary(negative_sequence_a=0.02)
    )
    assert result.status == "indeterminate"
    assert result.condition_id is None
    assert "supply_voltage_unbalance_pct" in result.missing_metrics
    assert not result.triggered_rule_ids


def test_non_finite_metric_is_not_evaluable() -> None:
    result = evaluate_triage(
        provenance_summary(
            negative_sequence_a=float("nan"),
            supply_voltage_unbalance_pct=0.0,
        )
    )
    assert result.status == "indeterminate"
    fault_01 = next(
        trace for trace in result.rules if trace.rule_id == "R01_STATOR_RESISTANCE_ASYMMETRY"
    )
    assert fault_01.status == "not_evaluable"
    assert "negative_sequence_a" in result.missing_metrics


def test_healthy_identity_is_not_inferred_from_metrics() -> None:
    result = evaluate_triage(
        provenance_summary(
            current_unbalance_pct=0.0,
            negative_sequence_a=0.0,
            final_slip=0.04,
            healthy_final_slip=0.02,
            late_mean_i_qs_a=8.0,
            healthy_late_mean_i_qs_a=5.0,
            final_speed_difference_rpm=-10.0,
        )
    )
    assert result.status == "classified"
    assert result.condition_id == "condition-02"
    assert result.triggered_rule_ids == ("R02_LOAD_STEP",)
    assert not result.ambiguities


def test_multiple_applicable_non_baseline_rules_are_ambiguous() -> None:
    result = evaluate_triage(
        provenance_summary(
            current_unbalance_pct=0.0,
            negative_sequence_a=1e-6,
            final_slip=0.04,
            healthy_final_slip=0.02,
            late_mean_i_qs_a=8.0,
            healthy_late_mean_i_qs_a=5.0,
            final_speed_difference_rpm=-10.0,
            fundamental_hz=50.0,
            reference_slip=0.02,
            lower_sideband_hz=48.0,
            lower_sideband_healthy_a=0.001,
            lower_sideband_fault_a=0.2,
            upper_sideband_healthy_a=0.001,
            upper_sideband_fault_a=0.2,
            sync_modulation_healthy_a=0.0,
            sync_modulation_fault_a=0.3,
        )
    )
    assert result.status == "ambiguous"
    assert result.condition_id is None
    assert set(result.triggered_rule_ids) == {
        "R02_LOAD_STEP",
        "R05_ROTOR_ASYMMETRY_PROXY",
    }
    assert result.ambiguities


@pytest.mark.parametrize(
    ("metric_name", "operator", "value", "expected", "should_pass"),
    [
        ("negative_sequence_a", ">", 0.01, 0.01, False),
        ("negative_sequence_a", ">", 0.5, 0.5, False),
        ("negative_sequence_a", "<", 0.005, 0.005, False),
        ("envelope_bpfo_fault", ">", 0.05, 0.05, False),
    ],
)
def test_strict_boundaries_are_explicit(
    metric_name: str,
    operator: str,
    value: float,
    expected: float,
    should_pass: bool,
) -> None:
    result = evaluate_triage(
        provenance_summary(
            **{
                metric_name: value,
                "supply_voltage_unbalance_pct": 0.0,
            }
        )
    )
    predicates = [
        predicate
        for trace in result.rules
        for predicate in trace.predicates
        if predicate.metric_name == metric_name
    ]
    assert predicates
    assert any(
        predicate.operator == operator
        and predicate.expected == expected
        and predicate.passed is should_pass
        for predicate in predicates
    )


def test_invalid_provenance_is_not_evaluable() -> None:
    result = evaluate_triage(
        {
            "parameter_provenance": "experimental",
            "metrics": {
                "supply_voltage_unbalance_pct": 4.0,
                "negative_sequence_a": 1.0,
            },
        }
    )
    assert result.status == "not_evaluable"
    assert result.condition_id is None
    assert not result.triggered_rule_ids


def test_serialized_result_contains_no_forbidden_score_fields() -> None:
    result = evaluate_triage(
        provenance_summary(
            supply_voltage_unbalance_pct=4.0,
            negative_sequence_a=1.0,
        )
    )
    text = json.dumps(result.to_dict()).lower()
    for forbidden in (
        "probability",
        "confidence",
        "anomaly",
        "severity",
        "health-index",
        "prediction",
    ):
        assert forbidden not in text