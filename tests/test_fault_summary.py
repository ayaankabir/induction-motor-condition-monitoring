"""Tests for imcm.reporting.fault_summary overview generator."""

from __future__ import annotations

from pathlib import Path
import pytest

from imcm.reporting.fault_summary import (
    CONDITIONS,
    build_markdown_table,
    build_overview_record,
    healthy_baseline,
    load_summaries,
    measured_per_condition,
    validate_consistency,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def loaded_summaries():
    return load_summaries(RESULTS_DIR)


def test_load_summaries_contains_all_studies(loaded_summaries) -> None:
    expected_keys = {"fault_01", "fault_02", "fault_03", "fault_04", "fault_05"}
    assert set(loaded_summaries.keys()) == expected_keys
    for key, data in loaded_summaries.items():
        assert isinstance(data, dict)
        assert "provenance" in data or "parameter_provenance" in data


def test_healthy_baseline_structure(loaded_summaries) -> None:
    baseline = healthy_baseline(loaded_summaries)
    assert "final_speed_rpm" in baseline
    assert "final_slip" in baseline
    assert "phase_current_rms_a" in baseline
    assert "current_unbalance_pct" in baseline
    assert "negative_sequence_a" in baseline
    assert "torque_ripple_nm" in baseline
    assert "power_balance_residual_rel" in baseline
    assert baseline["current_unbalance_pct"] < 0.01


def test_measured_per_condition_extracts_all_five(loaded_summaries) -> None:
    baseline = healthy_baseline(loaded_summaries)
    measured = measured_per_condition(loaded_summaries, baseline)
    expected_keys = {"fault_01", "fault_02", "fault_03", "fault_04", "fault_05"}
    assert set(measured.keys()) == expected_keys
    for key, data in measured.items():
        assert "primary_signature" in data
        assert "power_balance_residual_rel" in data


def test_validate_consistency_all_pass(loaded_summaries) -> None:
    checks = validate_consistency(RESULTS_DIR, loaded_summaries)
    assert len(checks) >= 8
    for check in checks:
        assert check["status"] == "pass", f"Check {check['name']} failed: {check['detail']}"


def test_build_markdown_table_contains_all_conditions(loaded_summaries) -> None:
    baseline = healthy_baseline(loaded_summaries)
    measured = measured_per_condition(loaded_summaries, baseline)
    checks = validate_consistency(RESULTS_DIR, loaded_summaries)
    md = build_markdown_table(baseline, measured, checks)

    assert "# Simulated condition overview" in md
    for cond in CONDITIONS:
        assert cond.table_label in md
    assert "Standing limitations" in md
    assert "literature-example" in md


def test_build_overview_record_structure(loaded_summaries) -> None:
    checks = validate_consistency(RESULTS_DIR, loaded_summaries)
    record = build_overview_record(RESULTS_DIR, loaded_summaries, checks)
    assert "conditions" in record
    assert len(record["conditions"]) == 6  # healthy + 5 studies
    assert "healthy_baseline" in record
    assert "consistency_checks" in record
    assert record["all_checks_pass"] is True
    assert all(c["status"] == "pass" for c in record["consistency_checks"])
