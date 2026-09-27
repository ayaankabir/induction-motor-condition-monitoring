"""Tests for the diagnostic explanation generator (src/imcm/reporting/explanation.py)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from imcm.reporting.explanation import (
    ConditionExplanation,
    MetricEvidence,
    generate_explanation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CONDITIONS = [
    ("healthy", "healthy_startup_summary.json", "baseline"),
    ("fault-01", "fault_01_stator_resistance_imbalance_summary.json", "internal-fault-proxy"),
    ("condition-02", "fault_02_increased_mechanical_load_summary.json", "operating-condition"),
    ("fault-03", "fault_03_supply_voltage_unbalance_summary.json", "supply-confounder"),
    ("fault-04", "fault_04_bearing_outer_race_summary.json", "vibration-signature"),
    ("fault-05", "fault_05_rotor_asymmetry_summary.json", "proxy-fault"),
]


@pytest.mark.parametrize("condition_id, filename, expected_category", CONDITIONS)
def test_generate_explanation_all_conditions(
    condition_id: str, filename: str, expected_category: str
) -> None:
    file_path = RESULTS_DIR / filename
    assert file_path.exists(), f"Summary file missing: {filename}"
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    expl = generate_explanation(condition_id, data)
    assert isinstance(expl, ConditionExplanation)
    assert expl.condition_id == condition_id
    assert expl.classification_category == expected_category
    assert len(expl.classification_label) > 0
    assert len(expl.physical_mechanism) > 0
    assert len(expl.why_classified) > 0
    assert len(expl.supporting_evidence) > 0
    assert len(expl.discrimination_vs_confounders) > 0
    assert len(expl.standing_limitations) > 0

    # Ensure all evidence has non-empty fields and valid roles
    valid_roles = {
        "primary_discriminant",
        "confounder_check",
        "normal_baseline",
        "unaffected_channel",
    }
    for ev in expl.supporting_evidence:
        assert isinstance(ev, MetricEvidence)
        assert ev.diagnostic_role in valid_roles
        assert len(ev.metric_name) > 0
        assert len(ev.interpretation) > 0

    # Serialization test
    d = expl.to_dict()
    assert isinstance(d, dict)
    assert d["condition_id"] == condition_id
    assert len(d["supporting_evidence"]) == len(expl.supporting_evidence)


def test_explanation_discrimination_content() -> None:
    """Verify specific discrimination points match scientific ground truth."""
    # Fault 01 vs Fault 03 separation
    with (RESULTS_DIR / "fault_01_stator_resistance_imbalance_summary.json").open() as f:
        f01_data = json.load(f)
    expl_f01 = generate_explanation("fault-01", f01_data)
    assert any("Fault 03" in disc for disc in expl_f01.discrimination_vs_confounders)
    assert any(ev.metric_name == "supply_voltage_unbalance_pct" for ev in expl_f01.supporting_evidence)

    # Fault 04 auxiliary vibration channel
    with (RESULTS_DIR / "fault_04_bearing_outer_race_summary.json").open() as f:
        f04_data = json.load(f)
    expl_f04 = generate_explanation("fault-04", f04_data)
    assert any(ev.metric_name == "electrical_traces_identical_to_healthy" for ev in expl_f04.supporting_evidence)

    # Fault 05 sidebands
    with (RESULTS_DIR / "fault_05_rotor_asymmetry_summary.json").open() as f:
        f05_data = json.load(f)
    expl_f05 = generate_explanation("fault-05", f05_data)
    assert any("sideband" in ev.metric_name.lower() for ev in expl_f05.supporting_evidence)
