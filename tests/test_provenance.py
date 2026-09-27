import json
from pathlib import Path

import pytest

from imcm.reporting.provenance import (
    ALLOWED_DATASET_LABELS,
    ProvenanceRecord,
    write_manifest,
)


def make_valid_record():
    return ProvenanceRecord(
        dataset_label="simulated",
        parameter_hash="abc123",
        parameter_source="software-defined reduced-order model",
        solver_settings={"method": "RK45"},
        initial_condition={"description": "start from rest"},
        git_commit="unknown",
        timestamp="2026-09-19T00:00:00Z",
        output_files=["results/example.json"],
    )


def test_allowed_dataset_labels():
    assert {
        "simulated",
        "literature-example",
        "experimental",
    }.issubset(set(ALLOWED_DATASET_LABELS))


def test_valid_record_creation():
    record = make_valid_record()

    assert record.dataset_label == "simulated"
    assert record.parameter_hash == "abc123"


def test_invalid_dataset_label_rejected():
    with pytest.raises((ValueError, TypeError)):
        ProvenanceRecord(
            dataset_label="invented-label",
            parameter_hash="abc123",
            parameter_source="test",
            solver_settings={},
            initial_condition={"description": "rest"},
            git_commit="unknown",
            timestamp="2026-09-19T00:00:00Z",
            output_files=[],
        )


def test_record_is_json_serializable():
    record = make_valid_record()

    if hasattr(record, "to_dict"):
        payload = record.to_dict()
    elif hasattr(record, "as_dict"):
        payload = record.as_dict()
    else:
        payload = record.__dict__

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert json.loads(encoded)["dataset_label"] == "simulated"


def test_manifest_is_written(tmp_path: Path):
    record = make_valid_record()
    output_path = tmp_path / "provenance.json"

    result = write_manifest(output_path, record)

    assert output_path.exists()

    data = json.loads(output_path.read_text())

    assert data["dataset_label"] == "simulated"

    # Support either returning the path or returning None.
    assert result is None or Path(result) == output_path