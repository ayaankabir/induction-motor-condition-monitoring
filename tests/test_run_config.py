"""Tests for the canonical run-config records and their provenance link."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imcm.reporting.provenance import ProvenanceRecord, read_manifest
from imcm.reporting.run_config import (
    CONDITION_TYPES,
    RUN_CONFIG_SCHEMA_VERSION,
    RunConfig,
    RunConfigError,
    emit_run_artifacts,
    make_run_id,
    read_run_config,
    run_config_hash,
    write_run_config,
)


def make_valid_config() -> RunConfig:
    return RunConfig(
        schema_version=RUN_CONFIG_SCHEMA_VERSION,
        run_id="run-abcd1234abcd1234",
        condition_id="healthy",
        condition_kind="healthy_baseline_no_fault",
        scenario_name="healthy_50hz_constant_torque",
        park_convention="krause_classical_2_3",
        motor_parameters={"r_s": 1.405, "r_r": 1.395},
        solver_settings={"method": "RK45"},
        initial_condition={"state": "rest"},
        analysis_windows=[{"name": "steady_state", "start_s": 0.8, "end_s": 1.0}],
        condition_parameters={"type": "healthy"},
        outputs=["results/example.json"],
        provenance_manifest={"path": "results/example_provenance.json"},
    )


def make_valid_provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        dataset_label="simulated",
        parameter_hash="abc123",
        parameter_source="literature_example",
        solver_settings={"method": "RK45"},
        initial_condition={"description": "start from rest"},
        git_commit="unknown",
        timestamp="2026-09-19T00:00:00Z",
        output_files=["results/example.json"],
    )


# ---------------------------------------------------------------------------
# Schema and condition tags
# ---------------------------------------------------------------------------


def test_condition_types_cover_all_six_studies():
    assert CONDITION_TYPES == {
        "healthy",
        "stator_resistance_imbalance",
        "increased_mechanical_load",
        "supply_voltage_unbalance",
        "bearing_outer_race",
        "rotor_asymmetry",
    }


def test_schema_version_is_one():
    assert RUN_CONFIG_SCHEMA_VERSION == "1.0"


def test_valid_config_creation():
    config = make_valid_config()
    assert config.condition_id == "healthy"
    assert config.condition_type == "healthy"
    assert config.schema_version == RUN_CONFIG_SCHEMA_VERSION


def test_missing_condition_type_rejected():
    with pytest.raises(RunConfigError):
        RunConfig(
            schema_version=RUN_CONFIG_SCHEMA_VERSION,
            run_id="run-x",
            condition_id="healthy",
            condition_kind="healthy_baseline_no_fault",
            scenario_name="healthy_50hz_constant_torque",
            park_convention="krause_classical_2_3",
            motor_parameters={},
            solver_settings={},
            initial_condition={},
            analysis_windows=[],
            condition_parameters={},
            outputs=[],
            provenance_manifest={},
        )


def test_invalid_condition_type_rejected():
    with pytest.raises(RunConfigError):
        RunConfig.create(
            condition_id="healthy",
            condition_kind="healthy_baseline_no_fault",
            scenario_name="healthy_50hz_constant_torque",
            park_convention="krause_classical_2_3",
            motor_parameters={},
            solver_settings={},
            initial_condition={},
            analysis_windows=[],
            condition_parameters={"type": "invented-condition"},
            outputs=[],
        )


def test_unsupported_schema_version_rejected():
    with pytest.raises(RunConfigError):
        RunConfig.create(
            condition_id="healthy",
            condition_kind="healthy_baseline_no_fault",
            scenario_name="healthy_50hz_constant_torque",
            park_convention="krause_classical_2_3",
            motor_parameters={},
            solver_settings={},
            initial_condition={},
            analysis_windows=[],
            condition_parameters={"type": "healthy"},
            outputs=[],
            schema_version="99.0",
        )


# ---------------------------------------------------------------------------
# Deterministic identifiers and hashing
# ---------------------------------------------------------------------------


def test_make_run_id_is_deterministic():
    kwargs = dict(
        condition_id="fault_05",
        scenario_name="fault_05_rotor_electrical_asymmetry",
        motor_parameters={"r_s": 1.405},
        solver_settings={"method": "RK45"},
        condition_parameters={"type": "rotor_asymmetry", "severity": 0.10},
    )
    assert make_run_id(**kwargs) == make_run_id(**kwargs)
    assert make_run_id(**kwargs).startswith("run-")


def test_make_run_id_changes_with_condition_type():
    base = dict(
        condition_id="fault_05",
        scenario_name="fault_05_rotor_electrical_asymmetry",
        motor_parameters={"r_s": 1.405},
        solver_settings={"method": "RK45"},
    )
    first = make_run_id(condition_parameters={"type": "rotor_asymmetry"}, **base)
    second = make_run_id(
        condition_parameters={"type": "bearing_outer_race"}, **base
    )
    assert first != second


def test_run_config_hash_is_deterministic():
    config = make_valid_config()
    assert run_config_hash(config) == run_config_hash(config)
    # Hash accepts a plain mapping too and matches the object hash.
    assert run_config_hash(config.to_dict()) == run_config_hash(config)


def test_run_config_hash_changes_with_content():
    config = make_valid_config()
    other = make_valid_config()
    other = RunConfig.from_dict({**other.to_dict(), "condition_id": "fault_01"})
    assert run_config_hash(config) != run_config_hash(other)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_config_is_json_serializable():
    config = make_valid_config()
    encoded = json.dumps(config.to_dict())
    decoded = json.loads(encoded)
    assert decoded["condition_parameters"]["type"] == "healthy"
    assert decoded["run_id"] == config.run_id


def test_required_fields_have_canonical_names():
    payload = make_valid_config().to_dict()
    expected = {
        "schema_version",
        "run_id",
        "condition_id",
        "condition_kind",
        "scenario_name",
        "park_convention",
        "motor_parameters",
        "solver_settings",
        "initial_condition",
        "analysis_windows",
        "condition_parameters",
        "outputs",
        "provenance_manifest",
    }
    assert set(payload) == expected


def test_round_trip_dict():
    config = make_valid_config()
    restored = RunConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_write_and_read_run_config(tmp_path: Path):
    config = make_valid_config()
    path = tmp_path / "run_config.json"
    written = write_run_config(path, config)
    assert written == path
    assert path.exists()

    reloaded = read_run_config(path)
    assert reloaded.run_id == config.run_id
    assert reloaded.condition_type == "healthy"
    assert run_config_hash(reloaded) == run_config_hash(config)


# ---------------------------------------------------------------------------
# Provenance link
# ---------------------------------------------------------------------------


def test_emit_run_artifacts_cross_links(tmp_path: Path):
    root = tmp_path
    results = root / "results"
    results.mkdir()

    config = make_valid_config()
    config.provenance_manifest = {
        "path": "results/example_provenance.json"
    }
    provenance = make_valid_provenance()

    config_path, manifest_path = emit_run_artifacts(
        config,
        provenance,
        run_config_path=results / "example_run_config.json",
        manifest_path=results / "example_provenance.json",
        root=root,
    )

    assert config_path.exists()
    assert manifest_path.exists()

    manifest = read_manifest(manifest_path)
    assert manifest.run_config is not None
    assert manifest.run_config["run_id"] == config.run_id
    assert manifest.run_config["schema_version"] == RUN_CONFIG_SCHEMA_VERSION
    assert manifest.run_config["hash_sha256"] == run_config_hash(config)
    assert manifest.run_config["path"] == "results/example_run_config.json"


def test_manifest_round_trips_run_config_link(tmp_path: Path):
    provenance = make_valid_provenance()
    provenance.run_config = {"path": "results/x.json", "run_id": "run-x"}
    path = tmp_path / "manifest.json"
    from imcm.reporting.provenance import write_manifest

    write_manifest(path, provenance)
    restored = read_manifest(path)
    assert restored.run_config == {"path": "results/x.json", "run_id": "run-x"}


def test_provenance_without_run_config_still_valid():
    provenance = make_valid_provenance()
    assert provenance.run_config is None
    payload = provenance.to_dict()
    assert "run_config" not in payload