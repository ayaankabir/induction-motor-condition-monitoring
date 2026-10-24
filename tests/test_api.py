"""Integration tests for the read-only condition-monitoring FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path for api.main import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import CONDITION_FILES, app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["read_only"] is True
    assert data["experimental_validation"] is False
    assert "data_source" in data


def test_get_all_conditions(client: TestClient) -> None:
    response = client.get("/api/conditions")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(CONDITION_FILES)
    assert len(data["conditions"]) == len(CONDITION_FILES)

    condition_ids = {c["condition_id"] for c in data["conditions"]}
    assert condition_ids == set(CONDITION_FILES.keys())

    for record in data["conditions"]:
        assert "condition_id" in record
        assert "source_file" in record
        assert "data" in record
        assert isinstance(record["data"], dict)
        assert "explanation" in record
        assert isinstance(record["explanation"], dict)
        assert record["explanation"]["condition_id"] == record["condition_id"]


@pytest.mark.parametrize("condition_id", list(CONDITION_FILES.keys()))
def test_get_individual_condition(client: TestClient, condition_id: str) -> None:
    response = client.get(f"/api/conditions/{condition_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["condition_id"] == condition_id
    assert data["source_file"] == CONDITION_FILES[condition_id]
    assert "data" in data
    assert "explanation" in data
    expl = data["explanation"]
    assert expl["condition_id"] == condition_id
    assert len(expl["supporting_evidence"]) > 0
    assert len(expl["discrimination_vs_confounders"]) > 0
    assert len(expl["standing_limitations"]) > 0
    # Mandatory scientific provenance checks in each summary payload
    payload = data["data"]
    assert "provenance" in payload or "parameter_provenance" in payload


def test_get_invalid_condition_returns_404(client: TestClient) -> None:
    response = client.get("/api/conditions/non_existent_condition_id")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Unknown condition ID" in detail


def test_simulate_endpoint_healthy(client: TestClient) -> None:
    """Regression test for /api/simulate returning 200 with valid simulation outputs."""
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.05&max_step=0.0001&output_dt=0.0001&rtol=0.000001&atol=1e-8"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_name"] == "healthy_50hz_constant_torque"
    assert data["t_end_s"] == 0.05
    assert "outputs" in data
    outputs = data["outputs"]
    assert "three_phase_currents_a" in outputs
    assert len(outputs["three_phase_currents_a"]) > 0
    first_pt = outputs["three_phase_currents_a"][0]
    assert "time_s" in first_pt
    assert "current_a" in first_pt
    assert "metadata" in data
    assert data["metadata"]["provenance"] == "simulated"
    assert data["metadata"]["experimental_validation"] is False


def test_simulate_endpoint_mqtt_streaming(client: TestClient) -> None:
    """Regression test for /api/simulate with mqtt_enabled=true."""
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.02&mqtt_enabled=true&mqtt_chunk_size=50"
    )
    assert response.status_code == 200
    data = response.json()
    assert "mqtt_streaming_report" in data
    report = data["mqtt_streaming_report"]
    assert "current_a" in report
    assert report["current_a"]["total_packets_received"] > 0
    assert report["current_a"]["duplicate_count"] == 0
