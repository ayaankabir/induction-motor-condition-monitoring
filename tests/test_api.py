"""Integration tests for the read-only condition-monitoring FastAPI service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path for api.main import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import main as api_main
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
        "/api/simulate?scenario_name=healthy&t_end=0.1&max_step=0.0001&output_dt=0.0001&rtol=0.000001&atol=1e-8"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_name"] == "healthy_50hz_constant_torque"
    assert data["t_end_s"] == 0.1
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
        "/api/simulate?scenario_name=healthy&t_end=0.1&mqtt_enabled=true&mqtt_chunk_size=50"
    )
    assert response.status_code == 200
    data = response.json()
    assert "mqtt_streaming_report" in data
    report = data["mqtt_streaming_report"]
    assert "current_a" in report
    assert report["current_a"]["total_packets_received"] > 0
    assert report["current_a"]["duplicate_count"] == 0


def test_simulate_limits_endpoint_matches_frontend_contract(client: TestClient) -> None:
    response = client.get("/api/simulate/limits")
    assert response.status_code == 200
    data = response.json()
    assert data["t_end"]["default"] == 1.0
    assert data["scenario"]["available"] == list(CONDITION_FILES)
    assert "r_s" in data["param_overrides"]["valid_keys"]
    assert data["provenance"]["always_simulated"] is True
    assert data["provenance"]["experimental_validation"] is False
    assert data["resource_limits"]["max_output_samples"] == api_main.MAX_SIMULATION_OUTPUT_SAMPLES


def test_simulation_error_response_does_not_expose_exception_details(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_private_path(**_: object) -> None:
        raise RuntimeError("/Users/private/local-data/trace.csv")

    monkeypatch.setattr(api_main, "simulate_healthy", raise_private_path)
    response = client.get("/api/simulate", params={"scenario_name": "healthy"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Simulation failed. Check the API logs for details."
    assert "/Users/private" not in response.text
    _assert_simulation_semaphore_available()


def _assert_simulation_semaphore_available() -> None:
    acquired = api_main._SIMULATION_SEMAPHORE.acquire(blocking=False)
    assert acquired, "simulation semaphore was not released after request completion"
    api_main._SIMULATION_SEMAPHORE.release()


def test_simulation_semaphore_is_released_after_success(client: TestClient) -> None:
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
    )

    assert response.status_code == 200
    _assert_simulation_semaphore_available()


def test_simulation_semaphore_is_released_after_validation_failure(client: TestClient) -> None:
    response = client.get("/api/simulate?scenario_name=healthy&t_end=0.01")

    assert response.status_code == 422
    _assert_simulation_semaphore_available()


def test_simulation_without_overrides_preserves_default_behavior(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_simulate = api_main.simulate_healthy
    observed_overrides: list[dict[str, float] | None] = []

    def capture_overrides(**kwargs: object):
        observed_overrides.append(kwargs["param_overrides"])
        return original_simulate(**kwargs)

    monkeypatch.setattr(api_main, "simulate_healthy", capture_overrides)
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
    )

    assert response.status_code == 200
    assert response.json()["outputs"]["three_phase_currents_a"]
    assert observed_overrides == [None]


def test_simulation_accepts_frontend_encoded_json_override_query(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_simulate = api_main.simulate_healthy
    observed_overrides: list[dict[str, float] | None] = []

    def capture_overrides(**kwargs: object):
        observed_overrides.append(kwargs["param_overrides"])
        return original_simulate(**kwargs)

    monkeypatch.setattr(api_main, "simulate_healthy", capture_overrides)
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
        "&param_overrides=%7B%22r_s%22%3A1%7D"
    )

    assert response.status_code == 200
    assert observed_overrides == [{"r_s": 1.0}]


def test_simulation_accepts_multiple_overrides_and_empty_object(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_simulate = api_main.simulate_healthy
    observed_overrides: list[dict[str, float] | None] = []

    def capture_overrides(**kwargs: object):
        observed_overrides.append(kwargs["param_overrides"])
        return original_simulate(**kwargs)

    monkeypatch.setattr(api_main, "simulate_healthy", capture_overrides)
    base = "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
    multiple = quote(json.dumps({"r_s": 1.2, "r_r": 0.8}), safe="")
    response_multiple = client.get(f"{base}&param_overrides={multiple}")
    response_empty = client.get(f"{base}&param_overrides=%7B%7D")

    assert response_multiple.status_code == 200
    assert response_empty.status_code == 200
    assert observed_overrides == [{"r_s": 1.2, "r_r": 0.8}, {}]


@pytest.mark.parametrize(
    ("encoded_value", "expected_status"),
    [
        (quote('{"r_s":', safe=""), 400),
        (quote('[1]', safe=""), 422),
        (quote('1', safe=""), 422),
        (quote('{"unknown":1}', safe=""), 422),
        (quote('{"r_s":"1"}', safe=""), 422),
        (quote('{"r_s":true}', safe=""), 422),
        (quote('{"r_s":NaN}', safe=""), 400),
        (quote('{"r_s":Infinity}', safe=""), 400),
        (quote('{"r_s":1e999}', safe=""), 422),
        (quote('{"r_s":0}', safe=""), 422),
        (quote('{"r_s":-1}', safe=""), 422),
    ],
)
def test_simulation_rejects_invalid_override_query_without_leaking_details(
    client: TestClient, encoded_value: str, expected_status: int
) -> None:
    response = client.get(f"/api/simulate?scenario_name=healthy&param_overrides={encoded_value}")

    assert response.status_code == expected_status
    assert response.json()["detail"]
    assert "/Users/" not in response.text


def test_simulation_works_without_ml_dataset_or_approval(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", tmp_path / "missing.csv")
    monkeypatch.delenv("IMCM_ML_DATASET_APPROVED", raising=False)

    limits = client.get("/api/simulate/limits")
    simulation = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
    )

    assert limits.status_code == 200
    assert simulation.status_code == 200
    assert simulation.json()["outputs"]["three_phase_currents_a"]


def test_simulation_numeric_parameter_boundaries_are_inclusive() -> None:
    common = {
        "t_end": 0.1,
        "max_step": 1.0e-6,
        "output_dt": 1.0e-6,
        "rtol": 1.0e-12,
        "atol": 1.0e-12,
        "noise_seed": 0,
        "mqtt_chunk_size": 10,
    }
    api_main._validate_simulation_parameters(**common)

    maximums = {
        "t_end": 60.0,
        "max_step": 1.0,
        "output_dt": 1.0,
        "rtol": 1.0,
        "atol": 1.0,
        "noise_seed": None,
        "mqtt_chunk_size": 10_000,
    }
    api_main._validate_simulation_parameters(**maximums)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("t_end", "0.099999"),
        ("t_end", "60.000001"),
        ("max_step", "0.000000999"),
        ("max_step", "1.000001"),
        ("output_dt", "0.000000999"),
        ("output_dt", "1.000001"),
        ("rtol", "0.0000000000009"),
        ("rtol", "1.000001"),
        ("atol", "0.0000000000009"),
        ("atol", "1.000001"),
        ("noise_seed", "-1"),
        ("mqtt_chunk_size", "9"),
        ("mqtt_chunk_size", "10001"),
    ],
)
def test_simulation_rejects_values_outside_documented_limits(
    client: TestClient, parameter: str, value: str
) -> None:
    response = client.get(f"/api/simulate?scenario_name=healthy&{parameter}={value}")

    assert response.status_code == 422
    assert response.json()["detail"]
    assert "/Users/" not in response.text


@pytest.mark.parametrize("parameter", ["t_end", "max_step", "output_dt", "rtol", "atol"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_simulation_rejects_non_finite_solver_parameters(
    client: TestClient, parameter: str, value: str
) -> None:
    response = client.get(f"/api/simulate?scenario_name=healthy&{parameter}={value}")

    assert response.status_code == 422
    assert "/Users/" not in response.text


def test_simulation_rejects_work_budgets_just_over_the_limit(client: TestClient) -> None:
    output_budget = client.get(
        "/api/simulate?scenario_name=healthy&t_end=1&output_dt=0.00008333333333333333"
    )
    step_budget = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.100001&max_step=0.000001"
    )
    long_duration = client.get("/api/simulate?scenario_name=healthy&t_end=1000000")
    tiny_interval = client.get("/api/simulate?scenario_name=healthy&output_dt=0.000000000001")

    assert output_budget.status_code == 422
    assert step_budget.status_code == 422
    assert long_duration.status_code == 422
    assert tiny_interval.status_code == 422
    for response in (output_budget, step_budget, long_duration, tiny_interval):
        assert "/Users/" not in response.text


def test_simulation_query_budget_boundary_and_oversized_request(client: TestClient) -> None:
    prefix = "scenario_name="
    accepted = client.get(
        "/api/simulate?" + prefix + "x" * (api_main.MAX_SIMULATION_QUERY_BYTES - len(prefix)),
        headers={"Origin": "https://motor-simulation-data-comparison.vercel.app"},
    )
    oversized = client.get(
        "/api/simulate?" + prefix + "x" * (api_main.MAX_SIMULATION_QUERY_BYTES - len(prefix) + 1),
        headers={"Origin": "https://motor-simulation-data-comparison.vercel.app"},
    )

    assert accepted.status_code == 404
    assert accepted.headers["access-control-allow-origin"] == "https://motor-simulation-data-comparison.vercel.app"
    assert oversized.status_code == 414
    assert oversized.headers["access-control-allow-origin"] == "https://motor-simulation-data-comparison.vercel.app"
    assert "/Users/" not in oversized.text


def test_simulation_concurrency_limit_is_enforced_per_process(client: TestClient) -> None:
    acquired = api_main._SIMULATION_SEMAPHORE.acquire(blocking=False)
    assert acquired
    try:
        response = client.get("/api/simulate?scenario_name=healthy")
    finally:
        api_main._SIMULATION_SEMAPHORE.release()

    assert response.status_code == 429
    assert "already running" in response.json()["detail"]


def test_simulation_work_budget_inclusive_boundaries() -> None:
    api_main._validate_simulation_work_budget(
        t_end=1.0,
        max_step=1.0e-5,
        output_dt=1.0 / (api_main.MAX_SIMULATION_OUTPUT_SAMPLES - 1),
    )
    api_main._validate_simulation_work_budget(
        t_end=1.0,
        max_step=1.0 / api_main.MAX_ESTIMATED_SOLVER_STEPS,
        output_dt=0.1,
    )


def test_rhs_evaluation_budget_aborts_model_and_returns_safe_4xx(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "MAX_SIMULATION_RHS_EVALUATIONS", 1)
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
    )

    assert response.status_code == 422
    assert "RHS evaluation budget" in response.json()["detail"]
    assert "/Users/" not in response.text


@pytest.mark.parametrize("chunk_size", [10, 10_000])
def test_mqtt_chunk_boundaries_are_accepted(client: TestClient, chunk_size: int) -> None:
    response = client.get(
        "/api/simulate?scenario_name=healthy&t_end=0.1&output_dt=0.001&max_step=0.001"
        f"&mqtt_enabled=true&mqtt_chunk_size={chunk_size}"
    )

    assert response.status_code == 200


def test_ml_analyze_endpoint_returns_frontend_schema_from_dataset(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMCM_ML_DATASET_APPROVED", "true")
    dataset_path = tmp_path / "labeled_dataset.csv"
    dataset_path.write_text(
        "true_label,provenance,scenario_name,neg_seq_current_a\n"
        "healthy,simulated,test_scenario,0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", dataset_path)

    response = client.post("/api/ml/analyze")
    assert response.status_code == 200
    data = response.json()
    assert data["records"] == 1
    assert data["class_distribution"] == {"healthy": 1}
    assert data["scenario_distribution"] == {"test_scenario": 1}
    assert data["feature_stats"]["neg_seq_current_a"]["mean"] == pytest.approx(0.1)
    assert isinstance(data["limitations"], list)
    assert isinstance(data["warnings"], list)
    assert data["ready_for_ml"] is False


def test_ml_analyze_missing_dataset_returns_actionable_404(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", tmp_path / "missing.csv")
    response = client.post("/api/ml/analyze")
    assert response.status_code == 404
    assert "explicitly approved labeled_dataset.csv" in response.json()["detail"]
    assert "verified provenance" in response.json()["detail"]


def test_ml_analyze_rejects_dataset_without_expected_provenance_schema(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMCM_ML_DATASET_APPROVED", "true")
    dataset_path = tmp_path / "labeled_dataset.csv"
    dataset_path.write_text(
        "true_label,scenario_name\nhealthy,test_scenario\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", dataset_path)

    response = client.post("/api/ml/analyze")
    assert response.status_code == 422
    assert "provenance" in response.json()["detail"]


def test_ml_analyze_rejects_unknown_provenance_label(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMCM_ML_DATASET_APPROVED", "true")
    dataset_path = tmp_path / "labeled_dataset.csv"
    dataset_path.write_text(
        "true_label,scenario_name,provenance\nhealthy,test_scenario,unknown\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", dataset_path)

    response = client.post("/api/ml/analyze")
    assert response.status_code == 422
    assert "unsupported provenance labels" in response.json()["detail"]


def test_ml_analyze_requires_explicit_dataset_approval(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IMCM_ML_DATASET_APPROVED", raising=False)
    dataset_path = tmp_path / "labeled_dataset.csv"
    dataset_path.write_text(
        "true_label,scenario_name,provenance\nhealthy,test_scenario,simulated\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "LABELED_DATASET_PATH", dataset_path)

    response = client.post("/api/ml/analyze")
    assert response.status_code == 503
    assert "explicitly approved" in response.json()["detail"]


def test_ml_model_and_interpretation_endpoints_return_frontend_schema(
    client: TestClient,
) -> None:
    model_response = client.get("/api/ml/model")
    assert model_response.status_code == 200
    model = model_response.json()
    assert model["model_type"] == "RandomForestClassifier"
    assert isinstance(model["config"]["feature_columns"], list)
    assert "ready" in model

    interpretation_response = client.get("/api/ml/interpretations")
    assert interpretation_response.status_code == 200
    interpretation = interpretation_response.json()
    assert "feature_importance" in interpretation
    assert "shap" in interpretation
    assert interpretation["limitations"]["exploratory_only"] is True


def test_production_origin_allows_simulation_get_and_ml_post_preflight(
    client: TestClient,
) -> None:
    origin = "https://motor-simulation-data-comparison.vercel.app"
    for path, method in (("/api/simulate/limits", "GET"), ("/api/ml/analyze", "POST")):
        response = client.options(
            path,
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert method in response.headers["access-control-allow-methods"]
        assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_cors_configured_origins_are_preserved_and_production_origin_added(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://existing.example, https://other.example")
    origins = api_main.get_cors_allowed_origins()
    assert origins == [
        "https://existing.example",
        "https://other.example",
        "https://motor-simulation-data-comparison.vercel.app",
    ]


def test_cors_preflight_rejects_disallowed_origin(client: TestClient) -> None:
    response = client.options(
        "/api/simulate",
        headers={
            "Origin": "https://unapproved.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_ml_analyze_openapi_operation_ids_are_distinct() -> None:
    schema = api_main.app.openapi()

    get_id = schema["paths"]["/api/ml/analyze"]["get"]["operationId"]
    post_id = schema["paths"]["/api/ml/analyze"]["post"]["operationId"]

    assert get_id != post_id
