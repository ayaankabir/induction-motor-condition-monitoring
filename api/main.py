import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from imcm.reporting.explanation import generate_explanation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_cors_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

    return origins or DEFAULT_CORS_ALLOWED_ORIGINS


CONDITION_FILES = {
    "healthy": "healthy_startup_summary.json",
    "condition-02": "fault_02_increased_mechanical_load_summary.json",
    "fault-01": "fault_01_stator_resistance_imbalance_summary.json",
    "fault-03": "fault_03_supply_voltage_unbalance_summary.json",
    "fault-04": "fault_04_bearing_outer_race_summary.json",
    "fault-05": "fault_05_rotor_asymmetry_summary.json",
}


app = FastAPI(
    title="Induction Motor Condition Monitoring API",
    description=(
        "Read-only API for simulated induction motor condition-monitoring "
        "results. Data comes directly from the repository JSON files. "
        "The results are simulated and have no experimental validation."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_condition_json(condition_id: str) -> dict[str, Any]:
    filename = CONDITION_FILES.get(condition_id)

    if filename is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown condition ID: {condition_id}",
        )

    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found: {file_path}",
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in {filename}: {error}",
        ) from error

    explanation = generate_explanation(condition_id, data)

    return {
        "condition_id": condition_id,
        "source_file": filename,
        "data": data,
        "explanation": explanation.to_dict(),
    }


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "induction-motor-condition-monitoring-api",
        "data_source": "repository JSON files",
        "read_only": True,
        "experimental_validation": False,
    }


@app.get("/api/conditions")
def get_conditions() -> dict[str, Any]:
    conditions = []

    for condition_id in CONDITION_FILES:
        result = read_condition_json(condition_id)

        conditions.append(
            {
                "condition_id": result["condition_id"],
                "source_file": result["source_file"],
                "data": result["data"],
                "explanation": result["explanation"],
            }
        )

    return {
        "count": len(conditions),
        "conditions": conditions,
    }


@app.get("/api/conditions/{condition_id}")
def get_condition(condition_id: str) -> dict[str, Any]:
    return read_condition_json(condition_id)