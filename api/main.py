import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from imcm.ml import run_exploratory_training, SKLEARN_AVAILABLE
from imcm.reporting.explanation import generate_explanation
from imcm.models.fifth_order_dq import simulate_healthy, illustrative_4kw_400v_50hz_4pole, CONVENTION_NAME
from imcm.models.operating_scenario import (
    first_milestone_scenario,
    fault_01_phase_a_resistance_imbalance_scenario,
    fault_02_increased_mechanical_load_scenario,
    fault_03_supply_voltage_unbalance_scenario,
    fault_04_bearing_outer_race_scenario,
    fault_05_rotor_asymmetry_scenario,
)
from imcm.reporting.provenance import ALLOWED_DATASET_LABELS


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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/simulate")
def simulate(
    scenario_name: str = "healthy",
    noise_seed: int | None = None,
    param_overrides: dict[str, float] | None = None,
    t_end: float = 1.0,
    max_step: float = 1e-4,
    output_dt: float = 1e-4,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    mqtt_enabled: bool = False,
    mqtt_chunk_size: int = 100,
) -> dict[str, Any]:
    """Run a simulation for the given scenario using simulate_healthy.

    If mqtt_enabled=True, the simulation results are streamed through the
    simulated MQTT replay pipeline (publisher + window buffer) and a
    streaming health report is returned alongside the results.
    """
    from imcm.models.fifth_order_dq import CONVENTION_NAME
    from imcm.iot.streaming import MQTTReplayPublisher, SignalWindowBuffer

    scenario_fn = {
        "healthy": first_milestone_scenario,
        "condition-02": fault_02_increased_mechanical_load_scenario,
        "fault-01": fault_01_phase_a_resistance_imbalance_scenario,
        "fault-03": fault_03_supply_voltage_unbalance_scenario,
        "fault-04": fault_04_bearing_outer_race_scenario,
        "fault-05": fault_05_rotor_asymmetry_scenario,
    }.get(scenario_name)

    if scenario_fn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario: {scenario_name}",
        )

    scenario = scenario_fn()

    try:
        result = simulate_healthy(
            scenario=scenario,
            t_end=t_end,
            max_step=max_step,
            output_dt=output_dt,
            rtol=rtol,
            atol=atol,
            param_overrides=param_overrides,
            noise_seed=noise_seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

    three_phase = result.v_abc
    qd0 = result.i_abc

    metadata = dict(result.metadata)
    metadata["provenance"] = "simulated"
    metadata["experimental_validation"] = False

    # Build SignalTrace from simulation result
    from imcm.data import SignalChannel, SignalTrace
    import numpy as np

    channels_dict = {
        "current_a": result.i_abc[0],
        "current_b": result.i_abc[1],
        "current_c": result.i_abc[2],
        "i_qs": qd0[0],
        "i_ds": qd0[1],
        "omega_m": result.omega_m,
        "tau_e": result.tau_e,
        "lambda_qs": result.lambda_qs,
        "lambda_ds": result.lambda_ds,
    }
    units_dict = {k: "A" for k in channels_dict}
    signal_types_dict = {"current_a": "current", "current_b": "current", "current_c": "current"}
    for k in ["i_qs", "i_ds", "omega_m", "tau_e", "lambda_qs", "lambda_ds"]:
        signal_types_dict[k] = "derived"

    fs = 1.0 / (result.t[1] - result.t[0]) if len(result.t) > 1 else 10000.0
    trace = SignalTrace.from_arrays(
        time_s=result.t,
        channels=channels_dict,
        units=units_dict,
        signal_types=signal_types_dict,
        sample_rate_hz=fs,
        sample_rate_source="declared",
        declared_sample_rate_hz=fs,
        metadata={"label": scenario.name},
    )

    # If MQTT streaming is enabled, run the simulated streaming pipeline
    mqtt_report = None
    if mqtt_enabled:
        publisher = MQTTReplayPublisher(trace, chunk_size=mqtt_chunk_size)
        buffer = SignalWindowBuffer(expected_fs_hz=fs)

        # Stream all channels
        for channel_name in ["current_a", "current_b", "current_c"]:
            for msg in publisher.stream_chunks(channel_name):
                buffer.ingest(msg)

        # Build streaming health report
        mqtt_report = {}
        for channel_name in ["current_a", "current_b", "current_c"]:
            report = buffer.health_by_channel.get(channel_name)
            if report:
                mqtt_report[channel_name] = {
                    "total_packets_received": report.total_packets_received,
                    "duplicate_count": report.duplicate_count,
                    "gap_count": report.gap_count,
                    "out_of_order_count": report.out_of_order_count,
                    "malformed_count": report.malformed_count,
                    "min_seq": report.min_seq,
                    "max_seq": report.max_seq,
                    "missing_sequences": report.missing_sequences[:10],  # limit for response size
                }

    base_response = {
        "scenario_name": scenario.name,
        "t_end_s": t_end,
        "outputs": {
            "three_phase_currents_a": [
                {"time_s": float(t), "current_a": float(c)}
                for t, c in zip(result.t, three_phase[0])
            ],
            "three_phase_currents_b": [
                {"time_s": float(t), "current_a": float(c)}
                for t, c in zip(result.t, three_phase[1])
            ],
            "three_phase_currents_c": [
                {"time_s": float(t), "current_a": float(c)}
                for t, c in zip(result.t, three_phase[2])
            ],
            "qd0_currents_a": [
                {"time_s": float(t), "current_a": float(c)}
                for t, c in zip(result.t, qd0[0])
            ],
            "qd0_currents_b": [
                {"time_s": float(t), "current_a": float(c)}
                for t, c in zip(result.t, qd0[1])
            ],
            "omega_m": [
                {"time_s": float(t), "current_a": float(w)}
                for t, w in zip(result.t, result.omega_m)
            ],
            "tau_e": [
                {"time_s": float(t), "current_a": float(tau)}
                for t, tau in zip(result.t, result.tau_e)
            ],
            "tau_l": [
                {"time_s": float(t), "current_a": float(tau_l)}
                for t, tau_l in zip(result.t, result.tau_l)
            ],
            "lambda_qs": [
                {"time_s": float(t), "current_a": float(lam)}
                for t, lam in zip(result.t, result.lambda_qs)
            ],
            "lambda_ds": [
                {"time_s": float(t), "current_a": float(lam)}
                for t, lam in zip(result.t, result.lambda_ds)
            ],
        },
        "metadata": metadata,
    }

    if mqtt_report:
        base_response["mqtt_streaming_report"] = mqtt_report

    return base_response


@app.get("/api/simulate/limits")
def simulate_limits() -> dict[str, Any]:
    """Return supported parameter bounds and simulation limits per the planner."""
    return {
        "t_end": {"min": 0.1, "max": 60.0, "default": 1.0, "unit": "s"},
        "max_step": {"min": 1.0e-6, "max": 1.0, "default": 1.0e-4, "unit": "s"},
        "output_dt": {"min": 1.0e-6, "max": 1.0, "default": 1.0e-4, "unit": "s"},
        "rtol": {"min": 1.0e-12, "max": 1.0, "default": 1.0e-6},
        "atol": {"min": 1.0e-12, "max": 1.0, "default": 1.0e-8},
        "y0": {"default": [0.0, 0.0, 0.0, 0.0, 0.0], "description": "5 initial state values (fluxes, speed)"},
        "param_overrides": {
            "valid_keys": ["r_s", "r_r", "l_ls", "l_lr", "l_m", "inertia", "viscous_friction"],
            "description": "Replace corresponding InductionMotorParameters field; all values must be positive",
        },
        "noise_seed": {"description": "Seeded Gaussian-noise injection after simulation; same seed + same config → identical results"},
        "scenario": {
            "description": "Scenario name mapping to operating conditions",
            "available": list(CONDITION_FILES.keys()),
        },
        "provenance": {
            "always_simulated": True,
            "parameter_provenance_literature_example": True,
            "experimental_validation": False,
        },
        "mqtt_chunk_size": {"min": 10, "max": 10000, "default": 100, "unit": "samples"},
        "mqtt_streaming": {
            "description": "When mqtt_enabled=true, simulation traces are chunked into JSON messages with versioned schema and reassembled via window buffer. Health report includes duplicate/gap/out-of-order detection."
        },
    }


@app.api_route("/api/ml/analyze", methods=["GET", "POST"])
def analyze_ml_dataset() -> dict[str, Any]:
    """Analyze the labeled dataset and report ML readiness.

    Returns dataset structure, limitations, and readiness assessment.
    This is an exploratory-only analysis; no generalization claims are made.
    """
    import pandas as pd
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parent.parent / "labeled_dataset.csv"

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="labeled_dataset.csv not found. Run dataset generation first.",
        )

    df = pd.read_csv(dataset_path)

    # Basic analysis
    n_records = len(df)
    class_counts = df["true_label"].value_counts().to_dict()
    scenario_counts = df["scenario_name"].value_counts().to_dict()
    feature_columns = [
        "neg_seq_current_a",
        "current_unbalance_pct",
        "dq_i_qs_rms_a",
        "dq_i_ds_rms_a",
        "spectral_peak_amp_a",
        "torque_ripple_rms_nm",
        "sideband_lower_amp_a",
    ]

    # Check scenario=class confounding
    scenario_class_mapping = {}
    for _, row in df.iterrows():
        scenario = row["scenario_name"]
        label = row["true_label"]
        if scenario not in scenario_class_mapping:
            scenario_class_mapping[scenario] = set()
        scenario_class_mapping[scenario].add(label)

    # Feature stats
    feature_stats = {}
    for feat in feature_columns:
        if feat in df.columns:
            vals = df[feat].fillna(0)
            feature_stats[feat] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }

    # Determine readiness
    scenario_with_multiple_classes = any(
        len(classes) > 1 for classes in scenario_class_mapping.values()
    )

    readiness = {
        "records": n_records,
        "classes": len(class_counts),
        "scenarios": len(scenario_counts),
        "class_distribution": class_counts,
        "scenario_distribution": scenario_counts,
        "scenario_class_mapping": {
            k: list(v) for k, v in scenario_class_mapping.items()
        },
        "feature_stats": feature_stats,
        "sklearn_available": SKLEARN_AVAILABLE,
        "limitations": [
            "Scenario=class confounding: each scenario contains only one class — "
            "no valid held-out test set possible",
            "Only one parameter set: results are motor-specific, not generalizable",
            "Cannot evaluate generalization to new scenarios",
            "All results are exploratory-only: trained on training data only",
        ],
        "warnings": [
            "Dataset generated from simulation only",
            "No experimental validation",
            "Limited to 6 scenarios with 1 parameter set each",
        ],
        "ready_for_ml": scenario_with_multiple_classes and SKLEARN_AVAILABLE,
    }

    return readiness


@app.get("/api/ml/model")
def get_ml_model_info() -> dict[str, Any]:
    """Get ML model information and readiness status.

    Returns current model configuration and status.
    """
    from imcm.ml.pipeline import MLPipeline
    from imcm.ml import SKLEARN_AVAILABLE, PipelineConfig

    config = PipelineConfig()

    info = {
        "sklearn_available": SKLEARN_AVAILABLE,
        "model_type": "RandomForestClassifier",
        "config": config.to_dict(),
        "feature_columns": config.feature_columns,
        "target_column": config.target_column,
        "group_column": config.group_column,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "scale_features": config.scale_features,
        "class_weight": config.class_weight,
        "cv_strategy": config.cv_strategy,
        "ready": SKLEARN_AVAILABLE,
    }

    if not SKLEARN_AVAILABLE:
        info["message"] = (
            "scikit-learn not available. Install with: "
            "pip install scikit-learn joblib shap"
        )
    else:
        info["message"] = "Model ready — install dependencies and run exploratory training"

    return info


@app.get("/api/ml/interpretations")
def get_ml_interpretations() -> dict[str, Any]:
    """Get ML interpretation information.

    Reports what interpretation features are available
    and what limitations apply.
    """
    return {
        "feature_importance": {
            "builtin": SKLEARN_AVAILABLE,
            "description": "Random Forest built-in feature importances",
            "available": SKLEARN_AVAILABLE,
        },
        "shap": {
            "available": False,
            "message": "SHAP not configured in this session. Install with: pip install shap",
        },
        "limitations": {
            "scenario_class_confounding": True,
            "single_parameter_set": True,
            "no_valid_held_out": True,
            "exploratory_only": True,
        },
        "message": (
            "Interpretation features available when sklearn installed. "
            "All results are exploratory-only with explicit limitations."
        ),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "read_only": True,
        "experimental_validation": False,
        "data_source": "repository_json_files",
    }


@app.get("/api/conditions")
def get_all_conditions() -> dict[str, Any]:
    """Get all available conditions."""
    conditions = []
    for condition_id in CONDITION_FILES.keys():
        filename = CONDITION_FILES[condition_id]
        file_path = RESULTS_DIR / filename

        data = {}
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        conditions.append({
            "condition_id": condition_id,
            "source_file": filename,
            "data": data,
            "explanation": generate_explanation(condition_id, data).to_dict() if data else {},
        })

    return {
        "count": len(conditions),
        "conditions": conditions,
    }


@app.get("/api/conditions/{condition_id}")
def get_condition(condition_id: str) -> dict[str, Any]:
    """Get a specific condition."""
    if condition_id not in CONDITION_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown condition ID: {condition_id}",
        )

    filename = CONDITION_FILES[condition_id]
    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found: {filename}",
        )

    import json
    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in {filename}: {error}",
        ) from error

    return {
        "condition_id": condition_id,
        "source_file": filename,
        "data": data,
        "explanation": generate_explanation(condition_id, data).to_dict(),
    }