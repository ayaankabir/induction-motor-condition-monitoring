import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from imcm.ml import run_exploratory_training, SKLEARN_AVAILABLE
from imcm.reporting.explanation import generate_explanation
from imcm.models.fifth_order_dq import (
    SimulationWorkLimitExceeded,
    simulate_healthy,
    illustrative_4kw_400v_50hz_4pole,
    CONVENTION_NAME,
)
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
logger = logging.getLogger(__name__)


DEFAULT_CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://motor-simulation-data-comparison.vercel.app",
]

# Vercel Preview deployment patterns for this project:
# 1. Git-based: https://motor-simulation-data-comparison-git-<branch>-<owner>.vercel.app
# 2. Hash-based: https://motor-simulation-data-comparison-<hash>.vercel.app
VERCEL_PREVIEW_PATTERN = re.compile(
    r"^https://motor-simulation-data-comparison(-git-[a-z0-9-]+-[a-z0-9-]+|-[a-z0-9]+)\.vercel\.app$"
)


LABELED_DATASET_PATH = PROJECT_ROOT / "labeled_dataset.csv"


def _is_vercel_preview_origin(origin: str) -> bool:
    """Check if origin matches this project's Vercel Preview domain pattern."""
    return bool(VERCEL_PREVIEW_PATTERN.match(origin))


def get_cors_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    if not origins:
        return DEFAULT_CORS_ALLOWED_ORIGINS.copy()

    # Retain configured origins and always include the production dashboard.
    return list(dict.fromkeys([*origins, DEFAULT_CORS_ALLOWED_ORIGINS[-1]]))


CONDITION_FILES = {
    "healthy": "healthy_startup_summary.json",
    "condition-02": "fault_02_increased_mechanical_load_summary.json",
    "fault-01": "fault_01_stator_resistance_imbalance_summary.json",
    "fault-03": "fault_03_supply_voltage_unbalance_summary.json",
    "fault-04": "fault_04_bearing_outer_race_summary.json",
    "fault-05": "fault_05_rotor_asymmetry_summary.json",
}

SUPPORTED_PARAM_OVERRIDE_KEYS = (
    "r_s",
    "r_r",
    "l_ls",
    "l_lr",
    "l_m",
    "inertia",
    "viscous_friction",
)

SIMULATION_NUMERIC_LIMITS = {
    "t_end": (0.1, 60.0),
    "max_step": (1.0e-6, 1.0),
    "output_dt": (1.0e-6, 1.0),
    "rtol": (1.0e-12, 1.0),
    "atol": (1.0e-12, 1.0),
}

# Public-request resource policies. These are engineering safeguards, not
# physical limits; expose them through /api/simulate/limits and the README.
MAX_SIMULATION_OUTPUT_SAMPLES = 12_000
MAX_ESTIMATED_SOLVER_STEPS = 100_000
MAX_SIMULATION_RHS_EVALUATIONS = 100_000
MAX_SIMULATION_QUERY_BYTES = 4_096
MAX_CONCURRENT_SIMULATIONS_PER_PROCESS = 1
_SIMULATION_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_SIMULATIONS_PER_PROCESS)


def _validate_simulation_parameters(
    *,
    t_end: float,
    max_step: float,
    output_dt: float,
    rtol: float,
    atol: float,
    noise_seed: int | None,
    mqtt_chunk_size: int,
) -> None:
    values = {
        "t_end": t_end,
        "max_step": max_step,
        "output_dt": output_dt,
        "rtol": rtol,
        "atol": atol,
    }
    for name, value in values.items():
        minimum, maximum = SIMULATION_NUMERIC_LIMITS[name]
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise HTTPException(
                status_code=422,
                detail=f"{name} must be finite and between {minimum:g} and {maximum:g}.",
            )

    if noise_seed is not None and noise_seed < 0:
        raise HTTPException(status_code=422, detail="noise_seed must be a non-negative integer.")

    if not 10 <= mqtt_chunk_size <= 10_000:
        raise HTTPException(
            status_code=422,
            detail="mqtt_chunk_size must be between 10 and 10000.",
        )


def _validate_simulation_work_budget(*, t_end: float, max_step: float, output_dt: float) -> None:
    output_ratio = Decimal(str(t_end)) / Decimal(str(output_dt))
    output_samples = int(output_ratio.to_integral_value(rounding=ROUND_FLOOR)) + 1
    if output_samples > MAX_SIMULATION_OUTPUT_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=(
                "Requested output exceeds the maximum of "
                f"{MAX_SIMULATION_OUTPUT_SAMPLES} samples; increase output_dt or reduce t_end."
            ),
        )

    solver_ratio = Decimal(str(t_end)) / Decimal(str(max_step))
    estimated_solver_steps = int(solver_ratio.to_integral_value(rounding=ROUND_CEILING))
    if estimated_solver_steps > MAX_ESTIMATED_SOLVER_STEPS:
        raise HTTPException(
            status_code=422,
            detail=(
                "Requested simulation exceeds the estimated solver-step budget of "
                f"{MAX_ESTIMATED_SOLVER_STEPS}; increase max_step or reduce t_end."
            ),
        )


def _validate_simulation_request(
    *,
    t_end: float,
    max_step: float,
    output_dt: float,
    rtol: float,
    atol: float,
    noise_seed: int | None,
    mqtt_chunk_size: int,
) -> None:
    _validate_simulation_parameters(
        t_end=t_end,
        max_step=max_step,
        output_dt=output_dt,
        rtol=rtol,
        atol=atol,
        noise_seed=noise_seed,
        mqtt_chunk_size=mqtt_chunk_size,
    )
    _validate_simulation_work_budget(
        t_end=t_end,
        max_step=max_step,
        output_dt=output_dt,
    )


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value} is not allowed")


def _parse_param_overrides(raw_value: str | None) -> dict[str, float] | None:
    """Parse and validate the frontend's URL-encoded JSON override object."""
    if raw_value is None:
        return None

    try:
        decoded = json.loads(raw_value, parse_constant=_reject_nonstandard_json_constant)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="param_overrides must be a valid JSON object containing finite numbers.",
        ) from None

    if not isinstance(decoded, dict):
        raise HTTPException(
            status_code=422,
            detail="param_overrides must decode to a JSON object.",
        )

    unsupported_keys = sorted(set(decoded) - set(SUPPORTED_PARAM_OVERRIDE_KEYS))
    if unsupported_keys:
        raise HTTPException(
            status_code=422,
            detail="Unsupported parameter override name(s): " + ", ".join(unsupported_keys),
        )

    validated: dict[str, float] = {}
    for key, value in decoded.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=422,
                detail=f"Parameter override '{key}' must be a finite number greater than zero.",
            )
        try:
            numeric_value = float(value)
        except OverflowError:
            numeric_value = math.inf
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise HTTPException(
                status_code=422,
                detail=f"Parameter override '{key}' must be a finite number greater than zero.",
            )
        validated[key] = numeric_value

    return validated


app = FastAPI(
    title="Induction Motor Condition Monitoring API",
    description=(
        "Read-only API for simulated induction motor condition-monitoring "
        "results. Data comes directly from the repository JSON files. "
        "The results are simulated and have no experimental validation."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def guard_simulation_requests(request: Request, call_next):
    if request.url.path == "/api/simulate" and request.method == "GET":
        if len(request.scope.get("query_string", b"")) > MAX_SIMULATION_QUERY_BYTES:
            return JSONResponse(
                status_code=414,
                content={"detail": f"Simulation query exceeds {MAX_SIMULATION_QUERY_BYTES} bytes."},
            )
        if not _SIMULATION_SEMAPHORE.acquire(blocking=False):
            return JSONResponse(
                status_code=429,
                content={"detail": "A simulation is already running in this API process; retry shortly."},
            )
        try:
            return await call_next(request)
        finally:
            _SIMULATION_SEMAPHORE.release()
    return await call_next(request)


class VercelPreviewCORSMiddleware:
    """Custom CORS middleware that supports Vercel Preview domain pattern matching."""

    def __init__(self, app):
        self.app = app
        self.allowed_origins = get_cors_allowed_origins()
        self.allow_credentials = False
        self.allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = ["*"]

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed (explicit list or Vercel Preview pattern)."""
        if origin in self.allowed_origins:
            return True
        return _is_vercel_preview_origin(origin)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract origin from headers
        origin = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"origin":
                origin = header_value.decode("latin-1")
                break

        is_preflight = scope["method"] == "OPTIONS"

        if origin and self._is_allowed_origin(origin):
            # Origin is allowed - add CORS headers to response
            async def send_with_cors(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"access-control-allow-origin", origin.encode("latin-1")))
                    headers.append((b"access-control-allow-credentials", b"false"))
                    if is_preflight:
                        headers.append((b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"))
                        headers.append((b"access-control-allow-headers", b"*"))
                    message["headers"] = headers
                await send(message)

            if is_preflight:
                # Handle preflight request
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"access-control-allow-origin", origin.encode("latin-1")),
                        (b"access-control-allow-credentials", b"false"),
                        (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                        (b"access-control-allow-headers", b"*"),
                    ],
                })
                await send({"type": "http.response.body", "body": b""})
                return

            await self.app(scope, receive, send_with_cors)
            return

        if is_preflight and origin:
            # Origin not allowed - reject preflight
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"detail": "CORS preflight rejected: origin not allowed"}',
            })
            return

        # For non-preflight requests with disallowed origins, pass through without CORS headers
        await self.app(scope, receive, send)


app.add_middleware(VercelPreviewCORSMiddleware)


@app.get("/api/simulate")
def simulate(
    scenario_name: str = "healthy",
    noise_seed: int | None = None,
    param_overrides: str | None = Query(
        default=None,
        description="URL-encoded JSON object mapping supported parameter names to positive finite numbers",
    ),
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

    parsed_param_overrides = _parse_param_overrides(param_overrides)
    _validate_simulation_request(
        t_end=t_end,
        max_step=max_step,
        output_dt=output_dt,
        rtol=rtol,
        atol=atol,
        noise_seed=noise_seed,
        mqtt_chunk_size=mqtt_chunk_size,
    )

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
            param_overrides=parsed_param_overrides,
            noise_seed=noise_seed,
            max_rhs_evaluations=MAX_SIMULATION_RHS_EVALUATIONS,
        )
    except SimulationWorkLimitExceeded:
        raise HTTPException(
            status_code=422,
            detail=(
                "Simulation exceeded its RHS evaluation budget of "
                f"{MAX_SIMULATION_RHS_EVALUATIONS}; use a less demanding configuration."
            ),
        ) from None
    except Exception:
        logger.exception("Simulation failed")
        raise HTTPException(
            status_code=500,
            detail="Simulation failed. Check the API logs for details.",
        )

    three_phase = result.i_abc
    qd0 = [result.i_qs, result.i_ds]

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
        "i_qs": result.i_qs,
        "i_ds": result.i_ds,
        "omega_m": result.omega_m,
        "tau_e": result.tau_e,
        "lambda_qs": result.lambda_qs,
        "lambda_ds": result.lambda_ds,
    }
    units_dict = {
        "current_a": "A",
        "current_b": "A",
        "current_c": "A",
        "i_qs": "A",
        "i_ds": "A",
        "omega_m": "rad/s",
        "tau_e": "N*m",
        "lambda_qs": "dimensionless",
        "lambda_ds": "dimensionless",
    }
    signal_types_dict = {
        "current_a": "current",
        "current_b": "current",
        "current_c": "current",
        "i_qs": "current",
        "i_ds": "current",
        "omega_m": "speed",
        "tau_e": "torque",
        "lambda_qs": "other",
        "lambda_ds": "other",
    }

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
    """Return supported numerical bounds and operational request budgets."""
    return {
        "t_end": {"min": 0.1, "max": 60.0, "default": 1.0, "unit": "s"},
        "max_step": {"min": 1.0e-6, "max": 1.0, "default": 1.0e-4, "unit": "s"},
        "output_dt": {"min": 1.0e-6, "max": 1.0, "default": 1.0e-4, "unit": "s"},
        "rtol": {"min": 1.0e-12, "max": 1.0, "default": 1.0e-6},
        "atol": {"min": 1.0e-12, "max": 1.0, "default": 1.0e-8},
        "y0": {"default": [0.0, 0.0, 0.0, 0.0, 0.0], "description": "5 initial state values (fluxes, speed)"},
        "param_overrides": {
            "valid_keys": list(SUPPORTED_PARAM_OVERRIDE_KEYS),
            "description": "URL-encoded JSON object; replace a parameter field with a finite value greater than zero",
        },
        "noise_seed": {
            "min": 0,
            "description": "Non-negative integer seed; same seed + same configuration gives identical noise",
        },
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
        "resource_limits": {
            "max_output_samples": MAX_SIMULATION_OUTPUT_SAMPLES,
            "max_estimated_solver_steps": MAX_ESTIMATED_SOLVER_STEPS,
            "max_rhs_evaluations": MAX_SIMULATION_RHS_EVALUATIONS,
            "max_query_bytes": MAX_SIMULATION_QUERY_BYTES,
            "max_concurrent_simulations_per_process": MAX_CONCURRENT_SIMULATIONS_PER_PROCESS,
            "note": "Engineering request budgets; parameter ranges also apply individually",
        },
        "mqtt_streaming": {
            "description": "When mqtt_enabled=true, simulation traces are chunked into JSON messages with versioned schema and reassembled via window buffer. Health report includes duplicate/gap/out-of-order detection."
        },
    }


@app.get("/api/ml/analyze", operation_id="analyze_ml_dataset_get")
@app.post("/api/ml/analyze", operation_id="analyze_ml_dataset_post")
def analyze_ml_dataset() -> dict[str, Any]:
    """Analyze the labeled dataset and report ML readiness.

    Returns dataset structure, limitations, and readiness assessment.
    This is an exploratory-only analysis; no generalization claims are made.
    """
    import pandas as pd
    dataset_path = LABELED_DATASET_PATH

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "ML analysis is unavailable: an explicitly approved labeled_dataset.csv "
                "with verified provenance and the expected schema is not installed."
            ),
        )

    if os.getenv("IMCM_ML_DATASET_APPROVED", "").strip().lower() != "true":
        raise HTTPException(
            status_code=503,
            detail=(
                "ML analysis is disabled until a dataset with verified provenance "
                "and the expected schema is reviewed and explicitly approved."
            ),
        )

    df = pd.read_csv(dataset_path)
    required_columns = {"true_label", "scenario_name", "provenance"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=(
                "ML analysis dataset has an invalid schema; missing required columns: "
                + ", ".join(missing_columns)
            ),
        )
    provenance_values = set(df["provenance"].dropna().astype(str).str.strip())
    invalid_provenance = sorted(provenance_values - ALLOWED_DATASET_LABELS)
    if df["provenance"].isna().any() or invalid_provenance:
        raise HTTPException(
            status_code=422,
            detail=(
                "ML analysis dataset contains missing or unsupported provenance labels; "
                "allowed labels are simulated, literature-example, and experimental."
            ),
        )
    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="ML analysis dataset must contain at least one labeled record.",
        )

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
