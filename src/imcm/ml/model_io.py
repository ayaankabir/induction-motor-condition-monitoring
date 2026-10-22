"""V2.3 Model I/O — save and load trained pipelines."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from imcm.ml.pipeline import MLPipeline, PipelineConfig, PipelineResult


@dataclass
class SavedPipeline:
    """Container for saved pipeline components."""

    config: dict[str, Any]
    model: Any  # RandomForestClassifier
    label_encoder: Any  # LabelEncoder
    scaler: Any | None  # StandardScaler or None
    feature_columns: list[str]
    target_column: str
    group_column: str
    classes_: list[str]
    version: str = "2.3"
    saved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON metadata."""
        return {
            "config": self.config,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "group_column": self.group_column,
            "classes": self.classes_,
            "version": self.version,
            "saved_at": self.saved_at,
        }


def save_model(
    pipeline: MLPipeline,
    path: str | Path,
    result: PipelineResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save trained pipeline to disk using joblib.

    Args:
        pipeline: Fitted MLPipeline instance
        path: Output path (directory or .joblib file)
        result: Optional PipelineResult to include
        metadata: Optional additional metadata

    Returns:
        Path to saved model file
    """
    from datetime import datetime

    path = Path(path)

    if path.suffix == ".joblib":
        model_dir = path.parent
        model_file = path
    else:
        model_dir = path
        model_file = path / "pipeline.joblib"

    model_dir.mkdir(parents=True, exist_ok=True)

    if not pipeline._is_fitted:
        raise RuntimeError("Pipeline not fitted. Cannot save unfitted model.")

    # Prepare components for saving
    saved = SavedPipeline(
        config=pipeline.config.to_dict(),
        model=pipeline.model,
        label_encoder=pipeline.label_encoder,
        scaler=pipeline.scaler,
        feature_columns=pipeline.config.feature_columns,
        target_column=pipeline.config.target_column,
        group_column=pipeline.config.group_column,
        classes_=list(pipeline.label_encoder.classes_),
        saved_at=datetime.now().isoformat(),
    )

    # Save model components with joblib
    joblib.dump(saved, model_file)

    # Save metadata as JSON
    meta_file = model_file.with_suffix(".json")
    meta_dict = saved.to_dict()
    if result is not None:
        meta_dict["evaluation"] = {
            "accuracy": result.accuracy,
            "macro_f1": result.macro_f1,
            "per_class_metrics": result.per_class_metrics,
            "feature_importances": result.feature_importances,
            "warnings": result.warnings,
            "limitations": result.limitations,
        }
    if metadata is not None:
        meta_dict["metadata"] = metadata

    with open(meta_file, "w") as f:
        json.dump(meta_dict, f, indent=2, default=str)

    return model_file


def load_model(path: str | Path) -> MLPipeline:
    """Load trained pipeline from disk.

    Args:
        path: Path to .joblib file or directory containing pipeline.joblib

    Returns:
        Fitted MLPipeline instance
    """
    path = Path(path)

    if path.is_dir():
        model_file = path / "pipeline.joblib"
    elif path.suffix == ".joblib":
        model_file = path
    else:
        model_file = path / "pipeline.joblib"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    saved: SavedPipeline = joblib.load(model_file)

    # Reconstruct config
    config = PipelineConfig(**saved.config)

    # Create pipeline and restore components
    pipeline = MLPipeline(config)
    pipeline.model = saved.model
    pipeline.label_encoder = saved.label_encoder
    pipeline.scaler = saved.scaler
    pipeline._is_fitted = True

    # Verify classes match
    loaded_classes = list(pipeline.label_encoder.classes_)
    if loaded_classes != saved.classes_:
        warnings.warn(
            f"Class mismatch: loaded {loaded_classes}, expected {saved.classes_}"
        )

    return pipeline


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    """Load model metadata from JSON file.

    Args:
        path: Path to .joblib file or directory containing pipeline.json

    Returns:
        Metadata dictionary
    """
    path = Path(path)

    if path.is_dir():
        meta_file = path / "pipeline.json"
    elif path.suffix == ".joblib":
        meta_file = path.with_suffix(".json")
    else:
        meta_file = path / "pipeline.json"

    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    with open(meta_file) as f:
        return json.load(f)


def verify_model(path: str | Path) -> dict[str, Any]:
    """Verify a saved model can be loaded and has expected structure.

    Returns:
        Verification report dictionary
    """
    report = {
        "loadable": False,
        "fitted": False,
        "has_model": False,
        "has_scaler": False,
        "has_label_encoder": False,
        "feature_columns": [],
        "classes": [],
        "errors": [],
    }

    try:
        pipeline = load_model(path)
        report["loadable"] = True

        if pipeline._is_fitted:
            report["fitted"] = True
        if pipeline.model is not None:
            report["has_model"] = True
        if pipeline.scaler is not None:
            report["has_scaler"] = True
        if pipeline.label_encoder is not None:
            report["has_label_encoder"] = True

        report["feature_columns"] = pipeline.config.feature_columns
        report["classes"] = list(pipeline.label_encoder.classes_)

    except Exception as e:
        report["errors"].append(str(e))

    return report