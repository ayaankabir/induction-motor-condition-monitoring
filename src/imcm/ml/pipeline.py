"""V2.3 ML Pipeline — Random Forest classifier with preprocessing and evaluation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Graceful sklearn import
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.inspection import permutation_importance

    SKLEARN_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"scikit-learn not available: {e}. ML functionality will be limited.")
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    accuracy_score = classification_report = confusion_matrix = f1_score = precision_recall_fscore_support = None
    LeaveOneGroupOut = None
    LabelEncoder = StandardScaler = compute_class_weight = None
    permutation_importance = None

from imcm.evaluation import EvaluationConfig, evaluate_splits
from imcm.reporting.provenance import ALLOWED_DATASET_LABELS


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""

    # Model parameters
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str | dict | None = "balanced"

    # Preprocessing
    scale_features: bool = True
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "neg_seq_current_a",
            "current_unbalance_pct",
            "dq_i_qs_rms_a",
            "dq_i_ds_rms_a",
            "spectral_peak_amp_a",
            "torque_ripple_rms_nm",
            "sideband_lower_amp_a",
        ]
    )
    target_column: str = "true_label"

    # Evaluation
    cv_strategy: str = "leave_one_scenario_out"
    group_column: str = "scenario_name"

    # Provenance
    require_provenance: bool = True
    allowed_labels: frozenset = field(default_factory=lambda: ALLOWED_DATASET_LABELS)

    # Output
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, frozenset):
                d[key] = list(value)
            else:
                d[key] = value
        return d


@dataclass
class PipelineResult:
    """Result of running the ML pipeline."""

    config: PipelineConfig
    model: RandomForestClassifier | None = None
    label_encoder: LabelEncoder | None = None
    scaler: StandardScaler | None = None

    # Training info
    train_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None
    train_scenarios: list[str] = field(default_factory=list)
    test_scenarios: list[str] = field(default_factory=list)

    # Predictions
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None

    # Metrics
    accuracy: float | None = None
    macro_f1: float | None = None
    per_class_metrics: dict[str, dict[str, float]] | None = None
    confusion_matrix: np.ndarray | None = None
    classification_report_str: str | None = None

    # Feature importance
    feature_importances: dict[str, float] | None = None

    # Baselines
    baseline_results: dict[str, Any] = field(default_factory=dict)

    # Warnings and limitations
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def is_valid_evaluation(self) -> bool:
        """Check if evaluation setup supports meaningful generalization claims."""
        return len(self.limitations) == 0


class MLPipeline:
    """End-to-end ML pipeline for fault classification."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.model: RandomForestClassifier | None = None
        if not SKLEARN_AVAILABLE:
            self._is_disabled = True
            self.label_encoder = None
            self.scaler = None
            return

        # Check if sklearn components are available
        required_attrs = ['RandomForestClassifier', 'LabelEncoder', 'StandardScaler',
                         'accuracy_score', 'classification_report', 'confusion_matrix',
                         'f1_score', 'precision_recall_fscore_support', 'LeaveOneGroupOut']
        missing = [attr for attr in required_attrs if eval(attr, globals(), locals()) is None]
        if missing:
            warnings.warn(f"ML disabled due to missing sklearn components: {missing}")
            self._is_disabled = True
            self.label_encoder = None
            self.scaler = None
        else:
            self._is_disabled = False
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler() if self.config.scale_features else None

    def load_dataset(self, csv_path: str | Path) -> pd.DataFrame:
        """Load and validate dataset."""
        df = pd.read_csv(csv_path)

        # Validate required columns
        missing = set(self.config.feature_columns + [self.config.target_column, self.config.group_column]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Check provenance
        if self.config.require_provenance:
            invalid_prov = set(df["provenance"].unique()) - self.config.allowed_labels
            if invalid_prov:
                raise ValueError(f"Invalid provenance labels: {invalid_prov}")

        # Replace NaN with 0 for features that may have NaN (e.g., sideband amps for non-fault_05)
        for col in self.config.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    def prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features, labels, and groups."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available. Cannot prepare data.")
        X = df[self.config.feature_columns].values
        y = self.label_encoder.fit_transform(df[self.config.target_column])
        groups = df[self.config.group_column].values
        return X, y, groups

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPipeline":
        """Train the Random Forest model."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available. Cannot fit model.")

        # Scale features
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        # Compute class weights if balanced
        if self.config.class_weight == "balanced":
            classes = np.unique(y)
            weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weight_dict = dict(zip(classes, weights))
        else:
            class_weight_dict = self.config.class_weight

        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight=class_weight_dict,
        )
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importances from trained model."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available. Cannot get feature importances.")
        importances = self.model.feature_importances_
        return dict(zip(self.config.feature_columns, importances))

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_groups: np.ndarray | None = None,
    ) -> PipelineResult:
        """Evaluate model on test data."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available. Cannot evaluate model.")

        # Scale test features if scaler is fitted
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        y_true_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        classes = self.label_encoder.classes_
        per_class = {}
        for i, cls in enumerate(classes):
            per_class[cls] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report_str = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

        # Feature importances
        feature_importances = self.get_feature_importances()

        result = PipelineResult(
            config=self.config,
            model=self.model,
            label_encoder=self.label_encoder,
            scaler=self.scaler,
            y_true=y_true_labels,
            y_pred=y_pred_labels,
            y_proba=y_proba,
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            classification_report_str=report_str,
            feature_importances=feature_importances,
        )

        return result

    def run_cross_validation(
        self, df: pd.DataFrame
    ) -> tuple[list[PipelineResult], dict[str, Any]]:
        """Run leave-one-scenario-out cross-validation."""
        X, y, groups = self.prepare_data(df)
        unique_groups = np.unique(groups)

        # Check evaluation validity: each class must appear in multiple groups
        class_to_groups = {}
        for cls in np.unique(y):
            class_groups = np.unique(groups[y == cls])
            class_to_groups[self.label_encoder.inverse_transform([cls])[0]] = class_groups

        limitations = []
        for cls, grps in class_to_groups.items():
            if len(grps) == 1:
                limitations.append(
                    f"Class '{cls}' appears in only ONE scenario ({grps[0]}). "
                    f"Leave-one-scenario-out CV cannot evaluate generalization for this class."
                )

        fold_results = []
        logo = LeaveOneGroupOut()

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_group = groups[test_idx[0]]

            # Skip folds where test class is not in training
            test_class = y[test_idx[0]]
            if test_class not in y[train_idx]:
                warnings.warn(
                    f"Fold {fold}: Test class {self.label_encoder.inverse_transform([test_class])[0]} "
                    f"not present in training data. Skipping for metric computation."
                )
                continue

            # Train on this fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Fresh model for each fold (or reuse with warm start)
            fold_model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state + fold,
                n_jobs=self.config.n_jobs,
                class_weight="balanced",
            )

            if self.scaler is not None:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                scaler = None

            fold_model.fit(X_train, y_train)

            # Evaluate
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)

            y_true_labels = self.label_encoder.inverse_transform(y_test)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)

            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            per_class = {}
            for i, cls in enumerate(self.label_encoder.classes_):
                per_class[cls] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }

            cm = confusion_matrix(y_test, y_pred)
            report_str = classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0
            )

            fold_results.append(
                PipelineResult(
                    config=self.config,
                    model=fold_model,
                    label_encoder=self.label_encoder,
                    scaler=scaler,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    train_scenarios=list(np.unique(groups[train_idx])),
                    test_scenarios=[test_group],
                    y_true=y_true_labels,
                    y_pred=y_pred_labels,
                    y_proba=y_proba,
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                    per_class_metrics=per_class,
                    confusion_matrix=cm,
                    classification_report_str=report_str,
                    feature_importances=dict(
                        zip(self.config.feature_columns, fold_model.feature_importances_)
                    ),
                    limitations=limitations.copy(),
                )
            )

        # Aggregate CV results
        valid_folds = [r for r in fold_results if r.macro_f1 is not None]
        cv_summary = {
            "n_folds": len(fold_results),
            "valid_folds": len(valid_folds),
            "mean_macro_f1": np.mean([r.macro_f1 for r in valid_folds]) if valid_folds else None,
            "std_macro_f1": np.std([r.macro_f1 for r in valid_folds]) if valid_folds else None,
            "mean_accuracy": np.mean([r.accuracy for r in valid_folds]) if valid_folds else None,
            "limitations": limitations,
            "class_to_groups": class_to_groups,
        }

        return fold_results, cv_summary


def run_exploratory_training(
    csv_path: str | Path,
    config: PipelineConfig | None = None,
) -> tuple[MLPipeline, PipelineResult, dict[str, Any]]:
    """Run exploratory training on full dataset (no CV)."""
    config = config or PipelineConfig()
    pipeline = MLPipeline(config)

    # Load and prepare data
    df = pipeline.load_dataset(csv_path)
    X, y, groups = pipeline.prepare_data(df)

    # Check if sklearn is available
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn not available. Running dummy pipeline for demonstration.")

        # Create a minimal result that shows the structure
        class DummyModel:
            def predict(self, X):
                return np.zeros(X.shape[0], dtype=int)
            def predict_proba(self, X):
                n_samples = X.shape[0]
                n_classes = 6
                return np.ones((n_samples, n_classes)) / n_classes

        dummy_model = DummyModel()

        result = PipelineResult(
            config=config,
            model=dummy_model,
            label_encoder=None,
            scaler=None,
            y_true=df[config.target_column].tolist(),
            y_pred=np.zeros(len(df), dtype=int),
            accuracy=0.0,
            macro_f1=0.0,
            feature_importances={},
        )

        result.warnings.append("SCikit-learn not installed. Model results are placeholders.")
        result.warnings.append("EXPLORATORY ONLY: No meaningful model was trained.")
        result.warnings.append(f"Dataset has {len(df)} records with scenario=class confounding.")

        class_dist = df[config.target_column].value_counts().to_dict()
        result.warnings.append(f"Class distribution: {class_dist}")

        return pipeline, result, {"class_distribution": class_dist, "n_scenarios": df["scenario_name"].nunique()}

    # Train on ALL data (exploratory only)
    pipeline.fit(X, y)

    # Evaluate on training data (overfit estimate only)
    result = pipeline.evaluate(X, y)
    result.train_indices = np.arange(len(df))
    result.test_indices = np.arange(len(df))
    result.train_scenarios = list(df[config.group_column].unique())
    result.test_scenarios = list(df[config.group_column].unique())

    # Evaluate baselines
    from imcm.ml.baselines import (
        MajorityClassBaseline,
        InterpretableBaseline,
        RandomBaseline,
        evaluate_baseline,
    )

    # Inverse-transform labels so evaluate_baseline gets string labels
    y_true_decoded = pipeline.label_encoder.inverse_transform(y)
    y_pred_decoded = pipeline.label_encoder.inverse_transform(
        pipeline.model.predict(pipeline.scaler.transform(X))
    )

    baseline_results = {}
    try:
        majority = MajorityClassBaseline(random_state=config.random_state)
        baseline_results["majority_class"] = evaluate_baseline(
            majority, X, y_true_decoded, X, y_true_decoded, name="majority_class"
        )

        interpretable = InterpretableBaseline(
            feature_names=config.feature_columns, random_state=config.random_state
        )
        baseline_results["interpretable"] = evaluate_baseline(
            interpretable, X, y_true_decoded, X, y_true_decoded, name="interpretable"
        )

        random_baseline = RandomBaseline(random_state=config.random_state)
        baseline_results["random"] = evaluate_baseline(
            random_baseline, X, y_true_decoded, X, y_true_decoded, name="random"
        )
    except Exception as e:
        warnings.warn(f"Baseline evaluation failed: {e}")

    result.baseline_results = baseline_results

    # Add limitation warnings
    result.warnings.append("EXPLORATORY ONLY: Model evaluated on training data (overfit estimate).")
    result.warnings.append(
        f"Dataset has {len(df)} records from {df['scenario_name'].nunique()} scenarios. "
        f"Scenario=class confounding prevents valid generalization assessment."
    )

    # Add class distribution info
    class_dist = df[config.target_column].value_counts().to_dict()
    result.warnings.append(f"Class distribution: {class_dist}")

    return pipeline, result, {"class_distribution": class_dist, "n_scenarios": df["scenario_name"].nunique()}