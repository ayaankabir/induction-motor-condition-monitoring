"""V2.3 Model interpretation — feature importance and SHAP explanations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# SHAP availability check
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


@dataclass
class FeatureImportanceReport:
    """Feature importance report for tree-based models."""

    importances: dict[str, float]
    std_importances: dict[str, float] | None = None
    method: str = "builtin"  # "builtin" or "permutation"
    n_samples: int = 0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "importances": self.importances,
            "std_importances": self.std_importances,
            "method": self.method,
            "n_samples": self.n_samples,
            "warnings": self.warnings,
        }

    def summary(self, top_k: int = 10) -> str:
        """Generate human-readable summary."""
        sorted_feats = sorted(self.importances.items(), key=lambda x: x[1], reverse=True)
        lines = [f"Feature Importance ({self.method}):", "-" * 40]
        for feat, imp in sorted_feats[:top_k]:
            std_str = f" ± {self.std_importances.get(feat, 0):.4f}" if self.std_importances else ""
            lines.append(f"  {feat}: {imp:.4f}{std_str}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class SHAPExplanation:
    """SHAP explanation for model predictions."""

    values: np.ndarray  # shape (n_samples, n_features) or (n_samples, n_features, n_classes)
    feature_names: list[str]
    class_names: list[str] | None = None
    base_values: np.ndarray | None = None
    data: np.ndarray | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def mean_abs_shap(self) -> dict[str, float]:
        """Mean absolute SHAP values per feature (global importance)."""
        if self.values.ndim == 3:
            # Multi-class: average over classes
            mean_shap = np.mean(np.abs(self.values), axis=(0, 2))
        else:
            # Binary or single-output
            mean_shap = np.mean(np.abs(self.values), axis=0)
        return dict(zip(self.feature_names, mean_shap))

    def summary(self, top_k: int = 10) -> str:
        """Generate human-readable summary."""
        mean_shap = self.mean_abs_shap()
        sorted_feats = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)
        lines = ["SHAP Feature Importance (mean |SHAP|):", "-" * 40]
        for feat, imp in sorted_feats[:top_k]:
            lines.append(f"  {feat}: {imp:.4f}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append("\nNote: SHAP values show feature association with model output, not causal effects.")
        return "\n".join(lines)


class FeatureImportanceReporter:
    """Report feature importances for tree-based models."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def report_builtin(self, model: Any, n_repeats: int = 10) -> FeatureImportanceReport:
        """Report built-in feature importances from tree ensemble."""
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = model.feature_importances_
        if len(importances) != len(self.feature_names):
            warnings.warn(f"Feature count mismatch: model has {len(importances)}, expected {len(self.feature_names)}")

        # For RandomForest, we can compute std across trees
        std_importances = None
        if hasattr(model, "estimators_") and model.estimators_:
            tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
            if tree_importances.shape[1] == len(self.feature_names):
                std_importances = dict(zip(self.feature_names, np.std(tree_importances, axis=0)))

        return FeatureImportanceReport(
            importances=dict(zip(self.feature_names, importances)),
            std_importances=std_importances,
            method="builtin",
            n_samples=len(model.estimators_) if hasattr(model, "estimators_") else 0,
        )

    def report_permutation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> FeatureImportanceReport:
        """Report permutation importance (requires sklearn >= 1.0)."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            warnings.warn("permutation_importance not available, falling back to builtin")
            return self.report_builtin(model)

        rng = np.random.default_rng(random_state)
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=rng, n_jobs=-1
        )

        return FeatureImportanceReport(
            importances=dict(zip(self.feature_names, result.importances_mean)),
            std_importances=dict(zip(self.feature_names, result.importances_std)),
            method="permutation",
            n_samples=n_repeats,
        )


class SHAPExplainer:
    """SHAP explanations for tree-based models."""

    def __init__(self, feature_names: list[str], class_names: list[str] | None = None):
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP not installed. Install with: pip install shap")
        self.feature_names = feature_names
        self.class_names = class_names
        self._explainer = None

    def explain(self, model: Any, X: np.ndarray, sample_limit: int | None = None) -> SHAPExplanation:
        """Generate SHAP explanations for model predictions."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP not available")

        # Limit sample size for performance
        if sample_limit is not None and X.shape[0] > sample_limit:
            idx = np.random.default_rng(42).choice(X.shape[0], sample_limit, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Create tree explainer
        self._explainer = shap.TreeExplainer(model)
        shap_values = self._explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class: list of (n_samples, n_features) arrays
            shap_values = np.array(shap_values)  # (n_classes, n_samples, n_features)
            shap_values = np.transpose(shap_values, (1, 2, 0))  # (n_samples, n_features, n_classes)
        elif shap_values.ndim == 3 and shap_values.shape[2] == 1:
            # Binary classification: (n_samples, n_features, 1)
            shap_values = shap_values.squeeze(-1)

        # Base values
        base_values = self._explainer.expected_value
        if isinstance(base_values, np.ndarray) and base_values.ndim == 1:
            base_values = base_values[0] if len(base_values) == 1 else base_values

        warnings_list = []
        if not SHAP_AVAILABLE:
            warnings_list.append("SHAP not available - explanations are placeholder")

        return SHAPExplanation(
            values=shap_values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            base_values=base_values,
            data=X_sample,
            warnings=warnings_list,
        )

    def explain_instance(
        self, model: Any, X: np.ndarray, instance_idx: int = 0
    ) -> dict[str, Any]:
        """Explain a single instance prediction."""
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP not available")

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(model)

        instance = X[instance_idx : instance_idx + 1]
        shap_values = self._explainer.shap_values(instance)

        # Format for single instance
        if isinstance(shap_values, list):
            # Multi-class
            result = {}
            for cls_idx, cls_shap in enumerate(shap_values):
                cls_name = self.class_names[cls_idx] if self.class_names else f"class_{cls_idx}"
                result[cls_name] = dict(zip(self.feature_names, cls_shap[0]))
            return result
        else:
            # Single output
            return dict(zip(self.feature_names, shap_values[0]))


def create_feature_importance_report(
    model: Any,
    feature_names: list[str],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    method: str = "builtin",
) -> FeatureImportanceReport:
    """Factory function to create feature importance report."""
    reporter = FeatureImportanceReporter(feature_names)

    if method == "builtin":
        return reporter.report_builtin(model)
    elif method == "permutation" and X is not None and y is not None:
        return reporter.report_permutation(model, X, y)
    else:
        warnings.warn(f"Method '{method}' not available, using builtin")
        return reporter.report_builtin(model)


def create_shap_explanation(
    model: Any,
    feature_names: list[str],
    X: np.ndarray,
    class_names: list[str] | None = None,
    sample_limit: int | None = 100,
) -> SHAPExplanation | None:
    """Factory function to create SHAP explanation. Returns None if SHAP unavailable."""
    if not SHAP_AVAILABLE:
        warnings.warn("SHAP not installed. Skipping SHAP explanations.")
        return None

    explainer = SHAPExplainer(feature_names, class_names)
    return explainer.explain(model, X, sample_limit=sample_limit)