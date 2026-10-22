"""V2.3 Baseline models for comparison."""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# Graceful sklearn import
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    from sklearn.utils.multiclass import unique_labels

    SKLEARN_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"scikit-learn not available: {e}. Baseline functionality will be limited.")
    SKLEARN_AVAILABLE = False
    BaseEstimator = ClassifierMixin = object
    check_X_y = check_array = check_is_fitted = unique_labels = None


@dataclass
class BaselineResult:
    """Result of baseline evaluation."""

    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    macro_f1: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    classification_report_str: str
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


if SKLEARN_AVAILABLE:
    class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
        """Majority class baseline - always predicts the most frequent class."""

        def __init__(self, random_state: int = 42):
            self.random_state = random_state
            self.majority_class_: Any = None
            self.classes_: np.ndarray | None = None

        def fit(self, X: np.ndarray, y: np.ndarray):
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
            # Find majority class
            unique, counts = np.unique(y, return_counts=True)
            self.majority_class_ = unique[np.argmax(counts)]
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)
            return np.full(X.shape[0], self.majority_class_)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)
            n_classes = len(self.classes_)
            proba = np.zeros((X.shape[0], n_classes))
            majority_idx = np.where(self.classes_ == self.majority_class_)[0][0]
            proba[:, majority_idx] = 1.0
            return proba

    class InterpretableBaseline(BaseEstimator, ClassifierMixin):
        """Simple interpretable baseline using single-feature thresholds."""

        def __init__(
            self,
            feature_names: list[str] | None = None,
            random_state: int = 42,
        ):
            self.feature_names = feature_names or []
            self.random_state = random_state
            self.classes_: np.ndarray | None = None
            self.thresholds_: dict[str, float] = {}
            self._feature_indices: dict[str, int] = {}

        def fit(self, X: np.ndarray, y: np.ndarray):
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)

            if self.feature_names:
                self._feature_indices = {name: i for i, name in enumerate(self.feature_names)}

            random.seed(self.random_state)
            np.random.seed(self.random_state)

            for cls in self.classes_:
                cls_mask = y == cls
                other_mask = ~cls_mask

                if not np.any(cls_mask) or not np.any(other_mask):
                    continue

                best_feature = None
                best_threshold = 0.0
                best_score = 0.0

                for feat_name, feat_idx in self._feature_indices.items():
                    if feat_idx >= X.shape[1]:
                        continue

                    cls_vals = X[cls_mask, feat_idx]
                    other_vals = X[other_mask, feat_idx]

                    if len(cls_vals) == 0 or len(other_vals) == 0:
                        continue

                    threshold = np.median(cls_vals)
                    pred = (X[:, feat_idx] > threshold).astype(int)
                    true_binary = cls_mask.astype(int)
                    score = np.mean(pred == true_binary)

                    if score > best_score:
                        best_score = score
                        best_feature = feat_name
                        best_threshold = threshold

                if best_feature is not None:
                    self.thresholds_[cls] = (best_feature, best_threshold, best_score)

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)

            n_samples = X.shape[0]
            predictions = []

            for i in range(n_samples):
                scores = {}
                for cls in self.classes_:
                    if cls in self.thresholds_:
                        feat_name, threshold, _ = self.thresholds_[cls]
                        feat_idx = self._feature_indices.get(feat_name)
                        if feat_idx is not None and feat_idx < X.shape[1]:
                            val = X[i, feat_idx]
                            scores[cls] = 1.0 if val > threshold else 0.0
                        else:
                            scores[cls] = 0.0
                    else:
                        scores[cls] = 0.0

                if scores:
                    pred = max(scores, key=scores.get)
                else:
                    pred = self.classes_[0]
                predictions.append(pred)

            return np.array(predictions)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)

            n_samples = X.shape[0]
            n_classes = len(self.classes_)
            proba = np.zeros((n_samples, n_classes))

            preds = self.predict(X)
            for i, pred in enumerate(preds):
                cls_idx = np.where(self.classes_ == pred)[0][0]
                proba[i, cls_idx] = 1.0

            return proba

    class RandomBaseline(BaseEstimator, ClassifierMixin):
        """Random baseline - predicts random class according to class distribution."""

        def __init__(self, random_state: int = 42):
            self.random_state = random_state
            self.classes_: np.ndarray | None = None
            self.class_prior_: np.ndarray | None = None

        def fit(self, X: np.ndarray, y: np.ndarray):
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
            unique, counts = np.unique(y, return_counts=True)
            self.class_prior_ = counts / counts.sum()
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)
            rng = np.random.default_rng(self.random_state)
            return rng.choice(self.classes_, size=X.shape[0], p=self.class_prior_)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            check_is_fitted(self)
            X = check_array(X)
            proba = np.tile(self.class_prior_, (X.shape[0], 1))
            return proba
else:
    class MajorityClassBaseline:
        def __init__(self, random_state: int = 42): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 1))

    class InterpretableBaseline:
        def __init__(self, feature_names=None, random_state=42): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 1))

    class RandomBaseline:
        def __init__(self, random_state=42): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
        def predict_proba(self, X): return np.zeros((len(X), 1))


def evaluate_baseline(
    baseline: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: Any = None,
    name: str = "baseline",
) -> BaselineResult:
    """Evaluate a baseline model."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not available. Cannot evaluate baseline.")

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    # Fit and predict
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    # Convert to labels if encoder provided
    if label_encoder is not None:
        y_true_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_true_labels = y_test
        y_pred_labels = y_pred

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    classes = baseline.classes_
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

    warnings_list = []
    if name == "majority_class":
        warnings_list.append("Majority class baseline - no generalization capability.")

    return BaselineResult(
        name=name,
        y_true=y_true_labels,
        y_pred=y_pred_labels,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_metrics=per_class,
        confusion_matrix=cm,
        classification_report_str=report_str,
        warnings=warnings_list,
    )
