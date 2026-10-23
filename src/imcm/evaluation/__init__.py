"""V2.3 Evaluation Design Infrastructure.

Provides group-aware evaluation protocols, baseline comparisons, and
metric computation for simulated induction motor dataset analysis.

Key design principles:
- Never claim meaningful classifier performance from insufficient data
- Always report provenance (simulated vs experimental vs literature)
- Group-aware validation (scenario-stratified splits, no leakage)
- Honest reporting of simulated-data limitations
- Metrics: macro-F1, per-class precision/recall, confusion matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imcm.models.fifth_order_dq import simulate_healthy
from imcm.reporting.provenance import ProvenanceRecord, ALLOWED_DATASET_LABELS


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocol."""

    # Split strategy: leave-one-scenario-out (6 folds for 6 scenarios)
    split_strategy: str = "leave_one_scenario_out"

    # Metrics to compute
    compute_macro_f1: bool = True
    compute_per_class: bool = True
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True

    # Baseline strategies
    baseline_strategies: list[str] = field(
        default_factory=lambda: ["majority_class", "random", "parameter_sweep"]
    )

    # Provenance requirements
    require_provenance: bool = True
    allowed_labels: frozenset = field(default_factory=lambda: ALLOWED_DATASET_LABELS)

    # Group-awareness
    group_by_scenario: bool = True
    prevent_split_leakage: bool = True


@dataclass
class EvaluationResult:
    """Result of running evaluation protocol."""

    config: EvaluationConfig
    scenario_splits: dict[str, dict[str, Any]]
    macro_f1: float | None = None
    per_class_metrics: dict[str, dict[str, float]] | None = None
    confusion_matrix: np.ndarray | None = None
    classification_report_str: str | None = None
    baselines: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def is_defensible(self) -> bool:
        """Check if dataset supports defensible evaluation (not classifier performance).

        This checks DATASET READINESS, not model performance. It does not require
        predicted_condition field - it checks total records, per-scenario balance,
        and provenance consistency.
        """
        # Exclude "aggregate" split from counts
        scenario_splits = {k: v for k, v in self.scenario_splits.items() if k != "aggregate"}

        # Total records check
        total_records = sum(
            s.get("n_test", 0) + s.get("n_train", 0) for s in scenario_splits.values()
        ) // max(1, len(scenario_splits))
        if total_records < 30:
            return False

        # Per-scenario balance check (max/min ratio <= 4.0)
        scenario_counts = [
            s.get("n_test", 0) + s.get("n_train", 0) for s in scenario_splits.values()
        ]
        if scenario_counts:
            ratio = max(scenario_counts) / min(scenario_counts)
            if ratio > 4.0:
                return False

        # Must have proper provenance (check from warnings if provenance issues exist)
        if self.config.require_provenance:
            for warning in self.warnings:
                if "provenance" in warning.lower() and "not in allowed labels" in warning.lower():
                    return False

        return True


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score without sklearn dependency.

    Uses manual computation to avoid requiring scikit-learn installation.
    """
    # Get unique labels
    all_labels = sorted(set(np.concatenate([y_true, y_pred]) if hasattr(y_true, '__iter__') else [y_true, y_pred]))
    n_labels = len(all_labels)
    if n_labels < 2:
        return 0.0

    # Map labels to indices
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    # Build confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=float)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1

    # Per-class F1 and macro average
    f1s = []
    for i in range(n_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, and F1 without sklearn dependency.

    Returns dict mapping each label to {precision, recall, f1}.
    """
    # Get all unique labels
    all_labels = sorted(set(np.concatenate([y_true, y_pred]) if hasattr(y_true, '__iter__') else [y_true, y_pred]))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    n_all = len(all_labels)

    # Build confusion matrix
    cm = np.zeros((n_all, n_all), dtype=float)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1

    # Per-class metrics for requested labels only
    result = {}
    for label in labels:
        if label not in label_to_idx:
            result[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        i = label_to_idx[label]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # column sum minus tp
        fn = cm[i, :].sum() - tp  # row sum minus tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = (
            2 * precision * recall / (precision + recall + 1e-12)
            if (precision + recall) > 0
            else 0.0
        )
        result[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def evaluate_splits(
    records: list[dict[str, Any]],
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Run evaluation on a list of provenance-labeled records.

    This is the core V2.3 evaluation protocol. It:
    1. Groups records by underlying simulation configuration (scenario)
    2. Performs leave-one-scenario-out splits
    3. Computes metrics while preventing split leakage
    4. Reports baselines and warns about data insufficiency

    CRITICAL: Does NOT claim meaningful classifier performance unless
    the dataset supports a defensible evaluation (typically >30 records
    with proper provenance across multiple scenarios).
    """
    if config is None:
        config = EvaluationConfig()

    warnings_list: list[str] = []

    n_total = len(records)
    warnings_list.append(f"Evaluating {n_total} records")

    # Check provenance consistency
    provenance_counts: dict[str, int] = {}
    for r in records:
        prov = r.get("provenance", "unknown")
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
    for prov, count in provenance_counts.items():
        if prov not in config.allowed_labels:
            warnings_list.append(
                f"Found {count} records with provenance '{prev}' not in allowed labels"
            )

    # Group by scenario to prevent split leakage
    scenarios: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        scenario = r.get("scenario_name", "unknown")
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(i)

    # Leave-one-scenario-out splits
    scenario_names = list(scenarios.keys())
    scenario_splits: dict[str, dict[str, Any]] = {}

    for test_scenario in scenario_names:
        test_indices = scenarios[test_scenario]
        train_indices = [
            idx for s, idxs in scenarios.items() if s != test_scenario for idx in idxs
        ]

        # Get labels for train/test - only check leakage if predicted_condition is available
        y_train = np.array([records[i].get("predicted_condition") for i in train_indices])
        y_test = np.array([records[i].get("predicted_condition") for i in test_indices])

        # Check for leakage: train and test labels should be disjoint by construction
        # Only check if we have actual predicted_condition values (not None)
        has_predictions = all(v is not None for v in y_train) and all(v is not None for v in y_test)
        if has_predictions:
            train_set = set(y_train)
            test_set = set(y_test)
            overlap = train_set & test_set
            if overlap and config.prevent_split_leakage:
                warnings_list.append(
                    f"Split leakage detected for {test_scenario}: {overlap} labels in both train and test"
                )
        else:
            # No predicted_condition available - skip leakage check
            train_set = set()
            test_set = set()
            overlap = set()

        scenario_splits[test_scenario] = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_labels": y_train,
            "test_labels": y_test,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "train_unique_labels": len(train_set),
            "test_unique_labels": len(test_set),
            "label_overlap": len(overlap),
        }

    # Compute metrics only if we have sufficient data
    if n_total >= 30 and config.compute_macro_f1:
        # Try to compute macro-F1 if we have proper labels
        all_y_true = np.array([r.get("true_condition", "unknown") for r in records])
        all_y_pred = np.array([r.get("predicted_condition", "unknown") for r in records])
        unique_labels = sorted(set(all_y_true) | set(all_y_pred))

        if len(unique_labels) >= 2:
            try:
                result_macro_f1 = compute_macro_f1(all_y_true, all_y_pred)
                scenario_splits["aggregate"] = {
                    "macro_f1": float(result_macro_f1),
                    "n_total": n_total,
                    "n_unique_labels": len(unique_labels),
                }
                warnings_list.append(
                    f"Macro-F1: {result_macro_f1:.4f} (n={n_total}, labels={unique_labels})"
                )
            except Exception as e:
                warnings_list.append(f"Could not compute macro-F1: {e}")
        else:
            warnings_list.append(
                "Insufficient label diversity for macro-F1 computation"
            )
    else:
        warnings_list.append(
            f"Insufficient records ({n_total}) for macro-F1; need >= 30 for defensible evaluation"
        )

    # Compute baselines
    baselines: dict[str, float] = {}
    if config.compute_per_class and n_total > 0:
        # Majority class baseline
        most_common = (
            max(set(all_y_true), key=lambda x: sum(1 for r in records if r.get("true_condition") == x))
            if "all_y_true" in dir()
            else "unknown"
        )
        majority_acc = sum(1 for r in records if r.get("true_condition") == most_common) / n_total
        baselines["majority_class_accuracy"] = float(majority_acc)
        warnings_list.append(
            f"Baseline - majority class accuracy: {majority_acc:.3f}"
        )

    # Random baseline
    import random
    random.seed(42)
    random_preds = [random.choice([r.get("true_condition", "unknown") for r in records]) for _ in records]
    # Compute random macro-F1 manually (no sklearn dependency)
    all_labels = sorted(set(all_y_true) | set(random_preds))
    if len(all_labels) >= 2:
        # Build confusion matrix
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        n_labels = len(all_labels)
        cm = np.zeros((n_labels, n_labels), dtype=float)
        for true, pred in zip(all_y_true, random_preds):
            if true in label_to_idx and pred in label_to_idx:
                cm[label_to_idx[true], label_to_idx[pred]] += 1
        # Per-class F1 and macro average
        f1s = []
        for i in range(n_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1s.append(f1)
        random_f1 = float(np.mean(f1s)) if f1s else 0.0
        baselines["random_macro_f1"] = random_f1
        warnings_list.append(
            f"Baseline - random macro-F1: {random_f1:.4f}"
        )

    # Parameter sweep baseline (using known parameter variations)
    baselines["parameter_sweep_note"] = "See V2.2 dataset for r_s variation baseline"

    result = EvaluationResult(
        config=config,
        scenario_splits=scenario_splits,
        macro_f1=scenario_splits.get("aggregate", {}).get("macro_f1"),
        per_class_metrics=None,  # Would need per-class computation
        confusion_matrix=None,
        classification_report_str=None,
        baselines=baselines,
        warnings=warnings_list,
    )

    # Final defensibility check
    if not result.is_defensible():
        result.warnings.append(
            "EVALUATION NOT DEFENSIBLE: Dataset insufficient for meaningful "
            "classifier performance claims. "
            "Current dataset has {} records across {} scenarios. "
            "Do not present classifier performance as meaningful until "
            "the dataset supports a defensible evaluation."
            "".format(n_total, len(scenario_names))
        )

    return result


def generate_evaluation_report(
    result: EvaluationResult,
    output_path: str | None = None,
) -> str:
    """Generate a human-readable evaluation report."""
    lines = [
        "=" * 60,
        "V2.3 EVALUATION REPORT",
        "=" * 60,
        "",
        f"Config: {result.config.split_strategy} split",
        f"Records evaluated: {sum(s.get('n_records', 0) for s in result.scenario_splits.values())}",
        f"Defensible: {result.is_defensible()}",
        "",
        "WARNINGS:",
    ]
    for w in result.warnings:
        lines.append(f"  - {w}")
    lines.extend(
        [
            "",
            "SCENARIO SPLITS:",
        ]
    )
    for split_name, split in result.scenario_splits.items():
        lines.append(
            f"  {split_name}: train={split.get('n_train', 0)}, "
            f"test={split.get('n_test', 0)}, "
            f"leakage={split.get('label_overlap', 0)}"
        )
    lines.extend(
        [
            "",
            "BASELINES:",
        ]
    )
    for name, value in (result.baselines or {}).items():
        lines.append(f"  {name}: {value}")
    lines.extend(
        [
            "",
            "METRICS:",
        ]
    )
    if result.macro_f1 is not None:
        lines.append(f"  macro-F1: {result.macro_f1:.4f}")
    else:
        lines.append("  macro-F1: N/A (insufficient data)")
    lines.extend(
        [
            "",
            "DEFENSIBILITY:",
        ]
    )
    if result.is_defensible():
        lines.append("  Evaluation is defensible - meaningful claims can be made.")
    else:
        lines.append(
            "  Evaluation is NOT defensible - do not present classifier "
            "performance as meaningful."
        )
    lines.extend(
        [
            "",
            "=" * 60,
        ]
    )

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text