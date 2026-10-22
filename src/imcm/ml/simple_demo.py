"""Simple demonstration of ML pipeline components without sklearn dependencies.

This script shows the structure and demonstrates the pipeline flow,
reporting actual metrics with explicit limitations.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path


def analyze_dataset(dataset_path: str = "labeled_dataset.csv") -> dict:
    """Analyze dataset structure and limitations."""
    df = pd.read_csv(dataset_path)

    # Basic analysis
    class_counts = df["true_label"].value_counts().to_dict()
    scenario_counts = df["scenario_name"].value_counts().to_dict()

    # Check scenario=class confounding
    scenario_class_mapping = {}
    for _, row in df.iterrows():
        scenario = row["scenario_name"]
        label = row["true_label"]
        if scenario not in scenario_class_mapping:
            scenario_class_mapping[scenario] = set()
        scenario_class_mapping[scenario].add(label)

    # Feature analysis
    features = [
        "neg_seq_current_a",
        "current_unbalance_pct",
        "dq_i_qs_rms_a",
        "dq_i_ds_rms_a",
        "spectral_peak_amp_a",
        "torque_ripple_rms_nm",
        "sideband_lower_amp_a",
    ]

    # Check feature distributions
    feature_stats = {}
    for feat in features:
        if feat in df.columns:
            vals = df[feat].fillna(0)
            feature_stats[feat] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }

    return {
        "n_records": len(df),
        "n_classes": len(class_counts),
        "n_scenarios": len(scenario_counts),
        "class_distribution": class_counts,
        "scenario_distribution": scenario_counts,
        "scenario_class_mapping": scenario_class_mapping,
        "feature_stats": feature_stats,
        "limitations": [
            f"Scenario=class confounding: each scenario contains only one class",
            f"Only one parameter set: {df['parameter_name'].unique()[0] if len(df['parameter_name'].unique()) == 1 else 'Multiple'}",
            f"No true held-out test set possible",
            f"Cannot evaluate generalization to new scenarios",
        ],
        "warnings": [
            "Dataset generated from simulation only",
            "No experimental validation",
            "Limited to 6 scenarios with 1 parameter set each",
        ]
    }


def report_analysis(analysis: dict):
    """Print human-readable analysis report."""
    print("=" * 60)
    print("V2.3 ML Pipeline - Dataset Analysis Report")
    print("=" * 60)
    print(f"Dataset: {analysis['n_records']} records")
    print(f"Classes: {analysis['n_classes']} ({list(analysis['class_distribution'].keys())})")
    print(f"Scenarios: {analysis['n_scenarios']}")
    print()

    print("Class Distribution:")
    for cls, count in analysis['class_distribution'].items():
        print(f"  {cls}: {count} records")
    print()

    print("Scenario Distribution:")
    for scenario, count in analysis['scenario_distribution'].items():
        classes = analysis['scenario_class_mapping'][scenario]
        print(f"  {scenario}: {count} records (classes: {classes})")
    print()

    print("Feature Statistics:")
    for feat, stats in analysis['feature_stats'].items():
        print(f"  {feat}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print()

    print("LIMITATIONS:")
    for lim in analysis['limitations']:
        print(f"  - {lim}")
    print()

    print("WARNINGS:")
    for warn in analysis['warnings']:
        print(f"  - {warn}")
    print()

    print("=" * 60)
    print("CONCLUSION: This dataset is suitable for exploratory analysis only.")
    print("No meaningful generalization claims can be made from this data.")
    print("=" * 60)


if __name__ == "__main__":
    analysis = analyze_dataset()
    report_analysis(analysis)