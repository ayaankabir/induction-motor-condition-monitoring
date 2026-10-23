"""V2.3 Machine Learning Pipeline for Induction Motor Condition Monitoring.

Provides a reproducible scikit-learn pipeline for fault classification
using electrical MCSA features. Always reports limitations honestly.

Key principles:
- Never claim meaningful classifier performance from insufficient data
- Always report provenance (simulated vs experimental vs literature)
- Group-aware validation (scenario-stratified splits, no leakage)
- Honest reporting of simulated-data limitations
- Clear separation: dataset readiness vs model performance
"""

from __future__ import annotations

# Note: sklearn is optional. Components are importable without it;
# functionality is available at runtime when sklearn is installed.
# Import pipeline components that don't require sklearn at module level:
from .pipeline import MLPipeline, PipelineConfig  # pylint: disable=import-error

# Import baselines that don't strictly need sklearn at init
from .baselines import MajorityClassBaseline  # pylint: disable=import-error

# Mark sklearn-dependent components as not-available-at-import
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None

# Export symbols for runtime use; consumers should check SKLEARN_AVAILABLE
# before calling sklearn-dependent functionality.
# run_exploratory_training is imported from pipeline and available at runtime.

# Determine sklearn availability
try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from .pipeline import run_exploratory_training
    __all__ = [
        "MLPipeline",
        "PipelineConfig",
        "MajorityClassBaseline",
        "SKLEARN_AVAILABLE",
        "SHAP_AVAILABLE",
        "shap",
        "run_exploratory_training",
    ]
except ImportError:
    __all__ = [
        "MLPipeline",
        "PipelineConfig",
        "MajorityClassBaseline",
        "SKLEARN_AVAILABLE",
        "SHAP_AVAILABLE",
        "shap",
    ]