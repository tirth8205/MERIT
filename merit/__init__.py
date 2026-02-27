"""MERIT: Multi-dimensional Evaluation of Reasoning in Transformers."""
__version__ = "3.0.0"
__author__ = "Tirth Kanani"

from merit.core.base import BaseMetric, MetricResult
from merit.core.device import DeviceManager
from merit.experiments.config import ExperimentConfig


# Lazy imports for heavy modules
def __getattr__(name):
    _lazy = {
        "LogicalConsistencyMetric": "merit.core.consistency",
        "FactualAccuracyMetric": "merit.core.factual",
        "ReasoningStepMetric": "merit.core.reasoning",
        "AlignmentMetric": "merit.core.alignment",
        # Backward compat
        "EnhancedLogicalConsistencyMetric": "merit.core.consistency",
        "EnhancedFactualAccuracyMetric": "merit.core.factual",
        "EnhancedReasoningStepMetric": "merit.core.reasoning",
        "EnhancedAlignmentMetric": "merit.core.alignment",
        "ExperimentRunner": "merit.experiments.runner",
        "ModelManager": "merit.models.manager",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
