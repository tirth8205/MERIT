"""Core metrics for MERIT evaluation."""

# Lightweight base classes — safe to import eagerly (no heavy deps)
from .base import BaseMetric, MetricResult
from .device import DeviceManager


# Lazy imports — avoid loading heavy dependencies (spacy, torch, nltk) at
# package-init time so that lightweight modules like core.base can be
# imported independently.
def __getattr__(name):
    _lazy_imports = {
        # New canonical names (one file per metric)
        "LogicalConsistencyMetric": ".consistency",
        "FactualAccuracyMetric": ".factual",
        "ReasoningStepMetric": ".reasoning",
        "AlignmentMetric": ".alignment",
        # Backward-compat aliases (old "Enhanced*" names)
        "EnhancedLogicalConsistencyMetric": ".consistency",
        "EnhancedFactualAccuracyMetric": ".factual",
        "EnhancedReasoningStepMetric": ".reasoning",
        "EnhancedAlignmentMetric": ".alignment",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseMetric",
    "MetricResult",
    "DeviceManager",
    # New canonical names
    "LogicalConsistencyMetric",
    "FactualAccuracyMetric",
    "ReasoningStepMetric",
    "AlignmentMetric",
    # Backward-compat aliases
    "EnhancedLogicalConsistencyMetric",
    "EnhancedFactualAccuracyMetric",
    "EnhancedReasoningStepMetric",
    "EnhancedAlignmentMetric",
]
