"""
Core metrics for MERIT evaluation.
"""


# Lazy imports — avoid loading heavy dependencies (spacy, torch, nltk) at
# package-init time so that lightweight modules like core.base can be
# imported independently.
def __getattr__(name):
    _lazy_imports = {
        "EnhancedLogicalConsistencyMetric": ".enhanced_metrics",
        "EnhancedFactualAccuracyMetric": ".enhanced_metrics",
        "EnhancedReasoningStepMetric": ".enhanced_metrics",
        "EnhancedAlignmentMetric": ".enhanced_metrics",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnhancedLogicalConsistencyMetric",
    "EnhancedFactualAccuracyMetric",
    "EnhancedReasoningStepMetric",
    "EnhancedAlignmentMetric",
]
