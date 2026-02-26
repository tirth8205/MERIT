"""
MERIT: Multi-dimensional Evaluation of Reasoning in Transformers

A framework for evaluating LLM reasoning quality beyond simple accuracy metrics.
Provides four core metrics: logical consistency, factual accuracy, reasoning steps, and alignment.

Example usage:
    >>> from merit.core.enhanced_metrics import EnhancedLogicalConsistencyMetric
    >>> metric = EnhancedLogicalConsistencyMetric()
    >>> result = metric.compute("The sky is blue. Water is wet.")
    >>> print(f"Consistency score: {result['score']:.2f}")
"""

__version__ = "2.0.0"
__author__ = "Tirth Kanani"
__license__ = "MIT"

# Lazy imports — heavy dependencies (spacy, torch, transformers) are only
# loaded when the caller actually accesses the top-level names.
def __getattr__(name):
    _lazy_imports = {
        "EnhancedLogicalConsistencyMetric": "merit.core.enhanced_metrics",
        "EnhancedFactualAccuracyMetric": "merit.core.enhanced_metrics",
        "EnhancedReasoningStepMetric": "merit.core.enhanced_metrics",
        "EnhancedAlignmentMetric": "merit.core.enhanced_metrics",
        "ExperimentRunner": "merit.experiments.robust_evaluation",
        "ExperimentConfig": "merit.experiments.robust_evaluation",
        "ModelManager": "merit.models.local_models",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'merit' has no attribute {name!r}")

__all__ = [
    "EnhancedLogicalConsistencyMetric",
    "EnhancedFactualAccuracyMetric",
    "EnhancedReasoningStepMetric",
    "EnhancedAlignmentMetric",
    "ExperimentRunner",
    "ExperimentConfig",
    "ModelManager",
]
