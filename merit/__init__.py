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

# Expose main classes for convenient imports
from merit.core.enhanced_metrics import (
    EnhancedLogicalConsistencyMetric,
    EnhancedFactualAccuracyMetric,
    EnhancedReasoningStepMetric,
    EnhancedAlignmentMetric,
)

from merit.experiments.robust_evaluation import (
    ExperimentRunner,
    ExperimentConfig,
)

from merit.models.local_models import ModelManager

__all__ = [
    "EnhancedLogicalConsistencyMetric",
    "EnhancedFactualAccuracyMetric",
    "EnhancedReasoningStepMetric",
    "EnhancedAlignmentMetric",
    "ExperimentRunner",
    "ExperimentConfig",
    "ModelManager",
]
