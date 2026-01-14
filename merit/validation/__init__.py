"""
Validation framework for MERIT metrics.
"""

from .human_validation import (
    HumanAnnotationCollector,
    MetricValidator,
    InterAnnotatorAgreement
)
from .baseline_comparison import (
    BaselineComparator,
    BERTScoreBaseline,
    ROUGEBaseline,
    LLMJudgeBaseline
)

__all__ = [
    "HumanAnnotationCollector",
    "MetricValidator", 
    "InterAnnotatorAgreement",
    "BaselineComparator",
    "BERTScoreBaseline",
    "ROUGEBaseline",
    "LLMJudgeBaseline"
]