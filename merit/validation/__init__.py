"""
Validation framework for MERIT metrics.
"""

from .baseline_comparison import (
    BaselineComparator,
    BERTScoreBaseline,
    ROUGEBaseline,
)

__all__ = [
    "BaselineComparator",
    "BERTScoreBaseline",
    "ROUGEBaseline",
]
