"""
Robust experimental design and evaluation framework for MERIT.
"""

from .robust_evaluation import (
    ExperimentConfig,
    ExperimentRunner,
    create_default_config
)

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner", 
    "create_default_config"
]