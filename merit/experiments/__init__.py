"""
Robust experimental design and evaluation framework for MERIT.
"""

from .config import ExperimentConfig
from .runner import ExperimentRunner, create_default_config
from .datasets import INSTRUCTION_TEMPLATES, load_dataset

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "create_default_config",
    "INSTRUCTION_TEMPLATES",
    "load_dataset",
]
