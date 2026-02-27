"""Baseline comparison methods for validating MERIT metrics."""
from merit.baselines.bertscore import BERTScoreBaseline
from merit.baselines.geval import GEvalBaseline

__all__ = ["BERTScoreBaseline", "GEvalBaseline"]
