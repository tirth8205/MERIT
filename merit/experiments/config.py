"""Experiment configuration for MERIT."""
import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for MERIT experiments"""
    experiment_name: str
    models: List[str]
    benchmarks: List[str]
    sample_sizes: List[int]
    num_runs: int = 3
    temperature: float = 0.7  # Higher temperature for more varied responses
    max_tokens: int = 200  # Shorter responses to avoid repetition
    random_seed: int = 42
    metrics: List[str] = field(default_factory=lambda: ["logical_consistency", "factual_accuracy", "reasoning_steps", "alignment"])
    baseline_methods: List[str] = field(default_factory=list)
    statistical_tests: List[str] = field(default_factory=list)
    output_dir: str = "experiments"
    use_instruction_format: bool = True  # Enable instruction formatting
    metric_mode: str = "heuristic"  # "heuristic", "llm_judge", or "both"

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
