"""
MERIT Quick Start
=================

Demonstrates the core evaluation pipeline:
1. Evaluate a single response across all four dimensions
2. Inspect detailed metric breakdowns
3. Configure a full experiment (optional)

Usage:
    python examples/quickstart.py
"""

from merit.core.consistency import LogicalConsistencyMetric
from merit.core.factual import FactualAccuracyMetric
from merit.core.reasoning import ReasoningStepMetric
from merit.core.alignment import AlignmentMetric

# A sample model response to evaluate
response = (
    "Photosynthesis is the process by which plants convert sunlight into energy. "
    "During this process, plants absorb carbon dioxide from the atmosphere and "
    "water from the soil. Using chlorophyll in their leaves, they convert these "
    "inputs into glucose and oxygen. The glucose provides energy for the plant's "
    "growth and metabolism, while the oxygen is released into the atmosphere."
)

# Initialize all four metrics
metrics = [
    LogicalConsistencyMetric(),
    FactualAccuracyMetric(),
    ReasoningStepMetric(),
    AlignmentMetric(),
]

# Evaluate
print("MERIT Evaluation Results")
print("=" * 50)

for metric in metrics:
    result = metric.compute(response)
    print(f"  {result.dimension:20s}  {result.score:.3f}")

print("=" * 50)
print()

# Each MetricResult contains detailed breakdowns
metric = LogicalConsistencyMetric()
result = metric.compute(response)
print(f"Detailed breakdown for '{result.dimension}':")
for key, value in result.details.items():
    print(f"  {key}: {value}")
