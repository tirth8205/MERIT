# MERIT: Multi-dimensional Evaluation of Reasoning in Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is MERIT?

MERIT evaluates how well language models **reason**, not just whether they get the right answer. Traditional evaluation only checks accuracy—MERIT reveals *why* a model succeeds or fails.

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Logical Consistency** | Self-contradictions in responses | Safety: inconsistent reasoning is unreliable |
| **Factual Accuracy** | Correctness of factual claims | Hallucination detection |
| **Reasoning Steps** | Clarity of step-by-step logic | Explainability / interpretability |
| **Alignment** | Ethical behavior, bias, respect | AI safety |

## Installation

```bash
git clone https://github.com/yourusername/merit
cd merit
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Evaluate a model on ARC dataset
python -m merit.cli evaluate --model tinyllama-1b --dataset arc --sample-size 50

# List available models
python -m merit.cli models list

# Test a model
python -m merit.cli models test tinyllama-1b --prompt "What is 2+2?"
```

## Available Models

All models are instruction-tuned for reasoning tasks:

| Model | Size | Memory | Best For |
|-------|------|--------|----------|
| `qwen2-0.5b` | 0.5B | ~1GB | Quick tests, low memory |
| `tinyllama-1b` | 1.1B | ~2GB | Good balance |
| `phi-2` | 2.7B | ~6GB | Strong reasoning |
| `mistral-7b-instruct` | 7B | ~14GB | Strong quality |
| `llama3-8b` | 8B | ~16GB | Best quality, most benchmarked |

## Supported Datasets

| Dataset | HuggingFace ID | Task |
|---------|----------------|------|
| `arc` | allenai/ai2_arc (ARC-Challenge) | Science reasoning |
| `hellaswag` | Rowan/hellaswag | Commonsense |
| `mmlu_logic` | cais/mmlu (formal_logic) | Logical reasoning |

## Full Dataset Evaluation

```bash
# Use --sample-size 0 for full dataset
python -m merit.cli evaluate --model qwen2-0.5b --dataset arc --sample-size 0
```

## Python API

```python
from merit import (
    ModelManager,
    ExperimentConfig,
    ExperimentRunner,
    EnhancedLogicalConsistencyMetric,
    EnhancedFactualAccuracyMetric
)

# Load and test a model
manager = ModelManager()
model = manager.load_model("tinyllama-1b")
response = model.generate("What is photosynthesis?")

# Evaluate with MERIT metrics
logic_metric = EnhancedLogicalConsistencyMetric()
result = logic_metric.compute(response)
print(f"Logical Consistency: {result['score']:.2f}")

fact_metric = EnhancedFactualAccuracyMetric()
result = fact_metric.compute(response)
print(f"Factual Accuracy: {result['score']:.2f}")

# Run full experiment
config = ExperimentConfig(
    experiment_name="my_experiment",
    models=["tinyllama-1b", "qwen2-0.5b"],
    benchmarks=["arc"],
    sample_sizes=[100],
    num_runs=3
)
runner = ExperimentRunner(config)
results = runner.run_full_experiment()
```

## How Metrics Work

### Logical Consistency
Uses sentence embeddings (all-MiniLM-L6-v2) to detect semantic contradictions and sentiment inconsistencies.

### Factual Accuracy
Verifies claims using three web sources (cascading):
1. **Wikidata** - structured knowledge (confidence: 0.85)
2. **Wikipedia** - article text (confidence: 0.75)
3. **DuckDuckGo** - broad coverage (confidence: 0.65)

### Reasoning Steps
Pattern-based detection of step-by-step reasoning + coherence analysis between steps.

### Alignment
Multi-faceted analysis: bias detection, ethical principles, respectfulness, sentiment.

## Project Structure

```
merit/
├── __init__.py              # Main exports
├── cli.py                   # Command-line interface
├── core/
│   └── enhanced_metrics.py  # 4 MERIT metrics
├── experiments/
│   └── robust_evaluation.py # ExperimentRunner
├── models/
│   └── local_models.py      # ModelManager
└── validation/
    └── baseline_comparison.py # BERT-Score, ROUGE comparison
```

## Hardware Support

- **Apple Silicon (M1/M2/M3/M4)**: MPS acceleration
- **NVIDIA GPUs**: CUDA support
- **CPU**: Universal fallback

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Internet connection (for fact verification APIs)
- 1-14GB RAM (depending on model)

## License

MIT License

## Citation

```bibtex
@software{merit2024,
  title={MERIT: Multi-dimensional Evaluation of Reasoning in Transformers},
  author={Tirth Patel},
  year={2024},
  version={2.0.0}
}
```
