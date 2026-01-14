# MERIT: Multi-dimensional Evaluation of Reasoning in Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is MERIT?

MERIT evaluates how well language models **reason**, not just whether they get the right answer. Instead of simple accuracy scores, MERIT provides four detailed metrics:

- **Logical Consistency**: Does the model contradict itself?
- **Factual Accuracy**: Are the facts correct?
- **Reasoning Steps**: Is the step-by-step logic clear?
- **Alignment**: Are the responses ethical and respectful?

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/merit
cd merit
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run an Evaluation

```bash
# Evaluate a model on the ARC dataset
merit evaluate --model tinyllama-1b --dataset arc --sample-size 10

# List available models
merit models list

# Check system info
merit system-info
```

## Available Models

All models are instruction-tuned and suitable for reasoning tasks:

| Model | Parameters | Memory | Description |
|-------|------------|--------|-------------|
| `llama2-7b-chat` | 7B | ~14GB | Meta's Llama-2 chat model |
| `mistral-7b-instruct` | 7B | ~14GB | Mistral's instruction model |
| `phi-2` | 2.7B | ~6GB | Microsoft Phi-2 - excellent reasoning |
| `tinyllama-1b` | 1.1B | ~2GB | Lightweight, good for quick tests |
| `qwen2-0.5b` | 0.5B | ~1GB | Smallest instruction model |

### Model Selection by Memory

```bash
# 16GB+ RAM
merit evaluate --model mistral-7b-instruct --dataset arc --sample-size 50

# 8GB RAM
merit evaluate --model phi-2 --dataset arc --sample-size 50

# 4GB RAM
merit evaluate --model tinyllama-1b --dataset arc --sample-size 50

# 2GB RAM
merit evaluate --model qwen2-0.5b --dataset arc --sample-size 50
```

## Supported Datasets

MERIT evaluates models on standard reasoning benchmarks:

| Dataset | Description | Task Type |
|---------|-------------|-----------|
| `arc` | AI2 Reasoning Challenge | Multiple choice science questions |
| `hellaswag` | HellaSwag | Commonsense reasoning |
| `mmlu_logic` | MMLU Formal Logic | Logical reasoning |

## CLI Commands

```bash
# Initialize MERIT configuration
merit init

# Evaluate a model
merit evaluate --model <model> --dataset <dataset> --sample-size <n>

# List available models
merit models list

# Test a specific model
merit models test <model>

# Show system information
merit system-info

# Run a full experiment from config file
merit run <config.yaml>
```

## Python API

```python
from merit import (
    ModelManager,
    ExperimentConfig,
    ExperimentRunner,
    EnhancedLogicalConsistencyMetric
)

# Quick evaluation
manager = ModelManager()
model = manager.load_model("tinyllama-1b")
response = model.generate("What is 2+2?", temperature=0.7)

# Evaluate with metrics
metric = EnhancedLogicalConsistencyMetric()
result = metric.compute(response)
print(f"Consistency: {result['score']:.2f}")

# Full experiment
config = ExperimentConfig(
    experiment_name="my_experiment",
    models=["tinyllama-1b", "qwen2-0.5b"],
    benchmarks=["arc"],
    sample_sizes=[50],
    num_runs=3
)

runner = ExperimentRunner(config)
results = runner.run_full_experiment()
```

## Experiment Configuration

Create a YAML config file for reproducible experiments:

```yaml
experiment:
  name: "reasoning_evaluation"
  models: ["tinyllama-1b", "phi-2"]
  datasets: ["arc", "hellaswag"]
  sample_sizes: [50, 100]
  num_runs: 3
  temperature: 0.7
  max_tokens: 200
  random_seed: 42
  metrics:
    - logical_consistency
    - factual_accuracy
    - reasoning_steps
    - alignment
  output_dir: "./results"
```

Run with:
```bash
merit run experiment.yaml
```

## Metrics Explained

### Logical Consistency (0.0-1.0)
Detects semantic contradictions and logical inconsistencies using sentence embeddings and sentiment analysis.

### Factual Accuracy (0.0-1.0)
Verifies factual claims against knowledge bases (structured facts + Wikipedia API).

### Reasoning Steps (0.0-1.0)
Evaluates the quality and clarity of step-by-step reasoning patterns.

### Alignment (0.0-1.0)
Assesses ethical alignment, respectfulness, and bias detection.

## Project Structure

```
merit/
├── core/                    # Evaluation metrics
│   └── enhanced_metrics.py  # Core metric implementations
├── models/                  # Model adapters
│   └── local_models.py      # Local model management
├── experiments/             # Experiment framework
│   └── robust_evaluation.py # Experiment runner
├── config/                  # Configuration management
├── knowledge/               # Knowledge base for fact checking
├── validation/              # Human validation and baselines
├── datasets/                # Dataset loaders
├── cli.py                   # Command-line interface
└── tests/                   # Test suite
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=merit --cov-report=html
```

## Hardware Support

- **Apple Silicon (M1/M2/M3/M4)**: Native MPS acceleration
- **NVIDIA GPUs**: CUDA support
- **CPU**: Works on any system

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- ~2-14GB RAM (depending on model)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{merit2024,
  title={MERIT: Multi-dimensional Evaluation of Reasoning in Transformers},
  author={Tirth Patel},
  year={2024},
  version={2.0.0}
}
```
