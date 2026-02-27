# MERIT: Multi-dimensional Evaluation of Reasoning in Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)]()

A NeurIPS-oriented framework for multi-dimensional evaluation of reasoning in language models. MERIT goes beyond accuracy to measure **logical consistency**, **factual accuracy**, **reasoning quality**, and **alignment** using both heuristic metrics and LLM-as-judge evaluation.

## Installation

```bash
git clone https://github.com/yourusername/merit
cd merit
pip install -e ".[full]"

# Required NLP model
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Evaluate a model with heuristic metrics
merit evaluate --model tinyllama-1b --dataset arc --sample-size 50

# Evaluate with LLM judge (requires ANTHROPIC_API_KEY)
merit evaluate --model tinyllama-1b --dataset gsm8k --mode judge

# Run both heuristic and judge evaluation
merit evaluate --model phi-2 --dataset bbh --mode both --output results.json

# Generate paper-ready LaTeX tables
merit report --input results.json --format latex --output paper_outputs/

# Export results as CSV
merit report --input results.json --format csv --output paper_outputs/

# Annotate responses for metric validation
merit annotate --input results.json --samples 100

# List available models
merit models list

# Test a model interactively
merit models test tinyllama-1b --prompt "Explain photosynthesis step by step."

# Compare multiple experiments
merit compare exp1.json exp2.json --output comparison.txt
```

## Metrics

MERIT evaluates four dimensions, each with heuristic and LLM-judge variants:

| Dimension | Heuristic | LLM Judge | What It Measures |
|-----------|-----------|-----------|-----------------|
| **Consistency** | Sentence embeddings + sentiment analysis | Claude rubric (1-5) | Self-contradictions in reasoning |
| **Factual** | Knowledge cache + web verification | Claude rubric (1-5) | Correctness of factual claims |
| **Reasoning** | Pattern detection + coherence analysis | Claude rubric (1-5) | Clarity of step-by-step logic |
| **Alignment** | Bias detection + ethical analysis | Claude rubric (1-5) | Safety, fairness, respect |

All metrics return a standardized `MetricResult(score, dimension, details)` on a 0-1 scale.

## Models

| Model | Size | Memory | Best For |
|-------|------|--------|----------|
| `qwen2-0.5b` | 0.5B | ~1GB | Quick tests, low memory |
| `tinyllama-1b` | 1.1B | ~2GB | Good balance |
| `phi-2` | 2.7B | ~6GB | Strong reasoning |
| `mistral-7b-instruct` | 7B | ~14GB | High quality |
| `llama3-8b` | 8B | ~16GB | Best quality |

## Datasets

| Dataset | Benchmark | Task Type |
|---------|-----------|-----------|
| `arc` | ARC-Challenge | Science reasoning (multiple choice) |
| `hellaswag` | HellaSwag | Commonsense (multiple choice) |
| `truthfulqa` | TruthfulQA | Truthfulness (multiple choice) |
| `mmlu_logic` | MMLU Formal Logic | Logical reasoning (multiple choice) |
| `gsm8k` | GSM8K | Math reasoning (open-ended) |
| `bbh` | BIG-Bench Hard | Complex reasoning (open-ended) |

## Python API

```python
from merit import BaseMetric, MetricResult, ExperimentConfig, DeviceManager

# --- Heuristic metrics ---
from merit.core.consistency import LogicalConsistencyMetric

metric = LogicalConsistencyMetric()
result = metric.compute("The sky is blue. It is a clear day.")
print(f"Consistency: {result.score:.2f}")  # MetricResult with .score, .dimension, .details

# --- LLM judge ---
from merit.core.llm_judge import LLMJudge, JudgeConfig

judge = LLMJudge(JudgeConfig())
result = judge.evaluate_consistency("The sky is blue.")
print(f"Judge score: {result.score:.2f}")

# --- Full experiment ---
from merit.experiments import ExperimentRunner

config = ExperimentConfig(
    experiment_name="my_experiment",
    models=["tinyllama-1b", "qwen2-0.5b"],
    benchmarks=["arc", "gsm8k"],
    sample_sizes=[100],
    num_runs=3,
)
runner = ExperimentRunner(config)
results = runner.run_full_experiment()

# --- Reporting ---
from merit.reporting.tables import generate_results_table

latex = generate_results_table(results)

# --- Statistical analysis ---
from merit.utils.stats import bootstrap_ci, cohens_d, aggregate_runs

ci_low, ci_high = bootstrap_ci([0.8, 0.82, 0.79])
effect = cohens_d([0.8, 0.82, 0.79], [0.6, 0.62, 0.59])
```

## Project Structure

```
merit/
├── __init__.py                  # Top-level exports (BaseMetric, MetricResult, etc.)
├── cli.py                       # CLI: evaluate, annotate, report, compare
├── core/
│   ├── base.py                  # BaseMetric, MetricResult abstractions
│   ├── device.py                # DeviceManager (MPS/CUDA/CPU)
│   ├── consistency.py           # Logical consistency metric
│   ├── factual.py               # Factual accuracy metric
│   ├── reasoning.py             # Reasoning quality metric
│   ├── alignment.py             # Alignment/safety metric
│   ├── llm_judge.py             # LLM-as-judge evaluation
│   └── knowledge_cache.py       # Deterministic knowledge cache
├── experiments/
│   ├── config.py                # ExperimentConfig dataclass
│   ├── runner.py                # ExperimentRunner
│   └── datasets.py              # Dataset loaders (ARC, GSM8K, BBH, etc.)
├── models/
│   ├── manager.py               # ModelManager
│   ├── huggingface.py           # HuggingFace adapter
│   └── ollama.py                # Ollama adapter
├── baselines/
│   ├── bertscore.py             # BERTScore baseline
│   └── geval.py                 # G-Eval baseline
├── evaluation/
│   └── annotation.py            # Claude annotation pipeline
├── reporting/
│   ├── tables.py                # LaTeX table generation
│   ├── plots.py                 # Radar charts, scaling plots
│   └── export.py                # CSV/JSON export
├── utils/
│   └── stats.py                 # Bootstrap CI, Cohen's d, aggregation
└── validation/
    └── baseline_comparison.py   # Baseline comparison utilities
```

## Hardware Support

- **Apple Silicon (M1/M2/M3/M4)**: MPS acceleration via `DeviceManager`
- **NVIDIA GPUs**: CUDA support
- **CPU**: Universal fallback

## Requirements

- Python 3.8+
- PyTorch 2.0+
- `ANTHROPIC_API_KEY` environment variable (for LLM judge and annotation)

## License

MIT License

## Citation

```bibtex
@software{merit2025,
  title={MERIT: Multi-dimensional Evaluation of Reasoning in Transformers},
  author={Tirth Patel},
  year={2025},
  version={3.0.0}
}
```
