<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# MERIT: Multi-dimensional Evaluation of Reasoning in Transformers

## Overview

MERIT is a comprehensive framework for evaluating reasoning capabilities in large language models (LLMs) beyond simple accuracy metrics. While existing evaluation methods often focus primarily on whether models produce correct answers, MERIT provides a multi-dimensional assessment examining the logical consistency, factual accuracy, step-by-step reasoning quality, and ethical alignment of model outputs.

## Features

- **Multi-dimensional Metrics**: Evaluate reasoning across multiple dimensions including logical consistency, factual accuracy, reasoning steps, and ethical alignment
- **Model-agnostic Design**: Support for multiple LLM providers through a flexible adapter architecture
- **Benchmark Integration**: Standardized evaluation across established benchmarks (ARC, HellaSwag, MMLU)
- **Visualization Tools**: Comprehensive analysis displays for reasoning flow and performance comparison
- **Rule-based Evaluation**: Transparent evaluation using deterministic pattern recognition rather than LLM-based judgments


## Installation

```bash
# Clone the repository
git clone https://github.com/tirth8205/MERIT
cd merit

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```


## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`


## Quick Start

```python
from merit.models.adapters.gemini import GeminiAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry

# Initialize model adapter with your API key
model_adapter = GeminiAdapter(api_key="<YOUR_API_KEY>", model_name="gemini-2.0-flash-001")

# Create evaluator with default metrics
evaluator = ReasoningEvaluator(
    metric_registry=get_default_metric_registry(),
    model_adapter=model_adapter
)

# Evaluate a simple reasoning example
result = evaluator.evaluate(
    prompt="What happens when you mix baking soda and vinegar?",
    reference="Baking soda (sodium bicarbonate) and vinegar (acetic acid) react to form carbon dioxide gas, water, and sodium acetate."
)

# Print the evaluation results
print(result)
```


## Running Benchmarks

The framework includes tools for evaluating models on standard benchmarks:

```bash
python -m merit.benchmark_evaluation --benchmark arc --model gemini-2.0-flash-001 --adapter gemini --api_key <YOUR_API_KEY> --samples 50
```

Available benchmarks:

- `arc` - AI2 Reasoning Challenge
- `hellaswag` - Common sense reasoning benchmark
- `mmlu_logic` - Logic subset of MMLU


## Visualization

Generate visualizations from benchmark results:

```bash
python generate_visualizations.py
```

This will create plots and an HTML report in the `results/` directory.

## Project Structure

```
merit/
├── core/                   # Core evaluation framework
│   ├── evaluation.py       # Main evaluator implementation
│   ├── metrics.py          # Reasoning metrics
│   ├── utils.py            # Utility functions 
│   └── visualization.py    # Visualization tools
├── datasets/               # Dataset handling
│   └── loaders.py          # Dataset loading utilities
├── models/                 # Model adapters
│   ├── adapters/           # Model-specific adapters
│   │   ├── gemini.py       # Google Gemini adapter
│   │   └── huggingface.py  # Hugging Face models adapter
│   └── prompts/            # Prompt templates
├── examples/               # Example usage scripts
│   ├── basic_example.py    # Simple evaluation example
│   └── dataset_evaluation.py  # Dataset evaluation example
└── benchmark_evaluation.py # Benchmark evaluation script
```


## Results

Evaluations across established benchmarks reveal interesting patterns in model reasoning:


| Benchmark | Samples | Accuracy | Logical Consistency |
| :-- | :-- | :-- | :-- |
| ARC | 1172 | 77.9% | 99.7% |
| HellaSwag | 200 | 80.0% | 99.9% |
| MMLU Logic | 126 | 78.6% | 99.5% |

For each benchmark, we used partial samples from the original datasets (ARC Challenge test set, HellaSwag validation set, and MMLU Logic test set). These results suggest that modern LLMs can construct internally coherent reasoning paths even when occasionally reaching incorrect conclusions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- This project was developed as part of research into reasoning evaluation methodologies for large language models.

