# Contributing to MERIT

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/tirth8205/MERIT.git
cd MERIT
pip install -e ".[dev,all]"
python -m spacy download en_core_web_sm

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — follow the existing code patterns and style.

3. **Run tests** before committing:
   ```bash
   pytest tests/ -v
   ```

4. **Format your code**:
   ```bash
   black merit/ tests/
   ```

5. **Submit a pull request** against `main`.

## Code Style

- **Formatter**: [Black](https://github.com/psf/black) with 100-character line length
- **Imports**: Standard library, third-party, then local — sorted with isort
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Required for public classes and functions

## Project Structure

- `merit/core/` — Metric implementations (consistency, factual, reasoning, alignment)
- `merit/models/` — Model adapters (HuggingFace, Ollama)
- `merit/experiments/` — Experiment runner and dataset loaders
- `merit/reporting/` — LaTeX tables, plots, and export
- `merit/utils/` — Statistical utilities
- `tests/` — Test suite (mirrors `merit/` structure)

## Adding a New Metric

1. Create a new file in `merit/core/` that subclasses `BaseMetric`
2. Implement the `compute(text: str) -> MetricResult` method
3. Add tests in `tests/`
4. Update the metric registry in `__init__.py`

## Adding a New Model Adapter

1. Subclass `BaseModelAdapter` in `merit/models/`
2. Register it in `merit/models/manager.py`
3. Add tests

## Reporting Issues

- Use the [bug report template](https://github.com/tirth8205/MERIT/issues/new?template=bug_report.yml) for bugs
- Use the [feature request template](https://github.com/tirth8205/MERIT/issues/new?template=feature_request.yml) for ideas
