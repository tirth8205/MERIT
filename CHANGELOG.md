# Changelog

All notable changes to MERIT are documented here.

## [3.0.0] — 2025-12-15

### Added
- Three-tier evaluation architecture: heuristic metrics, LLM judge, and annotation pipeline
- LLM-as-judge evaluation using Claude with structured rubrics (1-5 scale)
- Claude annotation pipeline for gold-standard metric validation
- Six benchmark datasets: ARC-Challenge, HellaSwag, TruthfulQA, MMLU Formal Logic, GSM8K, BIG-Bench Hard
- Statistical analysis utilities: bootstrap confidence intervals (10K resamples), Cohen's d effect sizes, Spearman rank correlations
- Paper-ready reporting: LaTeX tables (booktabs), radar charts, scaling plots, CSV/JSON export
- Knowledge cache for reproducible fact verification (Wikidata, Wikipedia, DuckDuckGo)
- Experiment runner for multi-model, multi-benchmark evaluations with configurable runs
- G-Eval and BERTScore baselines for comparison
- Full CLI: `evaluate`, `report`, `annotate`, `compare`, `models list/test`

### Changed
- Modular architecture with lazy imports for fast startup (PEP 562)
- Unified `MetricResult(score, dimension, details)` return type across all metrics
- Consolidated `DeviceManager` into single canonical source
- Separated runtime and development dependencies

### Fixed
- CLI metric mode mapping (heuristic/judge/both)
- Model manager device detection and memory estimation
- Knowledge cache thread safety

## [2.0.0] — 2025-09-01

### Added
- Four core metrics: logical consistency, factual accuracy, reasoning quality, alignment
- HuggingFace model adapters with FP16 and quantization support
- Five model adapters: TinyLlama, Phi-2, Mistral-7B, Llama3-8B, Qwen2-0.5B
- Automatic hardware detection: MPS (Apple Silicon), CUDA, CPU

### Changed
- Modular metric architecture with `BaseMetric` ABC
- Split monolithic modules into focused single-responsibility files

## [1.0.0] — 2025-06-01

### Added
- Initial multi-dimensional evaluation framework
- Core metric implementations for consistency and factual accuracy
- Basic CLI for model evaluation
