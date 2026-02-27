# MERIT Technical Details

## Design Philosophy

MERIT is built on the principle that **reasoning quality cannot be reduced to a single accuracy score**. This document explains the technical decisions behind the framework.

## Architecture Overview

MERIT v3.0 uses a **three-tier evaluation architecture**:

1. **Heuristic metrics** (fast, free, deterministic) — pattern-based and embedding-based analysis
2. **LLM-as-judge** (accurate, API cost) — structured rubric evaluation via Claude
3. **Annotation pipeline** (gold standard) — large-scale Claude annotation for metric validation

All metrics implement the `BaseMetric` ABC and return `MetricResult(score, dimension, details)` with scores normalized to 0-1.

## Architecture Decisions

### 1. Why Four Dimensions?

We chose these four dimensions based on analysis of LLM failure modes:

**Logical Consistency**
- **Motivation**: Models often produce self-contradictory outputs that humans miss
- **Heuristic approach**: Semantic embedding similarity (sentence-transformers) + VADER sentiment analysis + spaCy negation detection
- **LLM-judge approach**: Structured 1-5 rubric evaluating internal coherence
- **Trade-off**: False positives on rephrasing vs. missing subtle contradictions

**Factual Accuracy**
- **Motivation**: Hallucinated facts are a critical safety concern
- **Heuristic approach**: Multi-source web verification with knowledge cache:
  1. **Wikidata** (structured knowledge, highest confidence: 0.85)
  2. **Wikipedia** (detailed text, confidence: 0.75)
  3. **DuckDuckGo Instant Answers** (broad coverage, confidence: 0.65)
- **LLM-judge approach**: Claude evaluates factual correctness against reference
- **Reproducibility**: `KnowledgeCache` stores verification results for deterministic reruns

**Reasoning Steps**
- **Motivation**: Opaque reasoning prevents human oversight (scalable oversight problem)
- **Heuristic approach**: Pattern-based step detection + coherence analysis via embeddings
- **LLM-judge approach**: Rubric-based evaluation of step clarity and logical flow
- **Trade-off**: Heuristic-based vs. ML-based step detection

**Alignment**
- **Motivation**: Detecting bias and ethical concerns is core to AI safety
- **Heuristic approach**: Multi-faceted analysis (bias detection, respectfulness, ethical principles, sentiment)
- **LLM-judge approach**: Rubric evaluating safety, fairness, and respect
- **Trade-off**: Pattern matching vs. contextual understanding

### 2. Three-Tier Evaluation

| Tier | Speed | Cost | Accuracy | Use Case |
|------|-------|------|----------|----------|
| Heuristic | Fast (~1s/sample) | Free | Moderate | Development, large-scale screening |
| LLM Judge | Moderate (~3s/sample) | API cost | High | Paper experiments |
| Annotation | Slow (~5s/sample) | API cost | Gold standard | Metric validation |

### 3. Statistical Methodology

- **Multiple runs with different seeds**: Captures variance from sampling and generation
- **Bootstrap confidence intervals**: Non-parametric CI estimation (10,000 resamples)
- **Cohen's d effect sizes**: Quantifies practical significance beyond p-values
- **Spearman correlation with CI**: Measures metric agreement across evaluation tiers

### 4. Hardware Optimization

**Device priority: MPS > CUDA > CPU**
- MPS (Apple Silicon): Excellent performance/watt, unified memory
- CUDA: Best raw performance for large models
- CPU: Universal fallback

**Memory management**
- Dynamic model loading/unloading to support multiple models
- FP16 on GPU/MPS for 2x memory reduction
- Repetition penalty to prevent generation loops

## Implementation Decisions

### 1. BaseMetric ABC + MetricResult

All metrics implement a standardized interface:

```python
class BaseMetric(ABC):
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> str: ...
    def compute(self, response, reference=None, **kwargs) -> MetricResult: ...

@dataclass
class MetricResult:
    score: float      # Clamped to [0, 1]
    dimension: str    # One of: consistency, factual, reasoning, alignment
    details: dict     # Metric-specific analysis details
```

### 2. Lazy Imports (PEP 562)

Heavy dependencies (torch, transformers, sentence-transformers, spaCy) are loaded lazily via `__getattr__` in `__init__.py` files. This allows importing lightweight base classes (`BaseMetric`, `MetricResult`) without triggering multi-GB library loads.

### 3. Knowledge Cache

Fact verification results are cached in a file-backed JSON store (`KnowledgeCache`). Running with `reproducible=True` uses only cached results, ensuring deterministic evaluation across runs.

### 4. LLM-as-Judge Rubrics

Each dimension has a structured 1-5 rubric:
- **Score 1**: Critical failures (contradictions, fabricated facts, no reasoning, harmful content)
- **Score 3**: Moderate quality (minor issues, partial reasoning)
- **Score 5**: Excellent (fully coherent, accurate, clear, safe)

Scores are normalized to 0-1 via `(score - 1) / 4.0`.

### 5. Baseline Comparisons

- **BERTScore** (Zhang et al., 2020): Token-level BERT embeddings for precision/recall/F1
- **G-Eval** (Liu et al., 2023): LLM chain-of-thought evaluation baseline

### 6. Instruction-Tuned Models

All supported models are instruction-tuned:

| Model | Parameters | Type |
|-------|------------|------|
| qwen2-0.5b | 0.5B | Instruction-tuned |
| tinyllama-1b | 1.1B | Chat-tuned |
| phi-2 | 2.7B | Instruction-tuned |
| phi-3-mini | 3.8B | Instruction-tuned |
| codellama-7b-instruct | 7B | Code instruction-tuned |
| mistral-7b-instruct | 7B | Instruction-tuned |
| llama3-8b | 8B | Instruction-tuned |

### 7. Semantic Similarity via Sentence-Transformers

**Why `all-MiniLM-L6-v2`?**
- Good balance of speed (384-dim embeddings) and quality
- Trained on diverse semantic textual similarity tasks
- Works well for contradiction detection

## Supported Datasets

| Dataset | HuggingFace ID | Task Type |
|---------|----------------|-----------|
| ARC | allenai/ai2_arc (ARC-Challenge) | Science reasoning |
| HellaSwag | Rowan/hellaswag | Commonsense |
| TruthfulQA | truthfulqa/truthful_qa | Truthfulness |
| MMLU Logic | cais/mmlu (formal_logic) | Logical reasoning |
| GSM8K | openai/gsm8k | Math reasoning |
| BBH | lukaemon/bbh (logical_deduction) | Complex reasoning |

## Project Structure

```
merit/
├── __init__.py                  # Top-level exports (lazy imports for heavy deps)
├── cli.py                       # CLI: evaluate, annotate, report, compare
├── core/
│   ├── base.py                  # BaseMetric ABC, MetricResult dataclass
│   ├── device.py                # DeviceManager (MPS/CUDA/CPU)
│   ├── consistency.py           # Logical consistency metric
│   ├── factual.py               # Factual accuracy metric (with knowledge cache)
│   ├── reasoning.py             # Reasoning quality metric
│   ├── alignment.py             # Alignment/safety metric
│   ├── llm_judge.py             # LLM-as-judge evaluation (Claude rubrics)
│   └── knowledge_cache.py       # File-backed knowledge cache
├── experiments/
│   ├── config.py                # ExperimentConfig dataclass
│   ├── runner.py                # ExperimentRunner (heuristic + judge + baselines)
│   └── datasets.py              # Dataset loaders (ARC, GSM8K, BBH, etc.)
├── models/
│   ├── base.py                  # BaseModelAdapter ABC
│   ├── manager.py               # ModelManager
│   ├── huggingface.py           # HuggingFace model adapters
│   ├── device.py                # Model-specific device/memory utilities
│   └── ollama.py                # Ollama adapter
├── baselines/
│   ├── bertscore.py             # BERTScore baseline
│   └── geval.py                 # G-Eval baseline
├── evaluation/
│   └── annotation.py            # Claude annotation pipeline
├── reporting/
│   ├── tables.py                # LaTeX table generation
│   ├── plots.py                 # Radar charts, scaling plots, heatmaps
│   └── export.py                # CSV/JSON export
└── utils/
    └── stats.py                 # Bootstrap CI, Cohen's d, Spearman, aggregation
```

## Limitations

1. **Language**: English-only (sentence-transformers, spaCy models)
2. **Fact verification**: Requires internet access (or pre-populated knowledge cache)
3. **Reasoning step detection**: Heuristic-based, may miss implicit reasoning
4. **Model size**: Largest tested is 8B parameters
5. **LLM judge**: Requires Anthropic API key and incurs per-token cost

## Reproducibility

### Fixed Components
- Random seeds set across NumPy, PyTorch, and HuggingFace transformers
- Deterministic dataset sampling
- Knowledge cache for reproducible fact verification
- Versioned dependencies in requirements.txt

### Sources of Variance
1. **Temperature > 0**: Non-deterministic generation
2. **Wikipedia API**: Facts may change between queries (mitigated by knowledge cache)
3. **Hardware differences**: FP16 vs. FP32 numerical differences
4. **LLM judge**: Model API responses may vary slightly

### Recommendation
Run experiments with `num_runs >= 3` and report mean +/- std with bootstrap CIs.

## Performance Benchmarks

**Approximate runtime** (Apple M1, TinyLlama-1B):
- Single sample (heuristic): 2-5 seconds
- Single sample (heuristic + judge): 5-8 seconds
- 50 samples: ~3-8 minutes
- Full experiment (3 models, 6 datasets, 3 runs): ~2-4 hours

**Memory requirements**:
- Qwen2-0.5B: ~1GB
- TinyLlama-1B: ~2GB
- Phi-2: ~6GB
- Llama3/Mistral-7B: ~14-16GB
- Add ~1GB for metric models (sentence-transformers, spaCy)

## Contributing

For those extending MERIT:

1. **New metrics**: Subclass `BaseMetric` in `merit/core/`, implement `compute()` returning `MetricResult`
2. **New models**: Subclass `BaseModelAdapter` in `merit/models/huggingface.py`
3. **New datasets**: Add loading logic to `merit/experiments/datasets.py`
4. **New baselines**: Add to `merit/baselines/`
5. **Tests required**: All new features must include pytest tests

## References

**Semantic similarity**
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Evaluation methodologies**
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT
- Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment

**AI Safety relevance**
- Bowman et al. (2022). Measuring Progress on Scalable Oversight for Large Language Models

---

**Version**: 3.0.0
**Author**: Tirth Patel
