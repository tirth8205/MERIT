# MERIT Technical Details

## Design Philosophy

MERIT is built on the principle that **reasoning quality cannot be reduced to a single accuracy score**. This document explains the technical decisions behind the framework.

## Architecture Decisions

### 1. Why Four Metrics?

We chose these four dimensions based on analysis of LLM failure modes:

**Logical Consistency**
- **Motivation**: Models often produce self-contradictory outputs that humans miss
- **Technical approach**: Semantic embedding similarity + sentiment analysis
- **Trade-off**: False positives on rephrasing vs. missing subtle contradictions
- **Why not formal logic?**: Most LLM outputs don't follow formal logical syntax

**Factual Accuracy**
- **Motivation**: Hallucinated facts are a critical safety concern
- **Technical approach**: Multi-source verification (structured DB + Wikipedia API)
- **Trade-off**: Completeness of knowledge base vs. verification speed
- **Design choice**: Hybrid approach balances coverage and latency

**Reasoning Steps**
- **Motivation**: Opaque reasoning prevents human oversight (scalable oversight problem)
- **Technical approach**: Pattern-based detection + coherence analysis
- **Trade-off**: Heuristic-based vs. ML-based step detection
- **Why heuristics?**: More interpretable and doesn't require labeled data

**Alignment**
- **Motivation**: Detecting bias and ethical concerns is core to AI safety
- **Technical approach**: Multi-faceted analysis (bias, respectfulness, principles)
- **Trade-off**: Pattern matching vs. contextual understanding
- **Future work**: Integrate value alignment frameworks (e.g., Constitutional AI)

### 2. Statistical Methodology

**Why multiple runs with different seeds?**
- Many benchmarks have randomness in sampling and generation
- Single runs can be misleading due to variance
- Statistical tests require repeated measurements

**Why t-tests AND non-parametric tests?**
- T-tests: Assume normality, more powerful when assumptions hold
- Wilcoxon/Mann-Whitney: Distribution-free, robust to outliers
- We report both for transparency

**Why effect sizes (Cohen's d)?**
- Statistical significance ≠ practical significance
- Small differences can be significant with large samples
- Effect sizes quantify magnitude of differences

### 3. Hardware Optimization

**Device priority: MPS > CUDA > CPU**
- MPS (Apple Silicon): Excellent performance/watt, unified memory
- CUDA: Best raw performance for large models
- CPU: Universal fallback

**Memory management**
- Dynamic model loading/unloading to support multiple models
- FP16 on GPU/MPS for 2x memory reduction
- Batch size optimization based on available memory

**Why support Ollama?**
- Quantized models enable evaluation on consumer hardware
- ~50% memory reduction vs. full precision
- Critical for accessibility and reproducibility

## Implementation Decisions

### 1. Semantic Similarity via Sentence-Transformers

**Why `all-MiniLM-L6-v2`?**
- Good balance of speed (384-dim embeddings) and quality
- Trained on diverse semantic textual similarity tasks
- Works well for contradiction detection
- Alternative: `all-mpnet-base-v2` (higher quality, slower)

### 2. Knowledge Base Architecture

**Why SQLite + Wikipedia hybrid?**
- SQLite: Fast local lookups for common facts (~40 curated facts)
- Wikipedia API: Broad coverage for long-tail facts
- 7-day caching: Balances freshness and API rate limits

**Design choice**: Facts are stored as (subject, predicate, object) triples
- Enables graph-based reasoning in future versions
- Compatible with knowledge graph frameworks (RDFLib, etc.)

### 3. Baseline Comparisons

**Why these specific baselines?**
- BERT-Score: Industry standard for semantic similarity
- ROUGE: Widely used, enables comparison with existing work
- LLM Judge: Tests whether local models can evaluate reasoning

**Design decision**: All baselines are optional dependencies
- Core framework works without them
- Enables use in resource-constrained environments

## Limitations & Future Work

### Current Limitations

1. **Language**: English-only (sentence-transformers, spaCy models)
2. **Knowledge base coverage**: Limited to ~40 curated facts + Wikipedia
3. **Reasoning step detection**: Heuristic-based, may miss implicit reasoning
4. **Computational cost**: Full evaluation with statistical validation takes hours

### Planned Improvements

1. **Multi-language support**: Leverage multilingual sentence transformers
2. **Formal logic integration**: Add SMT solver for verifiable reasoning chains
3. **Adversarial robustness**: Test metrics against deliberately misleading outputs
4. **Human-in-the-loop refinement**: Active learning for metric improvement
5. **Causal reasoning**: Extend beyond correlation to causal inference evaluation

### Research Questions

**Metric validation**
- How well do automated metrics correlate with expert human judgment?
- Do metrics generalize across domains (math, ethics, science)?
- What failure modes are metrics blind to?

**Scaling laws**
- Do reasoning quality metrics improve monotonically with model size?
- Is there a "reasoning quality tax" for efficiency optimizations (quantization, pruning)?
- Can reasoning quality be improved post-training without accuracy loss?

**Safety applications**
- Can consistency metrics detect deceptive alignment?
- Do models reason differently on safety-critical vs. benign tasks?
- Which metric violations predict real-world failures?

## Reproducibility Guarantees

### Fixed Components
- Random seeds set across NumPy, PyTorch, and HuggingFace transformers
- Deterministic CUDA operations (when possible)
- Versioned dependencies in requirements.txt

### Sources of Variance
1. **Model non-determinism**: Some ops are non-deterministic even with seeds
2. **Wikipedia API**: Facts may change between queries
3. **Hardware differences**: FP16 vs. FP32 can cause small numerical differences

### Recommendation
Run experiments with `num_runs >= 3` and report confidence intervals

## Performance Benchmarks

**Approximate runtime** (on M1 Max 32GB, GPT-2 Medium):
- Single example evaluation: 2-5 seconds
- ARC dataset (100 examples): ~5 minutes
- Full experiment (3 models, 2 datasets, 5 runs): ~2 hours

**Memory requirements**:
- GPT-2 Medium: ~2GB
- Phi-3 Mini: ~8GB
- Mistral-7B: ~14GB
- Add ~2GB for metric models (sentence-transformers, spaCy)

## Code Quality Standards

- **Type hints**: All public APIs have type annotations
- **Docstrings**: Google-style docstrings for all classes/functions
- **Testing**: pytest with >80% coverage on core modules
- **Error handling**: Graceful degradation when optional dependencies missing
- **Logging**: Configurable via standard Python logging module

## Contributing Guidelines

For those extending MERIT:

1. **New metrics**: Inherit from `BaseMetric`, implement `compute()` method
2. **New models**: Inherit from `LocalModelAdapter`, implement `generate()` method
3. **New datasets**: Inherit from `ReasoningDataset`, implement `load()` method
4. **Tests required**: All new features must include pytest tests
5. **Documentation**: Update README.md and inline docstrings

## References

**Semantic similarity**
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Evaluation methodologies**
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT
- Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries

**AI Safety relevance**
- Bowman et al. (2022). Measuring Progress on Scalable Oversight for Large Language Models
- Ganguli et al. (2023). The Capacity for Moral Self-Correction in Large Language Models

---

**Version**: 2.0.0
**Last Updated**: January 2026
**Author**: Tirth Patel
