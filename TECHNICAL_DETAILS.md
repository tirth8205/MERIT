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
- **Technical approach**: Multi-source web verification using three APIs:
  1. **Wikidata** (structured knowledge, highest confidence: 0.85)
  2. **Wikipedia** (detailed text, confidence: 0.75)
  3. **DuckDuckGo Instant Answers** (broad coverage, confidence: 0.65)
- **Trade-off**: API latency vs. verification accuracy
- **Design choice**: Cascading verification - tries most reliable source first, falls back to broader sources

**Reasoning Steps**
- **Motivation**: Opaque reasoning prevents human oversight (scalable oversight problem)
- **Technical approach**: Pattern-based detection + coherence analysis
- **Trade-off**: Heuristic-based vs. ML-based step detection
- **Why heuristics?**: More interpretable and doesn't require labeled data

**Alignment**
- **Motivation**: Detecting bias and ethical concerns is core to AI safety
- **Technical approach**: Multi-faceted analysis (bias, respectfulness, principles)
- **Trade-off**: Pattern matching vs. contextual understanding

### 2. Statistical Methodology

**Why multiple runs with different seeds?**
- Many benchmarks have randomness in sampling and generation
- Single runs can be misleading due to variance
- Statistical tests require repeated measurements

**Why effect sizes?**
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
- Repetition penalty to prevent generation loops

## Implementation Decisions

### 1. Instruction-Tuned Models Only

All supported models are instruction-tuned for reasoning tasks:

| Model | Parameters | Type |
|-------|------------|------|
| llama2-7b-chat | 7B | Chat-tuned |
| mistral-7b-instruct | 7B | Instruction-tuned |
| phi-2 | 2.7B | Instruction-tuned |
| tinyllama-1b | 1.1B | Chat-tuned |
| qwen2-0.5b | 0.5B | Instruction-tuned |

**Why no base models (e.g., GPT-2)?**
- Base models don't follow instructions
- They generate repetitive, pattern-continuation outputs
- 0% task accuracy on reasoning benchmarks
- Instruction-tuned models achieve 60%+ accuracy

### 2. Semantic Similarity via Sentence-Transformers

**Why `all-MiniLM-L6-v2`?**
- Good balance of speed (384-dim embeddings) and quality
- Trained on diverse semantic textual similarity tasks
- Works well for contradiction detection

### 3. Fact Verification Architecture

**Why Wikidata + Wikipedia + DuckDuckGo?**
- **Wikidata**: Structured knowledge graph with high reliability for entity facts
- **Wikipedia**: Detailed text coverage for complex claims
- **DuckDuckGo**: Broad web coverage as fallback, no API key required
- **In-memory caching**: Avoids repeated API calls within a session
- **All free APIs**: No cost, reproducible by paper reviewers

### 4. Instruction Templates

Prompts are formatted to help models understand the task:

```
Answer the following multiple choice question. Choose the best answer
from the given options and explain your reasoning briefly.

Question: {question}

Instructions: Select ONE answer from the options above. Start your
response with the letter of your choice (A, B, C, or D), then briefly
explain why.
```

## Supported Datasets

All evaluation uses real benchmark datasets from HuggingFace:

| Dataset | HuggingFace ID | Task Type |
|---------|----------------|-----------|
| ARC | allenai/ai2_arc (ARC-Challenge) | Science reasoning |
| HellaSwag | Rowan/hellaswag | Commonsense |
| MMLU Logic | cais/mmlu (formal_logic) | Logical reasoning |

## Limitations

1. **Language**: English-only (sentence-transformers, spaCy models)
2. **Fact verification**: Requires internet access; limited by Wikipedia/Wikidata coverage
3. **Reasoning step detection**: Heuristic-based, may miss implicit reasoning
4. **Model size**: Largest supported is 7B parameters
5. **API latency**: Web-based fact checking adds ~1-2 seconds per claim

## Reproducibility

### Fixed Components
- Random seeds set across NumPy, PyTorch, and HuggingFace transformers
- Deterministic dataset sampling
- Versioned dependencies in requirements.txt

### Sources of Variance
1. **Temperature > 0**: Non-deterministic generation
2. **Wikipedia API**: Facts may change between queries
3. **Hardware differences**: FP16 vs. FP32 numerical differences

### Recommendation
Run experiments with `num_runs >= 3` and report mean ± std

## Performance Benchmarks

**Approximate runtime** (Apple M1, TinyLlama-1B):
- Single sample: 2-5 seconds
- 50 samples: ~3 minutes
- Full experiment (3 models, 2 datasets, 3 runs): ~30 minutes

**Memory requirements**:
- Qwen2-0.5B: ~1GB
- TinyLlama-1B: ~2GB
- Phi-2: ~6GB
- Llama2/Mistral-7B: ~14GB
- Add ~1GB for metric models (sentence-transformers, spaCy)

## Contributing

For those extending MERIT:

1. **New metrics**: Add to `merit/core/enhanced_metrics.py`
2. **New models**: Add adapter class to `merit/models/local_models.py`
3. **New datasets**: Add loading logic to `merit/experiments/robust_evaluation.py`
4. **Tests required**: All new features must include pytest tests

## References

**Semantic similarity**
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Evaluation methodologies**
- Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT

**AI Safety relevance**
- Bowman et al. (2022). Measuring Progress on Scalable Oversight for Large Language Models

---

**Version**: 2.0.0
**Author**: Tirth Patel
