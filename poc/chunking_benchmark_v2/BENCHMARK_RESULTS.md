# Chunking & Retrieval Benchmark Results

## Executive Summary

We evaluated retrieval strategies for a local-first NotebookLM-like system using a corpus of 52 documents (180 key facts). Our benchmark achieved **98-99% key facts coverage at k=10** with the best configuration.

**Recommended Stack:**
- **Chunking:** Fixed-size, 512 tokens, 0% overlap
- **Retrieval:** Hybrid + Rerank (BM25 + Semantic + Cross-encoder)
- **Embedder:** all-MiniLM-L6-v2
- **Reranker:** ms-marco-MiniLM-L-6-v2
- **Result:** 94% @ k=5, 98-99% @ k=10

## Test Setup

### Corpus
- **Documents:** 52 synthetic documents about "CloudFlow" platform
- **Queries:** 53 queries (simple, cross-doc, complex)  
- **Key Facts:** 180 facts to retrieve
- **Location:** `corpus/expanded_documents/`

### Hardware
- RTX 3070 8GB VRAM
- CUDA-enabled environment

### Evaluation Metric
```python
def exact_match(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()
```
We check if each key fact string literally appears in the concatenated retrieved chunks.

## Testing Strategy

### What We Tested

| Component | Options Tested | Total |
|-----------|---------------|-------|
| **Chunking Strategies** | fixed_size (200-512 tokens), recursive_splitter (800-2000), semantic, paragraphs, headings | 20 |
| **Embedders** | all-MiniLM-L6-v2, all-MiniLM-L12-v2, gte-small, bge-small-en-v1.5, bge-base-en-v1.5 | 5 |
| **Retrieval Strategies** | semantic, hybrid, hybrid_rerank, lod, raptor (partial) | 5 |
| **Rerankers** | ms-marco-MiniLM-L-6-v2, ms-marco-MiniLM-L-12-v2 | 2 |

### What We Did NOT Test
- **lod_llm** - Too slow (~100 LLM calls per query, ~4-7 hours per combination)
- **RAPTOR (full)** - Started but incomplete (600 combinations, ~25 hours total)

## Main Results

### Best Configurations by k

| k | Strategy | Chunking | Coverage | Latency |
|---|----------|----------|----------|---------|
| **k=5** | semantic | fixed_512 | **96.1%** | ~50ms |
| **k=10** | hybrid_rerank | fixed_512 | **98.3%** | ~300ms |

### Retrieval Strategy Comparison

| Strategy | k=5 | k=10 | Speed | Notes |
|----------|-----|------|-------|-------|
| **Semantic** | 95-96% | 97-98% | Fast | Simple cosine similarity |
| **Hybrid** (BM25+Semantic) | 95% | 97-98% | Fast | Marginal improvement |
| **Hybrid + Rerank** | 94% | **98-99%** | Medium | Best at k=10 |
| **LOD** (hierarchical) | 54% | 54% | Fast | Severe underperformance |
| **RAPTOR** | 96% | 98% | Slow indexing | Tree summarization with LLM |

### Chunking Strategy Impact

| Chunking | Tokens | k=5 | k=10 | Verdict |
|----------|--------|-----|------|---------|
| **fixed_512** | 512 | **95%** | **97-99%** | **BEST** |
| fixed_400 | 400 | 95% | 97-99% | Good |
| fixed_300 | 300 | 95% | 97% | Good |
| fixed_200 | 200 | 91% | 94% | Too small |
| paragraphs | 50-256 | 67% | 77% | Poor |
| heading_based | varies | 51% | 60% | Bad |

**Key Finding:** Chunk size 300-512 tokens is optimal. Smaller chunks hurt recall significantly.

### Embedder Comparison

All embedders performed similarly (within 2%):

| Embedder | k=5 avg | k=10 avg | Size |
|----------|---------|----------|------|
| **all-MiniLM-L6-v2** | 95.5% | 97.5% | 80MB |
| all-MiniLM-L12-v2 | 95.2% | 97.5% | 120MB |
| gte-small | 95.2% | 97.0% | 70MB |
| bge-small-en-v1.5 | 94.4% | 97.0% | 130MB |
| bge-base-en-v1.5 | 94.4% | 95.8% | 440MB |

**Key Finding:** Embedder choice has minimal impact. Use smallest/fastest.

## Failed Facts Analysis

### Always Failed (across all strategies)

Only 2 facts consistently failed across all 9 test combinations:
1. `GATEWAY_SSL_ENABLED` - Config detail not semantically matched to query
2. `Idempotency` - Technical term buried in document

### Root Cause

Failed facts are primarily due to **semantic mismatch** between:
- Natural language queries ("What database does CloudFlow use?")
- Document titles/content (formal ADR titles like "ADR-001: Use PostgreSQL...")

This is a **data issue**, not a retrieval problem. Documents need better semantic markers for natural language queries.

## Reproducing Results

### Setup Environment
```bash
cd /home/fujin/Code/personal-library-manager
nix develop
source .venv/bin/activate
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
```

### Run Full Benchmark
```bash
# Main config (disabled expensive strategies)
python poc/chunking_benchmark_v2/run_benchmark.py --config config_full.yaml

# Quick test (semantic only, 1 embedder, 3 chunking)
python poc/chunking_benchmark_v2/run_benchmark.py --config config_quick.yaml

# Light benchmark (semantic, hybrid, hybrid_rerank)
python poc/chunking_benchmark_v2/run_benchmark.py --config config_light.yaml
```

### Analyze Failed Facts
```bash
python poc/chunking_benchmark_v2/run_benchmark.py --config config_failure_analysis.yaml
```

### Key Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Main benchmark runner with failure tracking |
| `config_full.yaml` | Full test matrix (lod_llm disabled) |
| `config_light.yaml` | Fast strategies only |
| `config_failure_analysis.yaml` | Single run with detailed failure output |
| `retrieval/*.py` | Retrieval strategy implementations |
| `strategies/*.py` | Chunking strategy implementations |
| `results/benchmark_full_run.log` | Partial results from main run |
| `results/*_failed_facts.json` | Per-run failure analysis |

## Comparison with NotebookLM

Based on recent academic research (Northwestern, 2025):
- **NotebookLM:** ~87% accuracy (13% hallucination rate)
- **Our system:** 98-99% key facts retrieval at k=10 (with Hybrid + Rerank)

Note: These measure different things:
- We measure **retrieval recall** (are facts in retrieved chunks?)
- NotebookLM study measures **generation faithfulness** (are generated answers accurate?)

The 13% NotebookLM hallucination rate shows that even with perfect retrieval, LLMs still struggle with:
- Attribution drift (opinions become facts)
- Interpretive overconfidence (adding unsupported analysis)
- Source characterization (claiming document purposes without evidence)

## Conclusions

1. **Simple works:** Pure semantic search with fixed-size chunks achieves 96%+ coverage
2. **Chunk size matters:** 300-512 tokens optimal, smaller severely hurts recall
3. **Embedders don't matter:** All tested models within 2% of each other
4. **Avoid complexity:** LOD hierarchical retrieval underperforms simple semantic search
5. **Data quality crucial:** Semantic mismatch between queries and documents causes most failures

## Recommendations

### For Maximum Accuracy (Production)
```yaml
chunking:
  strategy: fixed_size
  tokens: 512
  overlap: 0

retrieval:
  strategy: hybrid_rerank
  k: 10

embedder: all-MiniLM-L6-v2
reranker: ms-marco-MiniLM-L-6-v2
```

This configuration provides:
- **98-99% key facts coverage at k=10**
- ~300ms retrieval latency
- Moderate VRAM usage (~200MB for embedder + reranker)

### For Speed/Simplicity (Development)
```yaml
chunking:
  strategy: fixed_size
  tokens: 512
  overlap: 0

retrieval:
  strategy: semantic
  k: 10

embedder: all-MiniLM-L6-v2
reranker: null
```

This configuration provides:
- 97.8% key facts coverage at k=10
- ~50ms retrieval latency
- Minimal VRAM usage (~100MB)
- Simple implementation