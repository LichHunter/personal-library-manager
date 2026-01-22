# Retrieval Strategy Benchmark - Final Results

**Date**: 2025-01-22  
**Goal**: Achieve 98%+ key facts coverage  
**Corpus**: 16 documents, 30 queries, 133 key facts, 17 chunks (fixed_512)

## Executive Summary

**Best Configuration: BGE-base embedder achieves 97.7% key facts coverage** (fuzzy matching) - reaching the theoretical maximum for this corpus.

| Strategy | Exact Match | Fuzzy Match | Avg Query Time |
|----------|-------------|-------------|----------------|
| **bge_base** | **94.0%** | **97.7%** | 5.8ms |
| hybrid_bge | 94.0% | 97.7% | 6.0ms |
| baseline_minilm | 93.2% | 97.0% | 3.7ms |
| hybrid_minilm | 93.2% | 97.0% | 3.6ms |
| rerank_minilm_top20 | 89.5% | 93.2% | 65.6ms |
| rerank_bge_top20 | 89.5% | 93.2% | 67.8ms |
| hybrid_rerank_minilm | 89.5% | 93.2% | 65.3ms |
| hybrid_rerank_bge | 89.5% | 93.2% | 67.4ms |

## Key Findings

### 1. BGE Embedder is the Best Upgrade (+0.8% exact, +0.7% fuzzy)
- `BAAI/bge-base-en-v1.5` outperforms `all-MiniLM-L6-v2`
- Only 1.6x slower (5.8ms vs 3.7ms) - negligible for RAG
- Achieves theoretical maximum (97.7%) with fuzzy matching

### 2. Hybrid BM25+Semantic Provides No Benefit
- Same results as pure semantic on this corpus
- Adds minimal overhead (~0.2ms)
- Likely because corpus is small and well-covered by semantic search

### 3. Cross-Encoder Reranking HURTS Performance
- Drops from 94.0% to 89.5% (exact) and 97.7% to 93.2% (fuzzy)
- With only 17 chunks, reranking shuffles already-good results poorly
- Cross-encoders optimized for passage ranking, not chunk retrieval
- **Avoid reranking with small chunk counts (<100)**

### 4. Theoretical Ceiling Reached
| Metric | Value |
|--------|-------|
| Total key facts | 133 |
| Exact match max (all docs) | 125 (94.0%) |
| Fuzzy match max (all docs) | 130 (97.7%) |
| BGE at k=5 (fuzzy) | 130 (97.7%) ✓ |

The remaining 3 facts (2.3%) are genuinely missing from the corpus.

## Recommended Configuration

```python
from sentence_transformers import SentenceTransformer

# Embedder
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

# Chunking
from strategies import FixedSizeStrategy
chunker = FixedSizeStrategy(chunk_size=512, overlap=0)

# Retrieval
k = 5  # Top-5 chunks

# Query encoding (BGE requires prefix)
def encode_query(query: str):
    return embedder.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    )

# Expected performance:
# - Exact string matching: 94.0%
# - Fuzzy matching (LLM behavior): 97.7%
```

## Why Not Higher Than 97.7%?

Three facts (2.3%) don't exist in any form in the corpus:
1. Facts using different terminology than documents
2. Implied information not explicitly stated
3. Ground truth errors

To reach 98%+, you would need:
1. Fix ground truth to match actual corpus content, OR
2. Add missing information to documents, OR
3. Use LLM-based fact verification instead of string matching

## Strategies to Avoid

| Strategy | Why Avoid |
|----------|-----------|
| Cross-encoder reranking | Hurts results with small chunk counts |
| Semantic chunking | 370x slower, worse coverage |
| Chunk overlap | No benefit in our tests |
| Smaller chunks | Dramatically worse coverage |

## Performance Characteristics

| Strategy | Index Time | Query Time | Memory |
|----------|------------|------------|--------|
| MiniLM semantic | Fast | 3.7ms | Low |
| BGE semantic | Medium | 5.8ms | Medium |
| Hybrid (any) | +BM25 build | +0.2ms | +BM25 index |
| Reranking (any) | Same | +60ms | +Model |

## Files

```
poc/chunking_benchmark_v2/
├── benchmark_retrieval_strategies.py  # This benchmark
└── results/
    ├── retrieval_benchmark.json       # Raw results
    └── RETRIEVAL_BENCHMARK_FINAL.md   # This report
```

## Next Steps (if 98%+ truly required)

1. **Expand test corpus** - More documents may show different patterns
2. **LLM-based fact verification** - Replace string matching with semantic similarity
3. **Query expansion** - Generate multiple query variants to catch edge cases
4. **Domain-specific fine-tuning** - If consistent terminology gaps exist
