# 01. Reranking (Cross-Encoder)

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P0 - Easy Win |
| **Complexity** | LOW |
| **Expected Improvement** | +20-40% MRR |
| **PLM Changes Required** | Add post-processing step |
| **External Dependencies** | Cross-encoder model |

## Description

Add a cross-encoder reranking step after initial retrieval. Cross-encoders process query-document pairs jointly, enabling deeper semantic understanding than bi-encoders.

## How It Works

```
1. Retrieve top-50 candidates via hybrid search (existing PLM)
2. Score each (query, candidate) pair with cross-encoder
3. Re-sort by cross-encoder scores
4. Return top-10
```

## Why It Works

- **Bi-encoders** (embeddings): Encode query and doc separately → fast but shallow
- **Cross-encoders**: Encode query+doc together → slow but deep semantic matching
- Cross-encoders see word interactions that bi-encoders miss

## Model Options

| Model | Size | License | Latency (50 docs) |
|-------|------|---------|-------------------|
| BGE-reranker-v2-m3 | 560MB | Apache 2.0 | ~300ms (GPU) |
| Jina Reranker v2 | 560MB | Apache 2.0 | ~300ms (GPU) |
| ms-marco-MiniLM-L-6-v2 | 80MB | Apache 2.0 | ~100ms (GPU) |
| Cohere Rerank 4 | API | Proprietary | ~500ms |

## Algorithm

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, candidates: list[str], top_k: int = 10) -> list[int]:
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = scores.argsort()[::-1][:top_k]
    return ranked_indices
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `candidates_k` | How many to retrieve before reranking | 50 |
| `final_k` | How many to return after reranking | 10 |
| `model` | Which cross-encoder to use | BGE-reranker-v2-m3 |

## Expected Results

| Metric | Before | After | Source |
|--------|--------|-------|--------|
| MRR@10 | 0.60-0.70 | 0.75-0.85 | Literature avg |
| nDCG@10 | +20-30% relative | | MIT study |

## Tradeoffs

| Pros | Cons |
|------|------|
| Highest ROI improvement | Adds 200-500ms latency |
| Simple to implement | Requires GPU for speed |
| No retraining needed | Additional model to host |
| Works with any retriever | |

## Implementation Steps

1. Add `sentence-transformers` dependency
2. Create `Reranker` class
3. Integrate into `HybridRetriever.retrieve()`
4. Benchmark against baseline

## References

- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Cross-Encoders vs Bi-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Cohere Rerank](https://docs.cohere.com/docs/rerank)
