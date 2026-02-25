# 05. Multi-Scale Indexing

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P2 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | +1-37% (AI21 benchmarks) |
| **PLM Changes Required** | Multiple indices + RRF aggregation |
| **External Dependencies** | None (more storage) |

## Description

Index the same corpus at multiple chunk sizes simultaneously. At query time, search all indices in parallel and aggregate results using Reciprocal Rank Fusion (RRF).

## How It Works

```
Indexing Phase:
1. Chunk corpus at sizes: 128, 256, 512, 1024, 2048 tokens
2. Create separate embedding index for each size
3. Store all indices

Query Phase:
1. Query ALL indices in parallel
2. Each index returns ranked results
3. Aggregate rankings using RRF
4. Return fused top-K
```

## Why It Works

- Different queries need different chunk sizes
- Oracle experiments show 20-40% headroom when selecting optimal size per query
- RRF approximates oracle selection without explicit prediction
- Documents that appear across multiple scales are likely relevant

## AI21 Research Results

| Dataset | Best Single Scale | Multi-Scale + RRF | Improvement |
|---------|-------------------|-------------------|-------------|
| QMSum | 0.72 | 0.78 | +8% |
| NarrativeQA | 0.65 | 0.89 | +37% |
| Seinfeld | 0.81 | 0.86 | +6% |
| MTEB (avg) | varies | +1-3% | consistent |

## Algorithm

### RRF Aggregation

```python
def rrf_aggregate(rankings: dict[str, list[str]], k: int = 60) -> list[str]:
    """
    rankings: {chunk_size: [doc_id1, doc_id2, ...]}
    k: RRF constant (default 60)
    """
    scores = defaultdict(float)
    
    for chunk_size, ranked_docs in rankings.items():
        for rank, doc_id in enumerate(ranked_docs):
            scores[doc_id] += 1 / (k + rank + 1)
    
    # Sort by score descending
    sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return sorted_docs
```

### Full Pipeline

```python
def multi_scale_retrieve(query: str, indices: dict[str, Index], k: int = 10):
    # 1. Query all indices in parallel
    rankings = {}
    for chunk_size, index in indices.items():
        results = index.search(query, k=50)
        rankings[chunk_size] = [r.doc_id for r in results]
    
    # 2. Aggregate with RRF
    fused_doc_ids = rrf_aggregate(rankings)[:k]
    
    # 3. Return results (could be chunks or documents)
    return fused_doc_ids
```

## Storage Analysis

| Configuration | Chunk Sizes | Storage Multiplier |
|---------------|-------------|-------------------|
| Minimal | 256, 1024 | 2x |
| Balanced | 128, 512, 2048 | 3x |
| Full | 128, 256, 512, 1024, 2048 | 5x |

For PLM (~20K chunks at 512 tokens):
- Current: ~20K embeddings (~150MB FAISS)
- With 3 scales: ~60K embeddings (~450MB)
- With 5 scales: ~100K embeddings (~750MB)

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `chunk_sizes` | Sizes to index | [128, 512, 2048] |
| `rrf_k` | RRF constant | 60 |
| `retrieve_k` | Results per index | 50 |
| `final_k` | Final results | 10 |

## Tradeoffs

| Pros | Cons |
|------|------|
| No per-query prediction needed | 2-5x storage overhead |
| Proven improvements (AI21) | More embedding compute at index time |
| Parallel queries (low latency) | More complex index management |
| Works with any embedding model | |

## Implementation Steps

1. Create multi-scale chunking pipeline
2. Build separate index per scale
3. Implement parallel query across indices
4. Implement RRF aggregation
5. Benchmark against single-scale baseline
6. Tune: which scales provide best tradeoff?

## Open Questions

- Do we need all 5 scales, or is 2-3 sufficient?
- Should we weight scales differently in RRF?
- Return chunks or aggregate to documents?

## References

- [AI21 Blog: Query-Dependent Chunking](https://www.ai21.com/blog/query-dependent-chunking/)
- [GitHub: AI21Labs/multi-window-chunk-size](https://github.com/AI21Labs/multi-window-chunk-size)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
