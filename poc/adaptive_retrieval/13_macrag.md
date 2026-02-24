# 13. MacRAG (Multi-Scale Adaptive Context RAG)

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P3 |
| **Complexity** | HIGH |
| **Expected Improvement** | Comprehensive context handling |
| **PLM Changes Required** | Full pipeline redesign |
| **External Dependencies** | Compression + scaling models |

## Description

Combines multiple techniques: compress context, slice into manageable pieces, and scale up/down based on task complexity. End-to-end adaptive RAG pipeline.

## How It Works

```
1. COMPRESS: Reduce retrieved context to essential information
2. SLICE: Divide into logical segments (can process in parallel)
3. SCALE-UP: Aggregate results, expand context if needed
4. Iterate until sufficient
```

## Components

### 1. Compress

```python
def compress_stage(documents: list[str], query: str) -> str:
    # Remove irrelevant content
    # Keep query-relevant passages
    return compressed_context
```

### 2. Slice

```python
def slice_stage(context: str, max_slice_tokens: int = 2000) -> list[str]:
    # Divide into processable chunks
    # Maintain logical boundaries
    return slices
```

### 3. Scale-Up

```python
def scale_up_stage(slices: list[str], query: str) -> str:
    # Process each slice
    # Aggregate results
    # If insufficient, expand context
    partial_answers = [process(s, query) for s in slices]
    
    if needs_more_context(partial_answers):
        # Fetch more documents, repeat
        pass
    
    return aggregate(partial_answers)
```

## Full Algorithm

```python
def macrag_retrieve_and_answer(query: str):
    # Initial retrieval
    documents = retrieve(query, k=20)
    
    # Stage 1: Compress
    compressed = compress_stage(documents, query)
    
    # Stage 2: Slice
    slices = slice_stage(compressed)
    
    # Stage 3: Scale-up (with iteration)
    max_iterations = 3
    for i in range(max_iterations):
        partial_results = process_slices(slices, query)
        
        if is_sufficient(partial_results, query):
            return aggregate(partial_results)
        
        # Need more context - expand
        more_docs = retrieve_more(query, exclude=documents)
        documents.extend(more_docs)
        compressed = compress_stage(documents, query)
        slices = slice_stage(compressed)
    
    return aggregate(partial_results)  # Best effort
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `initial_k` | Initial retrieval size | 20 |
| `max_slice_tokens` | Tokens per slice | 2000 |
| `compression_ratio` | How much to compress | 0.5 |
| `max_iterations` | Scale-up iterations | 3 |

## Tradeoffs

| Pros | Cons |
|------|------|
| Handles long documents | Very complex pipeline |
| Multi-hop reasoning | Multiple LLM calls |
| Comprehensive approach | Hard to debug |
| Works for complex queries | May be overkill for simple queries |

## PLM Consideration

**Lowest priority** because:
- Extremely complex
- Multiple components to build
- Simpler approaches likely sufficient for K8s docs
- High latency and cost

Consider only if:
- Handling very long documents
- Multi-hop reasoning required
- All simpler approaches failed

## Relationship to Other Approaches

MacRAG is essentially a combination of:
- Multi-scale indexing (#5)
- Adaptive compression (#12)
- Iterative expansion (#4)
- Cluster selection (#11)

Better to implement individual components first.

## Implementation Steps

1. Implement compression (see #12)
2. Implement slicing logic
3. Implement scale-up iteration
4. Integrate components
5. Extensive testing and tuning

## References

- [MacRAG Paper](https://arxiv.org/abs/2505.06569)
- Builds on: RAPTOR, LongRAG, RECOMP
