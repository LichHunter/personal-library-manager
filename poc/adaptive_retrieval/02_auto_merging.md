# 02. Auto-Merging Retriever

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P0 - Uses Existing Data |
| **Complexity** | LOW |
| **Expected Improvement** | TBD (context sufficiency) |
| **PLM Changes Required** | Modify retrieve() logic |
| **External Dependencies** | None |

## Description

Automatically merge retrieved chunks into their parent (heading) when a majority of sibling chunks are retrieved. This indicates the query likely needs the full section.

## How It Works

```
1. Retrieve top-K chunks via hybrid search
2. Group chunks by heading_id
3. For each heading:
   - If (retrieved_chunks / total_chunks_in_heading) > threshold:
     - Replace individual chunks with full heading content
4. Return merged results
```

## Why It Works

- If 3 out of 4 chunks from a heading are retrieved, user probably needs the whole heading
- Chunks are for **finding**, headings are for **answering**
- Avoids fragmented context that confuses LLM

## PLM Advantage

PLM already stores:
- `heading_id` for every chunk
- Can query chunk counts per heading
- No re-indexing needed

## Algorithm

```python
def retrieve_with_auto_merge(query: str, k: int = 10, merge_threshold: float = 0.5):
    # 1. Standard retrieval
    chunks = hybrid_search(query, k=k)
    
    # 2. Group by heading
    heading_groups = defaultdict(list)
    for chunk in chunks:
        heading_groups[chunk.heading_id].append(chunk)
    
    # 3. Decide: merge or keep individual
    results = []
    for heading_id, chunk_list in heading_groups.items():
        total_in_heading = get_chunk_count(heading_id)
        ratio = len(chunk_list) / total_in_heading
        
        if ratio >= merge_threshold and total_in_heading > 1:
            # Merge: return full heading
            results.append(get_heading_content(heading_id))
        else:
            # Keep individual chunks
            results.extend(chunk_list)
    
    return results
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `merge_threshold` | Ratio to trigger merge | 0.5 |
| `min_siblings` | Min chunks in heading to consider merge | 2 |
| `max_heading_tokens` | Skip merge if heading too large | 2048 |

## Expected Behavior

| Scenario | Retrieved | Total in Heading | Action |
|----------|-----------|------------------|--------|
| Factoid query | 1 chunk | 4 | Keep chunk (25% < 50%) |
| Procedural query | 3 chunks | 4 | Merge to heading (75% > 50%) |
| Broad query | 2 chunks | 8 | Keep chunks (25% < 50%) |

## Tradeoffs

| Pros | Cons |
|------|------|
| Zero external dependencies | May over-merge sometimes |
| Uses existing PLM data | Needs heading content storage |
| Simple logic | Threshold tuning required |
| Improves context completeness | |

## Implementation Steps

1. Add heading chunk count query to storage
2. Add heading content retrieval function
3. Implement merge logic in retriever
4. Benchmark: chunks-only vs auto-merge vs always-heading

## References

- [LlamaIndex AutoMergingRetriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/)
- [Hierarchical Node Parser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#hierarchicalnodeparser)
