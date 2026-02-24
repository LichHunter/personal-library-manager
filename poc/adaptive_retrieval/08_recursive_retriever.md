# 08. Recursive Retriever

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P1 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | Better context via references |
| **PLM Changes Required** | Add reference following logic |
| **External Dependencies** | None |

## Description

Retrieve on small chunks or summaries, then follow references/links to fetch larger parent content. Chunks contain pointers to their source sections.

## How It Works

```
1. Search on small chunks (or summaries)
2. Each chunk has reference to parent (heading_id, doc_id)
3. Follow references to fetch parent content
4. Return parent content (not the small chunks)
```

## Why It Works

- Small chunks/summaries are better for matching (focused embeddings)
- Parent content is better for answering (complete context)
- References enable this decoupling

## Difference from Auto-Merge

| Aspect | Auto-Merge | Recursive |
|--------|------------|-----------|
| Trigger | Majority of siblings retrieved | Any chunk retrieved |
| Logic | Count-based threshold | Always follow reference |
| Granularity | Merge if threshold met | Always return parent |

## Algorithm

```python
def recursive_retrieve(query: str, k: int = 10):
    # 1. Search on chunks
    chunks = search_chunks(query, k=k)
    
    # 2. Collect unique parent references
    parent_refs = []
    seen = set()
    for chunk in chunks:
        ref = chunk.heading_id  # or doc_id
        if ref not in seen:
            parent_refs.append(ref)
            seen.add(ref)
    
    # 3. Fetch parent content
    parents = [get_heading_content(ref) for ref in parent_refs[:k]]
    
    return parents
```

## Variants

### A. Chunk → Heading

```python
# Always return full heading for any matched chunk
parents = [get_heading(chunk.heading_id) for chunk in chunks]
```

### B. Summary → Document

```python
# Search on document summaries, return full documents
summaries = search_summaries(query, k=k)
documents = [get_document(s.doc_id) for s in summaries]
```

### C. Multi-Level

```python
# Search on smallest unit, return configurable level up
def recursive_retrieve(query, k, return_level="heading"):
    chunks = search_chunks(query, k=k)
    
    if return_level == "heading":
        return [get_heading(c.heading_id) for c in chunks]
    elif return_level == "document":
        return [get_document(c.doc_id) for c in chunks]
```

## PLM Implementation

PLM already has the data structure:
- Chunks with `heading_id` and `doc_id`
- Can implement immediately

```python
def plm_recursive_retrieve(query: str, k: int = 10):
    # Existing hybrid search
    chunks = hybrid_retriever.retrieve(query, k=k)
    
    # Get unique headings
    heading_ids = list(dict.fromkeys(c.heading_id for c in chunks))
    
    # Fetch heading content (need to implement this query)
    headings = storage.get_headings_by_ids(heading_ids[:k])
    
    return headings
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `search_k` | Chunks to search | 20-50 |
| `return_level` | heading or document | heading |
| `max_parents` | Maximum parents to return | 10 |

## Tradeoffs

| Pros | Cons |
|------|------|
| Simple logic | Always expands (may over-retrieve) |
| Uses existing PLM data | Needs parent content storage |
| Predictable behavior | No adaptation per query |
| Fast (single search + lookups) | |

## Implementation Steps

1. Add `get_heading_content(heading_id)` to storage
2. Add `get_document_content(doc_id)` to storage
3. Implement recursive retrieval wrapper
4. Benchmark: chunk-only vs recursive

## References

- [LlamaIndex RecursiveRetriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/)
- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
