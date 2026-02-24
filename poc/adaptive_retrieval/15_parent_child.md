# 15. Parent-Child Retrieval

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P0 - Foundational Pattern |
| **Complexity** | LOW |
| **Expected Improvement** | +4-70% context quality |
| **PLM Changes Required** | Use existing heading_id/doc_id |
| **External Dependencies** | None |

## Description

Index and search on small chunks (children) for precise matching, but return larger parent units (headings/documents) for complete context.

## How It Works

```
1. Indexing: Store small chunks with parent references
2. Search: Query matches small chunks
3. Return: Fetch and return parent content

Chunk → heading_id → Full Heading Content
```

## Why It Works

The fundamental RAG paradox:
- **Small chunks** = precise embeddings, better matching
- **Large chunks** = complete context, better answering

Parent-child gives you BOTH:
- Search precision from children
- Answer quality from parents

## PLM Already Has This Data

```sql
-- PLM chunks table has:
chunk_id, content, heading_id, doc_id, ...

-- We can:
1. Search chunks (current behavior)
2. Look up heading_id
3. Return full heading content (NEW)
```

## Algorithm

```python
def parent_child_retrieve(query: str, k: int = 10, return_level: str = "heading"):
    # 1. Search children (chunks)
    chunks = hybrid_search(query, k=k*2)  # Over-retrieve
    
    # 2. Get unique parent IDs (preserve order)
    if return_level == "heading":
        parent_ids = list(dict.fromkeys(c.heading_id for c in chunks))
    else:  # document
        parent_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    
    # 3. Fetch parent content
    parents = []
    for pid in parent_ids[:k]:
        content = get_parent_content(pid, level=return_level)
        parents.append(content)
    
    return parents
```

## Implementation for PLM

```python
# Need to add to PLM storage:

def get_heading_content(heading_id: str) -> str:
    """Fetch all chunks for a heading, concatenate."""
    chunks = db.query(
        "SELECT content FROM chunks WHERE heading_id = ? ORDER BY position",
        [heading_id]
    )
    return "\n".join(c.content for c in chunks)

def get_document_content(doc_id: str) -> str:
    """Fetch all chunks for a document, concatenate."""
    chunks = db.query(
        "SELECT content FROM chunks WHERE doc_id = ? ORDER BY position",
        [doc_id]
    )
    return "\n".join(c.content for c in chunks)
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `return_level` | heading or document | heading |
| `search_k` | Chunks to search | 20 |
| `return_k` | Parents to return | 5-10 |
| `dedup_children` | Skip if same parent | True |

## Comparison with Related Approaches

| Approach | Trigger for Parent | Behavior |
|----------|-------------------|----------|
| **Parent-Child** | Always | Return parent for any matched child |
| **Auto-Merge** | Majority of siblings | Return parent only if threshold met |
| **Recursive** | Reference following | Follow links to parent |

Parent-child is the simplest: always return parent level.

## When to Use Document vs Heading

| Return Level | When | Example |
|--------------|------|---------|
| **Heading** | Most queries | "How to deploy StatefulSet" |
| **Document** | Broad topics | "Overview of Kubernetes networking" |
| **Chunk** | Factoid | "Default port for Redis" |

Default to heading - it's the sweet spot.

## Tradeoffs

| Pros | Cons |
|------|------|
| Simple implementation | May over-retrieve for factoids |
| Uses existing PLM data | Needs parent content reconstruction |
| Proven pattern | Fixed behavior (not adaptive) |
| No training needed | |

## Implementation Steps

1. Add `get_heading_content()` to storage
2. Add `get_document_content()` to storage
3. Modify `retrieve()` to return parent level
4. Benchmark: chunks vs parents

## Metrics

| Metric | Description |
|--------|-------------|
| Context completeness | % of queries answerable with returned context |
| Token efficiency | Answer quality / tokens returned |
| Retrieval accuracy | Still finding relevant content? |

## References

- [LangChain ParentDocumentRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
- [LlamaIndex Small-to-Big Retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/)
