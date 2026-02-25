# 04. Iterative Expansion

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P1 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | Optimal context per query |
| **PLM Changes Required** | Add expansion logic + sufficiency check |
| **External Dependencies** | Optional LLM for sufficiency check |

## Description

Start with minimal context (chunks), check if sufficient to answer the query, and progressively expand to larger units (headings, documents) until sufficient.

## How It Works

```
1. Retrieve chunks (smallest granularity)
2. Check: Is this sufficient to answer the query?
   - If YES → Return chunks
   - If NO → Continue
3. Expand to headings (get full heading for retrieved chunks)
4. Check sufficiency again
   - If YES → Return headings
   - If NO → Continue
5. Expand to documents
6. Return documents
```

## Why It Works

- Avoids over-retrieval for simple queries
- Ensures complex queries get enough context
- Adapts per-query without pre-classification

## Sufficiency Check Options

### Option A: Heuristic-Based (Fast, No LLM)

```python
def is_sufficient_heuristic(query: str, context: list[str]) -> bool:
    # Extract key terms from query
    query_terms = extract_keywords(query)  # Simple: split + remove stopwords
    
    # Check coverage
    context_text = " ".join(context).lower()
    covered = sum(1 for term in query_terms if term.lower() in context_text)
    coverage_ratio = covered / len(query_terms) if query_terms else 0
    
    # Check minimum size
    total_tokens = sum(len(c.split()) for c in context)
    
    return coverage_ratio > 0.7 and total_tokens > 150
```

### Option B: LLM-Based (Accurate, Slower)

```python
def is_sufficient_llm(query: str, context: list[str]) -> bool:
    context_text = "\n---\n".join(context)
    
    prompt = f"""Given ONLY this context:
{context_text}

Can you fully and accurately answer this question: {query}

Reply with ONLY: YES or NO"""
    
    response = llm.generate(prompt, max_tokens=5)
    return "YES" in response.upper()
```

### Option C: Embedding Similarity (Medium Speed/Accuracy)

```python
def is_sufficient_embedding(query: str, context: list[str]) -> bool:
    query_emb = embed(query)
    context_emb = embed(" ".join(context))
    
    similarity = cosine_similarity(query_emb, context_emb)
    return similarity > 0.75
```

## Algorithm

```python
def iterative_retrieve(query: str, k: int = 10, max_iterations: int = 2):
    # Level 1: Chunks
    chunks = retrieve_chunks(query, k=k)
    if is_sufficient(query, [c.content for c in chunks]):
        return chunks, "chunk"
    
    # Level 2: Headings
    heading_ids = set(c.heading_id for c in chunks)
    headings = [get_heading_content(hid) for hid in heading_ids]
    if is_sufficient(query, headings):
        return headings, "heading"
    
    # Level 3: Documents
    doc_ids = set(c.doc_id for c in chunks)
    documents = [get_document_content(did) for did in doc_ids]
    return documents, "document"
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `sufficiency_method` | Which check to use | heuristic (start) |
| `coverage_threshold` | Keyword coverage ratio | 0.7 |
| `min_tokens` | Minimum context size | 150 |
| `max_iterations` | Maximum expansion steps | 2 |

## Expected Behavior

| Query Type | Level 1 (Chunks) | Level 2 (Headings) | Final |
|------------|------------------|--------------------| ------|
| "What port does X use?" | Sufficient | - | Chunk |
| "How do I deploy X?" | Insufficient | Sufficient | Heading |
| "Explain X architecture" | Insufficient | Insufficient | Document |

## Tradeoffs

| Pros | Cons |
|------|------|
| Adapts per-query | Multiple retrieval rounds possible |
| No pre-classification needed | Sufficiency check adds latency |
| Always finds sufficient context | Heuristics may be inaccurate |

## Implementation Steps

1. Implement heuristic sufficiency check
2. Add heading content retrieval
3. Add document content retrieval
4. Implement expansion loop
5. Benchmark: track expansion rates per query type
6. (Optional) Add LLM sufficiency check for edge cases

## Monitoring

Track these metrics to understand behavior:
- Expansion rate: % of queries that expand beyond chunks
- Average final level: chunk / heading / document distribution
- Sufficiency check accuracy (if ground truth available)

## References

- [CRAG - Corrective RAG](https://arxiv.org/abs/2401.15884) (similar iterative concept)
- [Self-RAG](https://arxiv.org/abs/2310.11511) (LLM-based sufficiency)
