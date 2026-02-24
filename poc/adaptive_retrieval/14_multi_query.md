# 14. Multi-Query Retrieval

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P2 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | +20-40% recall (with reranking) |
| **PLM Changes Required** | Query expansion + merge logic |
| **External Dependencies** | LLM for query generation |

## Description

Generate multiple query variants using LLM, retrieve for each variant, merge and deduplicate results. Captures different perspectives of the same question.

## How It Works

```
1. Original query → LLM generates 3-5 variants
2. Retrieve top-K for EACH variant
3. Merge all results (union)
4. Deduplicate by document ID
5. (Optional) Rerank merged results
6. Return top-K
```

## Why It Works

- Single query embedding may miss relevant documents
- Different phrasings match different docs
- LLM generates semantically equivalent but lexically different queries
- Union increases recall

## Difference from Query Expansion (PLM Already Has)

| Aspect | Query Expansion | Multi-Query |
|--------|-----------------|-------------|
| Method | Add synonyms/terms | Generate new questions |
| Depth | Lexical variation | Semantic variation |
| Cost | ~0 (local rules) | 1 LLM call |
| Example | "deploy" → "deploy kubernetes install" | "How to deploy?" → ["Kubernetes deployment steps", "Install app on K8s", "Deploy container to cluster"] |

## Algorithm

```python
MULTI_QUERY_PROMPT = """Generate 3 different versions of this question 
to help retrieve relevant documents. Provide different perspectives.

Original: {query}

Variants (one per line):"""

def multi_query_retrieve(query: str, k: int = 10):
    # 1. Generate variants
    response = llm.generate(MULTI_QUERY_PROMPT.format(query=query))
    variants = response.strip().split("\n")
    variants = [v.strip() for v in variants if v.strip()]
    
    # 2. Include original
    all_queries = [query] + variants
    
    # 3. Retrieve for each
    all_results = []
    seen_ids = set()
    
    for q in all_queries:
        results = retrieve(q, k=k)
        for r in results:
            if r.doc_id not in seen_ids:
                all_results.append(r)
                seen_ids.add(r.doc_id)
    
    # 4. Sort by best score across queries
    all_results.sort(key=lambda r: r.score, reverse=True)
    
    # 5. Return top-k
    return all_results[:k]
```

## Example Transformation

```
Original: "How do I make my pods restart automatically?"

Generated Variants:
1. "Kubernetes pod restart policy configuration"
2. "Liveness probe setup for automatic container restart"
3. "Configure restartPolicy in pod spec"
4. "Health check to restart failing pods"
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `n_variants` | Number of variants to generate | 3-5 |
| `per_query_k` | Results per variant | 10 |
| `include_original` | Include original query | True |
| `merge_method` | How to combine results | union + sort |

## Tradeoffs

| Pros | Cons |
|------|------|
| Captures multiple perspectives | LLM call adds latency (~200-500ms) |
| Improves recall | LLM cost per query |
| Works with any retriever | May retrieve too much (need reranking) |
| No training needed | Generated queries may drift |

## PLM Consideration

Medium priority because:
- PLM already has query expansion
- Adds LLM latency per query
- But: significantly different from term expansion
- Consider combining with reranking (#1)

## Implementation Steps

1. Create query variant generation prompt
2. Implement multi-query retrieval
3. Implement deduplication + merge
4. Add reranking (recommended)
5. Benchmark: single-query vs multi-query

## Metrics

| Metric | Description |
|--------|-------------|
| Recall improvement | Docs found with multi vs single |
| Unique doc rate | How many new docs from variants |
| Latency impact | Time added for LLM + extra retrievals |

## References

- [LangChain MultiQueryRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- [Query Rewriting in RAG](https://blog.langchain.dev/query-transformations/)
