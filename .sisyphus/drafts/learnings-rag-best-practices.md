# Learnings: RAG Best Practices (2024-2026)

## The 5 Things That Actually Matter

### 1. Hybrid Search (BM25 + Vector + RRF)
- **30-40% better accuracy** than either alone
- Production standard in 2026
```python
def rrf(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 2. Cross-Encoder Reranking
- **14-48% accuracy improvement**
- Two-stage: Fast retrieval → Slow reranking
- Models: `bge-reranker-v2-m3`, `ms-marco-MiniLM-L-12-v2`

### 3. Query Classification
- Skip retrieval for simple queries
- **30-40% cost reduction**
- Classify: Simple/Medium/Complex → Different strategies

### 4. Systematic Evaluation
- Build test set (100+ queries minimum)
- Metrics: Precision@10, Faithfulness, NDCG
- Monitor continuously

### 5. User Feedback Loop
- Thumbs up/down on every answer
- Catch silent degradation
- Use for continuous improvement

## Chunking Best Practices

### Strategy by Document Type
| Type | Strategy | Size | Overlap |
|------|----------|------|---------|
| Technical docs | Semantic | Variable | 10-15% |
| Narrative | Recursive | 512-1024 | 50-100 |
| Structured | By section | Variable | None |
| Short docs | Fixed | 256-512 | 50 |

### Multi-Scale: RAPTOR
- Leaf nodes: Original chunks
- Level 1: Cluster summaries
- Level 2: Summary of summaries
- **65.5% accuracy** on narrative understanding

### Practical Rules
1. Start simple: Fixed 512 + 50 overlap
2. Don't split mid-sentence, code blocks, tables
3. Add metadata: doc_id, heading, page, chunk_index
4. Test on YOUR data

## Query Understanding

### Query Expansion Techniques
1. **HyDE**: Generate hypothetical answer, embed that
2. **Multi-Query**: Generate 3-5 variations, merge with RRF
3. **Decomposition**: Break into sub-queries
4. **Step-Back**: Generate broader version

### Caution
Query expansion increases latency/cost. Use selectively based on query classification.

## Academic Research Highlights

### Adaptive-RAG (2024)
- Dynamically select strategy based on query complexity
- Simple → No retrieval, Complex → Multi-step

### CRAG (2024)
- Self-correction: Evaluate retrieval quality
- Actions: Use/Ambiguous→Web search/Incorrect→Decompose

### Self-RAG (2024)
- Reflection tokens: [Retrieval], [IsRel], [IsSup], [IsUse]
- Model controls its own behavior

## Production Optimization

### Semantic Caching
```python
def cache_lookup(query, threshold=0.95):
    embedding = embed(query)
    similar = vector_db.search(embedding, top_k=1)
    if similar[0].score > threshold:
        return cache[similar[0].id]
    return None
```
- **Up to 68.8% LLM cost reduction**

### Adaptive Retrieval (CAR)
- Dynamic k based on query
- Narrow queries: k=3
- Broad queries: k=20
- Detect transition point in similarity scores

## Common Pitfalls

1. ❌ Retrieval without evaluation
2. ❌ Over-engineering early
3. ❌ Ignoring chunking (one size doesn't fit all)
4. ❌ No reranking
5. ❌ Static top-k
6. ❌ No query classification
7. ❌ Ignoring cost
8. ❌ No user feedback

## Implementation Timeline

### MVP (Week 1-2)
- Fixed chunking, hybrid search, basic RRF
- Manual testing on 20 queries

### Production v1 (Month 1)
- Cross-encoder reranking
- Query classification
- Test set (100 queries)
- Basic metrics

### Production v2 (Month 2-3)
- Semantic caching
- Adaptive retrieval
- Query expansion
- A/B testing

### Advanced (Month 4+)
- Hierarchical chunking
- Self-correction (CRAG)
- Continuous evaluation

---
*Sources: Adaptive-RAG, CRAG, Self-RAG, RAPTOR papers + production guides*
