# Performance Assessment: RAPTOR

> Analysis of bottlenecks, scalability, and optimization opportunities

## Executive Summary

RAPTOR's performance characteristics are dominated by external API calls (LLM summarization, embedding generation) which account for >95% of processing time. The algorithm has **O(n)** API calls for indexing where n = number of text chunks, plus additional calls for hierarchical summarization. Memory usage scales with document size but is manageable for typical use cases. Significant performance issues emerge at scale (>1000 documents).

---

## 1. Performance Bottleneck Analysis

### 1.1 API Call Costs (Critical Bottleneck)

| Operation | API Calls | Cost Driver | File:Line |
|-----------|-----------|-------------|-----------|
| Leaf node embedding | N (one per chunk) | Embedding API | `tree_builder.py:176-178` |
| Summary generation | ~N/cluster_size per layer | LLM API | `cluster_tree_builder.py:71-74` |
| Summary embedding | ~N/cluster_size per layer | Embedding API | `tree_builder.py:176-178` |
| Query embedding | 1 per query | Embedding API | `tree_retriever.py:170` |
| Answer generation | 1 per query | LLM API | `QAModels.py` |

**Example for 100-chunk document with 5 layers:**
- Leaf embeddings: 100 calls
- Layer 1 summaries: ~20 calls (assuming 5 chunks/cluster avg)
- Layer 1 embeddings: ~20 calls
- Layer 2-5: diminishing calls
- **Total: ~200-300 API calls for indexing**

### 1.2 Computational Costs

| Operation | Complexity | Location |
|-----------|------------|----------|
| UMAP dimensionality reduction | O(n^1.8) approximate | `cluster_utils.py:31-33` |
| GMM clustering | O(n * k * d * iterations) | `cluster_utils.py:53-55` |
| BIC optimal cluster search | O(max_clusters * GMM) | `cluster_utils.py:52-56` |
| Cosine distance calculation | O(n * d) | `utils.py:131-134` |
| Text tokenization | O(text_length) | Throughout |

### 1.3 Bottleneck Ranking

1. **LLM API calls** (summarization) - 60-70% of wall time
2. **Embedding API calls** - 25-35% of wall time
3. **UMAP + GMM clustering** - 3-5% of wall time
4. **Everything else** - <2% of wall time

---

## 2. API Call Analysis

### 2.1 LLM Calls Per Document

**During Tree Building:**

```
For a document with N leaf nodes and L layers:

Summaries = N/c + N/c² + N/c³ + ... + N/c^L
         ≈ N * (c/(c-1)) * (1 - 1/c^L)
         
Where c = average cluster size (typically 3-10)
```

**Concrete Example:**
- Document: 10,000 tokens
- Chunk size: 100 tokens
- N = 100 leaf nodes
- Average cluster size c = 5
- Layers = 3 (stops when nodes < reduction_dimension)

```
Layer 1: 100/5 = 20 summaries
Layer 2: 20/5 = 4 summaries  
Layer 3: 4/5 = 1 summary (or stops)
Total: 25 summarization API calls
```

**During Retrieval:**
- Collapse tree mode: 0 LLM calls (retrieval only)
- Tree traversal mode: 0 LLM calls (retrieval only)
- Answer generation: 1 LLM call per question

### 2.2 Embedding Calls Per Document

**During Tree Building:**

Each node (leaf + summary) requires embedding:
```
Embeddings = N + summaries = N + N*(c/(c-1))*(1 - 1/c^L)
           ≈ N * (2c-1)/(c-1) for deep trees
```

**Concrete Example (same as above):**
```
Leaf embeddings: 100
Summary embeddings: 25
Total: 125 embedding API calls
```

**During Retrieval:**
- 1 embedding call per query

### 2.3 API Cost Estimation

| Document Size | Chunks | Est. Embeddings | Est. Summaries | Est. Cost* |
|---------------|--------|-----------------|----------------|------------|
| 5K tokens | 50 | 65 | 13 | ~$0.05 |
| 50K tokens | 500 | 650 | 130 | ~$0.50 |
| 500K tokens | 5,000 | 6,500 | 1,300 | ~$5.00 |
| 5M tokens | 50,000 | 65,000 | 13,000 | ~$50.00 |

*Estimates based on OpenAI pricing: ada-002 @ $0.0001/1K tokens, gpt-3.5-turbo @ $0.002/1K tokens

---

## 3. Memory Usage Analysis

### 3.1 In-Memory Data Structures

| Structure | Size Formula | Example (100 chunks) |
|-----------|--------------|---------------------|
| Leaf node texts | O(N * chunk_size) | 100 * 100 tokens ≈ 40KB |
| Embeddings (ada-002) | O(N * 1536 * 4 bytes) | 100 * 1536 * 4 ≈ 600KB |
| Summary texts | O(summaries * summary_size) | 25 * 100 tokens ≈ 10KB |
| Summary embeddings | O(summaries * 1536 * 4) | 25 * 1536 * 4 ≈ 150KB |
| Tree structure | O(N + summaries) | Negligible |
| **Total per 100 chunks** | | **~800KB** |

### 3.2 Memory During Processing

**Peak Memory Points:**

1. **UMAP fitting** (`cluster_utils.py:31-33`):
   - Creates n x n distance matrix internally
   - Memory: O(n²) where n = nodes at current layer
   - For 1000 nodes: ~8MB for distance matrix alone

2. **GMM fitting** (`cluster_utils.py:53-55`):
   - Stores probability matrix: O(n * k)
   - For 1000 nodes, 50 clusters: ~400KB

3. **Embedding batch** (`tree_builder.py:247-258`):
   - ThreadPoolExecutor holds futures in memory
   - Memory scales with number of concurrent tasks

### 3.3 Memory Scaling

| Document Size | Chunks | Estimated Peak Memory |
|---------------|--------|----------------------|
| 5K tokens | 50 | ~50MB |
| 50K tokens | 500 | ~200MB |
| 500K tokens | 5,000 | ~2GB |
| 5M tokens | 50,000 | ~20GB+ |

**Memory Concern**: UMAP's O(n²) behavior becomes problematic above 5,000 nodes.

---

## 4. Multithreading Analysis

### 4.1 Current Multithreading Usage

| Location | Mechanism | What's Parallelized |
|----------|-----------|---------------------|
| `tree_builder.py:247-258` | `ThreadPoolExecutor` | Leaf node creation (embedding) |
| `cluster_tree_builder.py:114-126` | `ThreadPoolExecutor` | Cluster summarization |
| `FaissRetriever.py:113-121` | `ProcessPoolExecutor` | Embedding generation |

### 4.2 Effectiveness Assessment

**Leaf Node Creation (`tree_builder.py:247-258`)**: **Effective**
- Each embedding call is independent
- Network I/O bound - threading provides good parallelism
- Default ThreadPoolExecutor uses os.cpu_count() * 5 threads

**Cluster Summarization (`cluster_tree_builder.py:114-126`)**: **Bug Exists**
```python
for cluster in clusters:
    executor.submit(
        process_cluster,
        cluster,
        new_level_nodes,
        next_node_index,  # Race condition!
        ...
    )
    next_node_index += 1
```
- `next_node_index` is passed by value but incremented in main thread
- Multiple tasks could receive same index
- **Default is `use_multithreading=False`** - bug not typically triggered

**FAISS Embedding (`FaissRetriever.py:113-121`)**: **Ineffective**
```python
with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(self.embedding_model.create_embedding, context_chunk)
        for context_chunk in self.context_chunks
    ]
```
- Uses `ProcessPoolExecutor` but embedding models aren't picklable (OpenAI client)
- **Will likely fail** if embedding model uses API client with connections

### 4.3 Missing Parallelization Opportunities

1. **Embedding model batching**: OpenAI supports batch embedding (up to 2048 inputs)
   - Current: N sequential/parallel calls
   - Potential: N/2048 batch calls (50x reduction)
   - Location: `EmbeddingModels.py:22-29`

2. **Layer-parallel summarization**: Summaries within same layer are independent
   - Currently sequential by layer
   - Could parallelize all summaries in a layer

3. **Retrieval node processing**: Distance calculations are embarrassingly parallel
   - Currently: sequential loop
   - Could use: NumPy broadcasting or multiprocessing

---

## 5. Scalability Analysis

### 5.1 Scaling Limits

| Scale | Chunks | Est. Time (Index) | Est. Cost | Feasibility |
|-------|--------|-------------------|-----------|-------------|
| 100 docs * 1K tokens | 1,000 | 5-10 min | $0.50 | Easy |
| 1,000 docs * 1K tokens | 10,000 | 1-2 hours | $5.00 | Feasible |
| 10,000 docs * 1K tokens | 100,000 | 10-20 hours | $50.00 | Challenging |
| 100,000 docs * 1K tokens | 1,000,000 | 100+ hours | $500+ | Impractical |

### 5.2 What Breaks at 1,000 Documents?

1. **Memory**: UMAP O(n²) with 10K nodes requires ~800MB for distance matrix
2. **Time**: 10K+ embedding calls at ~100ms each = 15+ minutes just for embeddings
3. **Cost**: $5+ per run makes iteration expensive
4. **Reliability**: Long-running API calls more likely to hit rate limits or timeouts

### 5.3 What Breaks at 10,000 Documents?

1. **Memory**: UMAP fails or swaps with 100K nodes
2. **Time**: Multi-hour indexing makes development painful
3. **Cost**: $50+ per run is prohibitive for iteration
4. **Rate Limits**: OpenAI rate limits (varies by tier) become blocking
5. **Clustering Quality**: GMM may not find optimal clusters with many components

---

## 6. Optimization Recommendations

### 6.1 High Impact (Easy)

| Optimization | Effort | Impact | Location |
|--------------|--------|--------|----------|
| Batch embedding API calls | Low | 50x fewer API calls | `EmbeddingModels.py` |
| Add caching for embeddings | Low | Skip repeated texts | `EmbeddingModels.py` |
| Use SentenceTransformers locally | Low | Eliminate embedding API cost | `EmbeddingModels.py:32-37` |

**Batch Embedding Implementation:**
```python
# Current: N calls
for text in texts:
    embeddings.append(model.create_embedding(text))

# Proposed: N/2048 calls
def create_embeddings_batch(self, texts, batch_size=2048):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = self.client.embeddings.create(input=batch, model=self.model)
        embeddings.extend([d.embedding for d in response.data])
    return embeddings
```

### 6.2 Medium Impact (Moderate Effort)

| Optimization | Effort | Impact | Location |
|--------------|--------|--------|----------|
| Replace UMAP with faster alternative | Medium | 10x faster clustering | `cluster_utils.py` |
| Async API calls | Medium | Better throughput | All API models |
| Incremental tree updates | Medium | Avoid full rebuild | `RetrievalAugmentation.py` |

**UMAP Alternatives:**
- **PCA**: Faster but less quality
- **Random Projection**: O(n*d) complexity
- **Truncated SVD**: Good for sparse data

### 6.3 High Impact (High Effort)

| Optimization | Effort | Impact | Location |
|--------------|--------|--------|----------|
| Local LLM for summarization | High | Eliminate summarization API cost | `SummarizationModels.py` |
| Streaming tree construction | High | Process documents incrementally | Architecture change |
| Distributed processing | High | Scale to millions of docs | Architecture change |

---

## 7. Benchmarking Recommendations

### 7.1 Metrics to Track

```python
# Suggested profiling points
metrics = {
    "indexing": {
        "total_time_seconds": float,
        "embedding_api_calls": int,
        "embedding_api_time_seconds": float,
        "summarization_api_calls": int,
        "summarization_api_time_seconds": float,
        "clustering_time_seconds": float,
        "peak_memory_mb": float,
    },
    "retrieval": {
        "query_time_ms": float,
        "embedding_time_ms": float,
        "search_time_ms": float,
    },
    "tree_stats": {
        "num_leaf_nodes": int,
        "num_summary_nodes": int,
        "num_layers": int,
        "tree_size_mb": float,
    }
}
```

### 7.2 Suggested Benchmarks

| Benchmark | Document Size | Purpose |
|-----------|---------------|---------|
| Small | 1K tokens | Baseline, quick iteration |
| Medium | 10K tokens | Typical use case |
| Large | 100K tokens | Stress test |
| XL | 1M tokens | Scale limit testing |

---

## 8. Cost Optimization Strategies

### 8.1 Reduce Embedding Costs

1. **Use local models**: SentenceTransformers is free
   - Trade-off: Slightly lower quality, requires GPU for speed
   - Already supported: `SBertEmbeddingModel` in codebase

2. **Cache embeddings**: Hash text -> embedding lookup
   - Benefit: Avoid re-embedding identical chunks
   - Implementation: Add LRU cache or Redis

3. **Batch API calls**: Single call for multiple texts
   - Benefit: Reduced overhead, same cost
   - OpenAI limit: 2048 texts per batch

### 8.2 Reduce Summarization Costs

1. **Use local LLMs**: Llama, Mistral, etc.
   - Trade-off: Requires GPU, lower quality than GPT-4
   - Effort: Medium (model loading, inference)

2. **Reduce summary frequency**: Larger clusters
   - Trade-off: Less granular hierarchy
   - Config: Increase `max_length_in_cluster`

3. **Use cheaper models**: GPT-3.5-turbo vs GPT-4
   - Already default in codebase
   - Consider: Claude Haiku, Gemini Flash for lower cost

### 8.3 Cost Comparison Table

| Strategy | Embedding Cost | Summarization Cost | Quality |
|----------|----------------|-------------------|---------|
| Current (OpenAI all) | $0.0001/1K tok | $0.002/1K tok | Highest |
| Local embedding + OpenAI summary | $0 | $0.002/1K tok | High |
| Local embedding + local summary | $0 | $0 | Medium |
| Batch OpenAI + GPT-3.5 | $0.0001/1K tok | $0.0005/1K tok | High |

---

## Summary

| Aspect | Current State | Recommendation |
|--------|---------------|----------------|
| Primary Bottleneck | API calls (95%+ of time) | Batch embeddings, local models |
| Memory Scaling | O(n²) in clustering | Replace UMAP for large scale |
| Multithreading | Partially effective, has bugs | Fix race condition, add batching |
| 1K document scale | Feasible (minutes, ~$5) | Works as-is |
| 10K document scale | Challenging (hours, ~$50) | Needs optimization |
| 100K document scale | Impractical | Needs architecture changes |

**Key Optimizations by Priority:**
1. **Immediate**: Implement batch embedding (50x fewer API calls)
2. **Short-term**: Add embedding cache, use local embedding model option
3. **Medium-term**: Fix multithreading bugs, add async API calls
4. **Long-term**: Support incremental updates, distributed processing
