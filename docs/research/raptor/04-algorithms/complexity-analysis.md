# RAPTOR Complexity Analysis

> Time and space complexity analysis of RAPTOR algorithms with bottleneck identification and scaling characteristics.

## Overview

This document analyzes the computational complexity of RAPTOR's core operations:
1. Tree Building (indexing)
2. Retrieval (querying)
3. Storage (memory/disk)

---

## 1. Variable Definitions

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of text chunks (leaf nodes) |
| $L$ | Number of tree layers |
| $d$ | Embedding dimension (e.g., 1536 for OpenAI) |
| $r$ | UMAP reduction dimension (default: 10) |
| $k$ | Top-K selection parameter |
| $c$ | Average cluster size |
| $T$ | Total nodes in tree |
| $M$ | Maximum tokens per chunk |
| $S$ | Summarization token limit |

---

## 2. Tree Building Complexity

### 2.1 Text Splitting

```
function SPLIT_TEXT(text, max_tokens):
    sentences = regex_split(text)      // O(|text|)
    for each sentence:                 // O(num_sentences)
        tokens = tokenize(sentence)    // O(|sentence|)
        // chunk building logic        // O(1) amortized
```

**Time Complexity:** $O(|text|)$ - linear in text length

**Space Complexity:** $O(N \cdot M)$ - stores N chunks of max M tokens

### 2.2 Leaf Node Creation

For each chunk, we:
1. Generate embedding: $O(M)$ API call (token-dependent)
2. Store node: $O(d)$ for embedding storage

```
Total embedding calls: N
Time per call: O(M) + network latency
```

**Time Complexity:** $O(N \cdot (M + \text{API latency}))$

With multithreading (default), effective time is:
$$T_{embedding} = \frac{N \cdot (M + \text{latency})}{\text{num\_threads}}$$

**Space Complexity:** $O(N \cdot d)$ - N embeddings of dimension d

### 2.3 Clustering per Layer

#### UMAP Dimensionality Reduction

UMAP complexity for n points:
- Graph construction: $O(n \cdot k_{nn} \cdot \log n)$ where $k_{nn}$ = neighbors
- Optimization: $O(n)$ iterations

For global UMAP with $k_{nn} = \sqrt{n}$:
$$O(n \cdot \sqrt{n} \cdot \log n) = O(n^{1.5} \log n)$$

For local UMAP with fixed $k_{nn} = 10$:
$$O(n \cdot 10 \cdot \log n) = O(n \log n)$$

#### GMM Clustering

GMM with automatic cluster selection (BIC):
- Fit GMM with k clusters: $O(n \cdot k \cdot d \cdot i)$ where i = iterations
- BIC search from 1 to max_clusters (50): $O(50 \cdot n \cdot k_{max} \cdot r \cdot i)$

Simplified: $O(n \cdot r)$ per cluster count tested

**Total clustering per layer:**
$$O(n^{1.5} \log n + 50 \cdot n \cdot r)$$

### 2.4 Summarization per Layer

For a layer with $n$ nodes clustered into $c$ clusters:
- Number of summaries: $n / c$ (approximately)
- Tokens per summary input: $c \cdot M$
- LLM call: $O(c \cdot M)$ + API latency

**Time per layer:** $O(\frac{n}{c} \cdot c \cdot M) = O(n \cdot M)$ + API latency

**Note:** This is the major bottleneck due to LLM API calls.

### 2.5 Total Tree Building

```
Layer 0: N leaf nodes, embedding cost O(N*M)
Layer 1: ~N/c nodes, clustering O(N^1.5 log N), summarizing O(N*M)
Layer 2: ~N/c^2 nodes, clustering O((N/c)^1.5 log(N/c)), summarizing O((N/c)*M)
...
Layer L: ~N/c^L nodes
```

**Total Time Complexity:**

$$T_{build} = O(N \cdot M) + \sum_{l=0}^{L} \left[ O\left(\left(\frac{N}{c^l}\right)^{1.5} \log \frac{N}{c^l}\right) + O\left(\frac{N}{c^l} \cdot M\right) \right]$$

Dominated by:
1. **LLM summarization calls** (if API-bound)
2. **UMAP on first layer** (if compute-bound)

**Simplified:** $O(N^{1.5} \log N + L \cdot N \cdot M \cdot \text{API\_latency})$

**Space Complexity:** $O(T \cdot d)$ where $T = N + \frac{N}{c} + \frac{N}{c^2} + ... \approx \frac{c \cdot N}{c-1}$

For typical $c \approx 5$: $T \approx 1.25N$

### 2.6 Tree Building Summary Table

| Operation | Time Complexity | Bottleneck Factor |
|-----------|-----------------|-------------------|
| Text splitting | $O(\|text\|)$ | CPU |
| Leaf embeddings | $O(N \cdot M)$ | API latency |
| UMAP (per layer) | $O(n^{1.5} \log n)$ | CPU |
| GMM BIC (per layer) | $O(50 \cdot n \cdot r)$ | CPU |
| Summarization (per layer) | $O(\frac{n}{c} \cdot M)$ | API latency |
| **Total** | $O(N^{1.5} \log N + L \cdot N \cdot M)$ | **LLM API** |

---

## 3. Retrieval Complexity

### 3.1 Collapsed Tree Retrieval

```
function COLLAPSED_RETRIEVAL(query, tree, top_k, max_tokens):
    query_embedding = embed(query)     // O(|query|) + API
    
    all_nodes = tree.all_nodes         // T nodes
    distances = []
    for node in all_nodes:             // O(T)
        dist = cosine(query_emb, node.emb)  // O(d)
        distances.append(dist)
    
    sorted_indices = argsort(distances)  // O(T log T)
    
    selected = select_top_k(sorted_indices, max_tokens)  // O(k)
    context = concatenate(selected)    // O(k * M)
```

**Time Complexity:** $O(T \cdot d + T \log T)$

Since $T \approx 1.25N$ and typically $d \gg \log T$:
$$O(N \cdot d)$$

**Space Complexity:** $O(T)$ for distance array

### 3.2 Tree Traversal Retrieval

```
function TRAVERSAL_RETRIEVAL(query, tree, start_layer, num_layers, top_k):
    query_embedding = embed(query)     // O(|query|) + API
    
    current_nodes = tree.layer_to_nodes[start_layer]  // n_start nodes
    selected = []
    
    for layer in range(num_layers):    // L iterations
        distances = []
        for node in current_nodes:     // n_layer nodes
            dist = cosine(query_emb, node.emb)  // O(d)
            distances.append(dist)
        
        sorted_indices = argsort(distances)  // O(n_layer log n_layer)
        best = select_top_k(sorted_indices)  // O(k)
        selected.extend(best)
        
        // Get children for next layer
        children = get_children(best)  // O(k * avg_children)
        current_nodes = deduplicate(children)
```

**Time Complexity:**

Starting from root layer with $n_{root}$ nodes, each layer has at most $k \cdot c$ nodes to consider:

$$T_{traversal} = \sum_{l=0}^{L} O(n_l \cdot d + n_l \log n_l)$$

Where $n_l \leq k \cdot c^l$ (exponential in layer depth from root).

**Worst case:** $O(L \cdot k \cdot c \cdot d)$

**Best case (sparse selection):** $O(L \cdot k \cdot d)$

**Space Complexity:** $O(L \cdot k)$ for selected nodes

### 3.3 Retrieval Summary Table

| Method | Time Complexity | Space | Best For |
|--------|-----------------|-------|----------|
| Collapsed | $O(N \cdot d)$ | $O(N)$ | Small-medium trees |
| Traversal | $O(L \cdot k \cdot c \cdot d)$ | $O(L \cdot k)$ | Large trees, specific queries |

### 3.4 Retrieval Comparison

For a document with N=1000 chunks, d=1536, L=3, k=5, c=10:

**Collapsed Tree:**
- Distance calculations: $1250 \times 1536 = 1.92M$ operations
- Sorting: $1250 \log 1250 \approx 8K$ comparisons

**Tree Traversal:**
- Layer 2: 1 root, 5 comparisons
- Layer 1: ~10 nodes, 50 comparisons
- Layer 0: ~50 nodes, 250 comparisons
- Total: ~305 comparisons with $d=1536$ each = 468K operations

**Traversal is ~4x faster** for this configuration, with the gap widening for larger trees.

---

## 4. Space Complexity

### 4.1 Tree Storage

| Component | Size | Formula |
|-----------|------|---------|
| Node texts | Variable | $\sum_{i} \|text_i\|$ |
| Node embeddings | Fixed | $T \cdot d \cdot sizeof(float)$ |
| Children indices | Variable | $\sum_{i} \|children_i\|$ |
| Layer mapping | Small | $O(L \cdot N)$ |

**Total:** $O(T \cdot d + \sum texts)$

### 4.2 Concrete Example

For a 100-page document (~50K words):
- Chunks: N = 500 (at 100 tokens each)
- Total nodes: T = 625 (with c=5, L=3)
- Embedding dimension: d = 1536

**Memory breakdown:**
```
Embeddings: 625 * 1536 * 4 bytes = 3.84 MB
Texts: ~50K words * 5 bytes = 250 KB
Indices: ~625 * 5 * 4 bytes = 12.5 KB
Total: ~4.1 MB
```

### 4.3 Scaling with Document Size

| Document Size | Chunks (N) | Total Nodes (T) | Memory |
|---------------|------------|-----------------|--------|
| 10 pages | 50 | 63 | ~400 KB |
| 100 pages | 500 | 625 | ~4 MB |
| 1000 pages | 5000 | 6250 | ~40 MB |
| 10000 pages | 50000 | 62500 | ~400 MB |

---

## 5. Bottleneck Analysis

### 5.1 Tree Building Bottlenecks

```
           Bottleneck Hierarchy (Tree Building)
           
     +------------------------------------------+
     |         LLM Summarization API            |  <<< PRIMARY
     |   (Network latency, rate limits, cost)   |
     +------------------------------------------+
                        |
                        v
     +------------------------------------------+
     |          Embedding API Calls             |  <<< SECONDARY
     |    (Network latency, rate limits)        |
     +------------------------------------------+
                        |
                        v
     +------------------------------------------+
     |         UMAP Dimensionality              |  <<< TERTIARY
     |      Reduction (CPU-bound)               |
     +------------------------------------------+
                        |
                        v
     +------------------------------------------+
     |       GMM Clustering (CPU-bound)         |
     +------------------------------------------+
```

**Why LLM is the bottleneck:**
1. **Latency:** 1-5 seconds per call vs milliseconds for local operations
2. **Rate limits:** Typically 60-3500 RPM depending on tier
3. **Cost:** $0.001-0.06 per 1K tokens

**Mitigation strategies:**
- Batch embeddings where possible
- Use multithreading for LLM calls (already implemented)
- Consider local LLMs for summarization

### 5.2 Retrieval Bottlenecks

```
           Bottleneck Hierarchy (Retrieval)
           
     +------------------------------------------+
     |      Query Embedding (single API call)   |  <<< PRIMARY
     +------------------------------------------+
                        |
                        v
     +------------------------------------------+
     |     Distance Calculations (CPU/memory)   |  <<< SECONDARY
     +------------------------------------------+
                        |
                        v
     +------------------------------------------+
     |            Sorting Results               |
     +------------------------------------------+
```

**Why query embedding is the bottleneck:**
- Single API call dominates for small-medium trees
- For very large trees, distance calculation may dominate

---

## 6. Scaling Characteristics

### 6.1 Document Size Scaling

| Metric | Scaling | Notes |
|--------|---------|-------|
| Build time | $O(N^{1.5} \log N)$ | Dominated by LLM calls: $O(N)$ |
| Retrieval time | $O(N)$ collapsed | $O(\log N)$ traversal |
| Memory | $O(N)$ | Linear with document size |
| API calls (build) | $O(N)$ | One embed + one summary per chunk |
| API calls (retrieve) | $O(1)$ | Single query embedding |

### 6.2 Multi-Document Scaling

RAPTOR processes documents independently. For M documents:

| Metric | Scaling |
|--------|---------|
| Build time | $O(M \cdot N)$ |
| Memory (separate trees) | $O(M \cdot N)$ |
| Query time (each tree) | $O(N)$ or $O(\log N)$ |
| Query time (all trees) | $O(M \cdot N)$ |

**Note:** RAPTOR doesn't natively support cross-document retrieval with a unified tree.

### 6.3 Parameter Impact

| Parameter | Increase Effect on Build | Increase Effect on Retrieval |
|-----------|-------------------------|------------------------------|
| `max_tokens` (chunk size) | Fewer chunks, faster | Fewer nodes, faster |
| `num_layers` | More summarization calls | Deeper traversal |
| `reduction_dimension` | Slightly slower UMAP | No effect |
| `top_k` | No effect | More nodes selected |
| `threshold` (clustering) | More cluster overlap | No effect |

---

## 7. Optimization Recommendations

### 7.1 For Large Documents

1. **Increase `max_tokens`** (chunk size) to reduce N
2. **Use local embeddings** (e.g., sentence-transformers) to eliminate API latency
3. **Use local LLM** for summarization if quality acceptable
4. **Parallelize** with higher thread count

### 7.2 For Fast Retrieval

1. **Use tree traversal** instead of collapsed for large trees
2. **Pre-compute** query embeddings for common queries
3. **Cache** frequently accessed nodes
4. **Index** embeddings with FAISS for sub-linear search

### 7.3 For Memory Constraints

1. **Quantize embeddings** (float32 -> float16 or int8)
2. **Lazy load** node texts from disk
3. **Prune** low-relevance nodes from tree
4. **Compress** text content

---

## 8. Complexity Comparison with Alternatives

| System | Build Complexity | Retrieval Complexity | Memory |
|--------|------------------|----------------------|--------|
| RAPTOR | $O(N^{1.5} \log N)$ | $O(N)$ or $O(\log N)$ | $O(N \cdot d)$ |
| Basic Vector DB | $O(N)$ | $O(N)$ or $O(\log N)$ with index | $O(N \cdot d)$ |
| FAISS (IVF) | $O(N)$ | $O(\sqrt{N})$ | $O(N \cdot d)$ |
| BM25 | $O(N)$ | $O(N)$ worst, $O(\log N)$ avg | $O(N \cdot \|vocab\|)$ |
| RAPTOR + FAISS | $O(N^{1.5} \log N)$ | $O(\sqrt{N})$ | $O(N \cdot d)$ |

**RAPTOR trade-off:** Higher build cost for semantic hierarchy that enables multi-level abstraction retrieval.

---

## 9. Summary Tables

### Build Complexity

| Phase | Time | Space | Primary Cost |
|-------|------|-------|--------------|
| Split | $O(\|text\|)$ | $O(N \cdot M)$ | CPU |
| Embed | $O(N \cdot M)$ | $O(N \cdot d)$ | API |
| Cluster | $O(N^{1.5} \log N)$ | $O(N)$ | CPU |
| Summarize | $O(L \cdot N \cdot M)$ | $O(d)$ | API |
| **Total** | $O(N^{1.5} \log N + L \cdot N \cdot M)$ | $O(N \cdot d)$ | **LLM API** |

### Retrieval Complexity

| Method | Time | Space | Best Use Case |
|--------|------|-------|---------------|
| Collapsed | $O(N \cdot d)$ | $O(N)$ | Broad queries |
| Traversal | $O(L \cdot k \cdot d)$ | $O(L \cdot k)$ | Specific queries |

### Scaling Summary

| Metric | Small Doc (10pg) | Medium Doc (100pg) | Large Doc (1000pg) |
|--------|------------------|--------------------|--------------------|
| Build time | ~1 min | ~10 min | ~2 hours |
| Memory | ~400 KB | ~4 MB | ~40 MB |
| Collapsed retrieval | ~10 ms | ~100 ms | ~1 s |
| Traversal retrieval | ~5 ms | ~20 ms | ~50 ms |

*Times assume typical API latencies and local CPU processing.*
