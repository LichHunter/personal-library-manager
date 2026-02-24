# 11. Cluster-Based Adaptive Retrieval

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P3 |
| **Complexity** | HIGH |
| **Expected Improvement** | Better diversity + relevance |
| **PLM Changes Required** | Clustering pipeline + selection logic |
| **External Dependencies** | Clustering library |

## Description

Cluster retrieved documents by topic/content, then select representatives from each cluster. Ensures diversity while maintaining relevance.

## How It Works

```
1. Retrieve large candidate set (top-50)
2. Cluster candidates by semantic similarity
3. Score clusters by query relevance
4. Select top documents from top clusters
5. Return diverse, relevant set
```

## Why It Works

- Initial retrieval may have redundant results (all about same subtopic)
- Clustering groups similar documents
- Selection from clusters ensures coverage of different aspects
- Reduces redundancy, improves diversity

## Algorithm

```python
from sklearn.cluster import KMeans

def cluster_adaptive_retrieve(query: str, k: int = 10, n_clusters: int = 5):
    # 1. Over-retrieve
    candidates = retrieve(query, k=50)
    
    # 2. Get embeddings for clustering
    embeddings = [get_embedding(doc) for doc in candidates]
    
    # 3. Cluster
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 4. Group by cluster
    clusters = defaultdict(list)
    for doc, label in zip(candidates, cluster_labels):
        clusters[label].append(doc)
    
    # 5. Score clusters by best doc score
    cluster_scores = {}
    for label, docs in clusters.items():
        cluster_scores[label] = max(doc.score for doc in docs)
    
    # 6. Select from clusters (round-robin from top clusters)
    sorted_clusters = sorted(cluster_scores.keys(), 
                            key=lambda l: cluster_scores[l], 
                            reverse=True)
    
    results = []
    for cluster_label in sorted_clusters:
        cluster_docs = sorted(clusters[cluster_label], 
                             key=lambda d: d.score, reverse=True)
        results.extend(cluster_docs[:k // n_clusters + 1])
    
    return results[:k]
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `candidate_k` | Initial retrieval size | 50 |
| `n_clusters` | Number of clusters | 5-10 |
| `per_cluster` | Docs to take per cluster | 2-3 |
| `cluster_method` | Clustering algorithm | KMeans |

## Cluster Selection Strategies

### A. Round-Robin (Balanced)
Take equal number from each cluster.

### B. Weighted (Relevance-Biased)
Take more from higher-scoring clusters.

### C. Top-1 Per Cluster (Maximum Diversity)
Take only best doc from each cluster.

## Tradeoffs

| Pros | Cons |
|------|------|
| Ensures diverse coverage | Clustering adds latency |
| Reduces redundancy | May miss if cluster count wrong |
| Works with any retriever | Requires embedding access |
| Interpretable (clusters visible) | Parameter tuning needed |

## Implementation Steps

1. Implement clustering on retrieved candidates
2. Implement cluster scoring (query relevance)
3. Implement selection strategy
4. Tune n_clusters for corpus
5. Benchmark: standard vs cluster-adaptive

## PLM Consideration

Lower priority because:
- Adds complexity
- May not help if retrieved docs are already diverse
- Other approaches (reranking) simpler with proven gains

Consider if:
- Observing redundant retrieval results
- Need explicit diversity guarantee
- Multi-aspect queries common

## References

- [Cluster-based Adaptive Retrieval Paper](https://arxiv.org/abs/2511.14769)
- [MMR - Maximal Marginal Relevance](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
