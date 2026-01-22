# RAPTOR Clustering Algorithm

> Detailed analysis of the clustering algorithms used in RAPTOR for hierarchical document organization.

## Overview

RAPTOR uses a sophisticated two-stage clustering approach that combines:
1. **UMAP** (Uniform Manifold Approximation and Projection) for dimensionality reduction
2. **GMM** (Gaussian Mixture Models) for soft clustering with automatic cluster count selection

This document covers the implementation in `cluster_utils.py` (lines 1-186).

---

## Algorithm Architecture

```
Input Embeddings (high-dimensional)
         |
         v
+-------------------+
| Global Clustering |  <- UMAP reduces to `dim` dimensions
|   (UMAP + GMM)    |     GMM finds optimal clusters via BIC
+-------------------+
         |
         v (for each global cluster)
+-------------------+
| Local Clustering  |  <- UMAP with fixed neighbors
|   (UMAP + GMM)    |     GMM refines clusters
+-------------------+
         |
         v
+-------------------+
|    Validation     |  <- Token length check
| (Recursive split) |     Re-cluster if too large
+-------------------+
         |
         v
Final Node Clusters
```

---

## 1. UMAP Dimensionality Reduction

### What is UMAP?

UMAP is a manifold learning technique that reduces high-dimensional data to lower dimensions while preserving both local and global structure. It's similar to t-SNE but faster and better at preserving global relationships.

### Global Clustering UMAP

```python
# cluster_utils.py:23-34
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_neighbors` | `sqrt(n - 1)` | Adaptive to dataset size; balances local vs global structure |
| `n_components` | `dim` (default 10) | Target dimensionality for clustering |
| `metric` | `"cosine"` | Semantic similarity measure for text embeddings |

**Why `sqrt(n-1)` for neighbors?**
- Small datasets: More neighbors relative to size = captures global structure
- Large datasets: Fewer neighbors relative to size = focuses on local structure
- This heuristic provides automatic scaling

### Local Clustering UMAP

```python
# cluster_utils.py:37-43
def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_neighbors` | 10 (fixed) | Focuses on local structure within global clusters |
| `n_components` | `dim` | Same target dimensionality |
| `metric` | `"cosine"` | Consistent similarity measure |

**Why fixed neighbors for local?**
- Global clusters are already semantically coherent
- Fixed small value emphasizes fine-grained local relationships
- Prevents over-smoothing of local structure

---

## 2. Gaussian Mixture Model (GMM) Clustering

### Optimal Cluster Selection via BIC

The Bayesian Information Criterion (BIC) is used to automatically determine the optimal number of clusters:

```python
# cluster_utils.py:46-57
def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters
```

### BIC Formula

The Bayesian Information Criterion balances model fit against complexity:

$$BIC = -2 \cdot \ln(\hat{L}) + k \cdot \ln(n)$$

Where:
- $\hat{L}$ = maximized likelihood of the model
- $k$ = number of parameters (increases with more clusters)
- $n$ = number of data points

**Lower BIC = Better model** (balances fit vs complexity)

### Pseudocode: Optimal Cluster Selection

```
function GET_OPTIMAL_CLUSTERS(embeddings, max_clusters=50):
    max_clusters = min(max_clusters, len(embeddings))
    bic_scores = []
    
    for n_clusters in range(1, max_clusters):
        gmm = fit_gaussian_mixture(embeddings, n_components=n_clusters)
        bic = compute_bic(gmm, embeddings)
        bic_scores.append(bic)
    
    return argmin(bic_scores) + 1  // +1 because range starts at 1
```

---

## 3. Soft Clustering with Threshold

### What is Soft Clustering?

Unlike hard clustering (each point belongs to exactly one cluster), **soft clustering** assigns probability distributions over clusters. A data point can belong to multiple clusters.

```python
# cluster_utils.py:60-66
def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters
```

### How the Threshold Works

For each data point, GMM returns a probability vector:
```
probs[i] = [P(cluster_0), P(cluster_1), ..., P(cluster_k)]
```

The threshold determines membership:
```python
labels[i] = [j for j in range(k) if probs[i][j] > threshold]
```

**Example with threshold = 0.1:**
```
probs = [0.05, 0.35, 0.60]  # probabilities for 3 clusters

# Point belongs to clusters where prob > 0.1
labels = [1, 2]  # belongs to cluster 1 AND cluster 2
```

### Why Soft Clustering?

Soft clustering allows a text chunk to belong to **multiple semantic groups**:
- A paragraph about "machine learning in healthcare" might belong to both an "ML" cluster and a "healthcare" cluster
- This creates richer summaries by grouping related content from different perspectives
- Prevents information loss at cluster boundaries

**Trade-off:**
- Low threshold (e.g., 0.1): More overlap, potentially redundant summaries
- High threshold (e.g., 0.5): Less overlap, more distinct clusters

---

## 4. Complete Clustering Pipeline

### Main Orchestration Function

```python
# cluster_utils.py:69-123
def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
```

### Step-by-Step Algorithm

```
function PERFORM_CLUSTERING(embeddings, dim, threshold):
    // Stage 1: Global Clustering
    reduced_global = UMAP(embeddings, 
                          dim=min(dim, n-2),
                          n_neighbors=sqrt(n-1))
    
    global_clusters, n_global = GMM_CLUSTER(reduced_global, threshold)
    
    // Initialize results
    all_local_clusters = [empty_array for each embedding]
    total_clusters = 0
    
    // Stage 2: Local Clustering within each global cluster
    for i in range(n_global):
        // Get embeddings belonging to global cluster i
        cluster_mask = [i in gc for gc in global_clusters]
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            continue
        
        if len(cluster_embeddings) <= dim + 1:
            // Too few points for UMAP - assign all to single local cluster
            local_clusters = [[0] for each point]
            n_local = 1
        else:
            // Apply local UMAP + GMM
            reduced_local = UMAP(cluster_embeddings,
                                 dim=dim,
                                 n_neighbors=10)
            local_clusters, n_local = GMM_CLUSTER(reduced_local, threshold)
        
        // Map local clusters back to original indices
        for j in range(n_local):
            local_mask = [j in lc for lc in local_clusters]
            local_embeddings = cluster_embeddings[local_mask]
            
            // Find original indices
            original_indices = find_matching_rows(embeddings, local_embeddings)
            
            for idx in original_indices:
                all_local_clusters[idx].append(j + total_clusters)
        
        total_clusters += n_local
    
    return all_local_clusters
```

### Edge Cases Handled

1. **Too few points for UMAP** (line 93-95):
   - If `len(embeddings) <= dim + 1`, UMAP cannot reduce dimensions
   - Solution: Assign all points to a single cluster

2. **Empty global clusters** (line 91-92):
   - Some global clusters may be empty after soft assignment
   - Solution: Skip and continue

3. **Dimension limit** (line 72):
   - `dim = min(dim, len(embeddings) - 2)` ensures valid UMAP

---

## 5. RAPTOR_Clustering Class

The high-level clustering algorithm with validation and recursive splitting:

```python
# cluster_utils.py:132-185
class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
```

### Complete Algorithm

```
function RAPTOR_CLUSTERING(nodes, embedding_model, max_length=3500):
    // Extract embeddings from nodes
    embeddings = [node.embeddings[embedding_model] for node in nodes]
    
    // Perform two-stage clustering
    cluster_labels = PERFORM_CLUSTERING(embeddings, dim=10, threshold=0.1)
    
    // Group nodes by cluster label
    node_clusters = []
    
    for label in unique(flatten(cluster_labels)):
        // Find all nodes belonging to this cluster
        indices = [i for i, labels in enumerate(cluster_labels) if label in labels]
        cluster_nodes = [nodes[i] for i in indices]
        
        // Base case: single node cluster
        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue
        
        // Validation: check total token length
        total_tokens = sum(token_count(node.text) for node in cluster_nodes)
        
        if total_tokens > max_length:
            // RECURSIVE SPLIT: cluster is too large
            sub_clusters = RAPTOR_CLUSTERING(cluster_nodes, embedding_model, max_length)
            node_clusters.extend(sub_clusters)
        else:
            // Cluster is valid
            node_clusters.append(cluster_nodes)
    
    return node_clusters
```

### Why Recursive Splitting?

The `max_length_in_cluster` (default: 3500 tokens) ensures:
1. **Summarization quality**: LLMs have context limits and work better with focused content
2. **Consistent granularity**: Prevents one massive cluster and many tiny ones
3. **Cost control**: Limits tokens sent to summarization API

### Visualization of Recursive Split

```
Initial Cluster (5000 tokens)
         |
    [too large]
         |
         v
  RAPTOR_CLUSTERING(recursive)
         |
    +---------+
    |         |
Cluster A   Cluster B
(2800 tok)  (2200 tok)
  [OK]        [OK]
```

---

## 6. Configuration Parameters

### Default Values

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `reduction_dimension` | 10 | ClusterTreeConfig:20 | UMAP target dimensions |
| `threshold` | 0.1 | RAPTOR_Clustering:139 | Soft clustering cutoff |
| `max_length_in_cluster` | 3500 | RAPTOR_Clustering:136 | Token limit per cluster |
| `max_clusters` | 50 | get_optimal_clusters:47 | BIC search range |
| `n_neighbors` (local) | 10 | local_cluster_embeddings:38 | UMAP local neighbors |
| `metric` | "cosine" | Both UMAP functions | Distance metric |

### Parameter Tuning Guide

**`threshold` (0.0 - 1.0):**
- Lower (0.05-0.1): More cluster overlap, richer summaries, higher cost
- Higher (0.3-0.5): Less overlap, more distinct clusters, lower cost

**`reduction_dimension`:**
- Lower (5-10): More aggressive reduction, faster, may lose nuance
- Higher (15-20): Preserves more structure, slower, better for large docs

**`max_length_in_cluster`:**
- Lower (2000): More splits, finer granularity, more LLM calls
- Higher (4000+): Fewer splits, coarser summaries, fewer LLM calls

---

## 7. Mathematical Summary

### UMAP Objective

UMAP minimizes cross-entropy between high-dimensional and low-dimensional fuzzy set memberships:

$$C = \sum_{i,j} \left[ v_{ij} \log\frac{v_{ij}}{w_{ij}} + (1-v_{ij}) \log\frac{1-v_{ij}}{1-w_{ij}} \right]$$

Where:
- $v_{ij}$ = high-dimensional similarity between points i and j
- $w_{ij}$ = low-dimensional similarity

### GMM Probability

For a point $x$, the probability of belonging to cluster $k$:

$$P(k|x) = \frac{\pi_k \mathcal{N}(x|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x|\mu_j, \Sigma_j)}$$

Where:
- $\pi_k$ = mixture weight for cluster k
- $\mathcal{N}(x|\mu_k, \Sigma_k)$ = Gaussian density

### Soft Assignment

A point belongs to cluster $k$ if:

$$P(k|x) > \tau$$

Where $\tau$ is the threshold (default 0.1).

---

## 8. Source Code References

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `global_cluster_embeddings` | cluster_utils.py | 23-34 | Global UMAP reduction |
| `local_cluster_embeddings` | cluster_utils.py | 37-43 | Local UMAP reduction |
| `get_optimal_clusters` | cluster_utils.py | 46-57 | BIC-based cluster selection |
| `GMM_cluster` | cluster_utils.py | 60-66 | Soft GMM clustering |
| `perform_clustering` | cluster_utils.py | 69-123 | Two-stage pipeline |
| `RAPTOR_Clustering.perform_clustering` | cluster_utils.py | 133-185 | Full algorithm with validation |
