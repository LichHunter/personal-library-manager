# RAPTOR Data Flow

> Data transformation through the system from input to output

## 1. System Overview - End-to-End Data Flow

```
+--------------------------------------------------------------------+
|                        INPUT: Raw Text                              |
|                                                                     |
| "Once upon a time, there lived a kind princess in a grand castle.  |
|  She had three loyal friends: a wise owl, a brave knight, and a    |
|  magical fairy. One day, a dragon appeared and threatened the      |
|  kingdom. The princess and her friends worked together to defeat   |
|  the dragon and save everyone. The kingdom celebrated their        |
|  victory with a great feast. They all lived happily ever after."   |
+--------------------------------------------------------------------+
                                    |
                                    v
+====================================================================+
|                      RAPTOR PROCESSING                              |
|                                                                     |
|  1. CHUNKING  -->  2. EMBEDDING  -->  3. CLUSTERING                |
|                         |                   |                       |
|                         v                   v                       |
|                    4. SUMMARIZING  <--  cluster groups              |
|                         |                                           |
|                         v                                           |
|                    5. LAYER BUILDING (repeat 3-4)                   |
|                         |                                           |
|                         v                                           |
|                    6. TREE STRUCTURE                                |
+====================================================================+
                                    |
                                    v
+--------------------------------------------------------------------+
|                        OUTPUT: Hierarchical Tree                    |
|                                                                     |
|                    Layer 2 (Root):                                  |
|                         [S2]                                        |
|                        /    \                                       |
|                    Layer 1:                                         |
|                    [S0]    [S1]                                     |
|                   / | \    / | \                                    |
|                Layer 0 (Leaves):                                    |
|            [C0][C1][C2][C3][C4][C5][C6]                             |
+--------------------------------------------------------------------+
```

## 2. Stage 1: Chunking (Text to Chunks)

**Location:** `utils.py:22-100` - `split_text()`

### Input
```python
text: str = "Once upon a time, there lived a kind princess..."
tokenizer = tiktoken.get_encoding("cl100k_base")
max_tokens = 100  # default
```

### Transformation Logic
1. Split on sentence delimiters: `.`, `!`, `?`, `\n`
2. Count tokens per sentence
3. Accumulate sentences until `max_tokens` exceeded
4. Handle long sentences by sub-splitting on `,`, `;`, `:`

### Output
```python
chunks: List[str] = [
    "Once upon a time, there lived a kind princess in a grand castle",
    "She had three loyal friends: a wise owl, a brave knight, and a magical fairy",
    "One day, a dragon appeared and threatened the kingdom",
    "The princess and her friends worked together to defeat the dragon and save everyone",
    "The kingdom celebrated their victory with a great feast",
    "They all lived happily ever after",
]
# ~6 chunks (depends on tokenizer count)
```

### Data Shape Change
```
Input:  str (1 document, ~500 chars)
Output: List[str] (N chunks, each <= 100 tokens)

Example: 500 chars / ~100 tokens per chunk = ~5-7 chunks
```

## 3. Stage 2: Embedding (Chunks to Leaf Nodes)

**Location:** `tree_builder.py:238-258` - `multithreaded_create_leaf_nodes()`

### Input
```python
chunks: List[str] = ["Once upon a time...", "She had three...", ...]
```

### Transformation Logic
For each chunk, create a Node:
```python
# tree_builder.py:158-179 - create_node()
embeddings = {
    model_name: model.create_embedding(text)
    for model_name, model in self.embedding_models.items()
}
node = Node(text, index, children_indices=set(), embeddings=embeddings)
```

### Output
```python
leaf_nodes: Dict[int, Node] = {
    0: Node(
        text="Once upon a time, there lived a kind princess in a grand castle",
        index=0,
        children=set(),
        embeddings={
            "OpenAI": [0.023, -0.156, 0.089, ..., 0.034]  # 1536-dim
        }
    ),
    1: Node(
        text="She had three loyal friends: a wise owl, a brave knight...",
        index=1,
        children=set(),
        embeddings={"OpenAI": [0.045, -0.112, ..., 0.067]}
    ),
    # ... more nodes
}

layer_to_nodes[0] = list(leaf_nodes.values())
```

### Data Shape Change
```
Input:  List[str] of N chunks
Output: Dict[int, Node] with N nodes
        Each node has 1536-dim embedding vector (OpenAI default)

Memory: N * (text_size + 1536 * 4 bytes) per node
```

## 4. Stage 3: Clustering (Nodes to Clusters)

**Location:** `cluster_utils.py:132-185` - `RAPTOR_Clustering.perform_clustering()`

### Input
```python
nodes: List[Node] = [node0, node1, node2, node3, node4, node5]
embedding_model_name: str = "OpenAI"
reduction_dimension: int = 10
threshold: float = 0.1
max_length_in_cluster: int = 3500
```

### Transformation Logic

#### Step 3a: Extract Embeddings
```python
embeddings = np.array([
    node.embeddings["OpenAI"] for node in nodes
])
# Shape: (6, 1536)
```

#### Step 3b: Global UMAP Reduction
```python
# cluster_utils.py:23-34
n_neighbors = int((6 - 1) ** 0.5)  # ~2
reduced_global = umap.UMAP(
    n_neighbors=n_neighbors,
    n_components=min(10, 6-2),  # 4
    metric="cosine"
).fit_transform(embeddings)
# Shape: (6, 4)
```

#### Step 3c: Global GMM Clustering
```python
# cluster_utils.py:46-66
optimal_k = get_optimal_clusters(reduced_global)  # e.g., 2
gm = GaussianMixture(n_components=2)
gm.fit(reduced_global)
probs = gm.predict_proba(reduced_global)
# Shape: (6, 2)

# Soft assignment with threshold=0.1
labels = [np.where(prob > 0.1)[0] for prob in probs]
# Example: [[0], [0], [0,1], [1], [1], [1]]
# Node 2 belongs to BOTH clusters (soft clustering)
```

#### Step 3d: Local Clustering per Global Cluster
```python
# For global cluster 0: nodes [0, 1, 2]
# For global cluster 1: nodes [2, 3, 4, 5]
# Apply UMAP + GMM locally within each
```

#### Step 3e: Size Validation
```python
# cluster_utils.py:166-183
for cluster_nodes in clusters:
    total_tokens = sum(len(tokenizer.encode(n.text)) for n in cluster_nodes)
    if total_tokens > 3500:
        # Recursive re-clustering
        sub_clusters = RAPTOR_Clustering.perform_clustering(
            cluster_nodes, ...
        )
```

### Output
```python
clusters: List[List[Node]] = [
    [node0, node1],          # Cluster about princess & castle
    [node2, node3],          # Cluster about conflict & resolution  
    [node4, node5],          # Cluster about celebration & ending
]
```

### Data Shape Change
```
Input:  List[Node] with N nodes (each 1536-dim)
        |
        v
        Reduce to D dimensions (D=10 default)
        |
        v
        Group into K clusters
        |
        v
Output: List[List[Node]] with K clusters
        K << N (fewer clusters than nodes)
        Each cluster: avg N/K nodes
```

## 5. Stage 4: Summarization (Clusters to Summary Nodes)

**Location:** `cluster_tree_builder.py:66-85` - `process_cluster()`

### Input (per cluster)
```python
cluster: List[Node] = [node0, node1]
summarization_length: int = 100  # tokens
```

### Transformation Logic

#### Step 4a: Concatenate Text
```python
# utils.py:181-195 - get_text()
node_texts = ""
for node in cluster:
    node_texts += f"{' '.join(node.text.splitlines())}\n\n"

# Result:
# "Once upon a time, there lived a kind princess in a grand castle
#
#  She had three loyal friends: a wise owl, a brave knight, and a magical fairy"
```

#### Step 4b: LLM Summarization
```python
# SummarizationModels.py - GPT3TurboSummarizationModel
summarized_text = self.summarization_model.summarize(
    context=node_texts,
    max_tokens=100
)
# Result: "A princess lived in a castle with three loyal friends."
```

#### Step 4c: Create Summary Node
```python
# tree_builder.py:158-179
new_node = Node(
    text="A princess lived in a castle with three loyal friends.",
    index=next_node_index,  # e.g., 6
    children={0, 1},        # Indices of source nodes
    embeddings={
        "OpenAI": [0.056, -0.123, ..., 0.078]  # Embed summary
    }
)
```

### Output (all clusters)
```python
new_level_nodes: Dict[int, Node] = {
    6: Node(text="A princess lived in a castle...", children={0,1}, ...),
    7: Node(text="A dragon threatened the kingdom...", children={2,3}, ...),
    8: Node(text="The kingdom celebrated...", children={4,5}, ...),
}

layer_to_nodes[1] = list(new_level_nodes.values())  # [node6, node7, node8]
```

### Data Shape Change
```
Input:  K clusters, each with ~N/K nodes
        Total text: N * ~100 tokens = ~N00 tokens
        
Output: K summary nodes
        Each summary: ~100 tokens
        Total: K * 100 tokens

Compression ratio: N/K (e.g., 6 nodes -> 3 summaries = 2x)
```

## 6. Stage 5: Layer Building (Iterate Stages 3-4)

**Location:** `cluster_tree_builder.py:87-151` - `construct_tree()` loop

### Iteration Example

```
Layer 0 (Leaves): 6 nodes
    |
    | Cluster -> 3 groups
    | Summarize -> 3 nodes
    v
Layer 1: 3 nodes  
    |
    | Cluster -> 2 groups  
    | Summarize -> 2 nodes
    v
Layer 2: 2 nodes
    |
    | Cluster -> 1 group
    | Summarize -> 1 node
    v
Layer 3 (Root): 1 node
    |
    | STOP: len(nodes) <= reduction_dim + 1
```

### Stop Condition
```python
# cluster_tree_builder.py:95-100
if len(node_list_current_layer) <= self.reduction_dimension + 1:
    self.num_layers = layer
    break
```

## 7. Final Output: Tree Structure

### Complete Tree Object
```python
tree = Tree(
    all_nodes={
        # Layer 0 - Leaves
        0: Node("Once upon a time...", 0, set(), {...}),
        1: Node("She had three loyal friends...", 1, set(), {...}),
        2: Node("One day, a dragon appeared...", 2, set(), {...}),
        3: Node("The princess and her friends...", 3, set(), {...}),
        4: Node("The kingdom celebrated...", 4, set(), {...}),
        5: Node("They all lived happily...", 5, set(), {...}),
        # Layer 1 - Summaries
        6: Node("A princess with friends...", 6, {0,1}, {...}),
        7: Node("Dragon conflict resolved...", 7, {2,3}, {...}),
        8: Node("Celebration and ending...", 8, {4,5}, {...}),
        # Layer 2 - Higher Summaries
        9: Node("Story of princess vs dragon...", 9, {6,7}, {...}),
        10: Node("Happy conclusion...", 10, {8}, {...}),
        # Layer 3 - Root
        11: Node("Complete fairy tale...", 11, {9,10}, {...}),
    },
    root_nodes={11: Node(...)},
    leaf_nodes={0: ..., 1: ..., 2: ..., 3: ..., 4: ..., 5: ...},
    num_layers=3,
    layer_to_nodes={
        0: [node0, node1, node2, node3, node4, node5],
        1: [node6, node7, node8],
        2: [node9, node10],
        3: [node11],
    }
)
```

### Visual Representation
```
                    Layer 3 (Root)
                        [11]
                    "Complete fairy tale about a princess,
                     dragon, and happy ending"
                       /         \
                      /           \
              Layer 2             
              [9]                 [10]
        "Princess vs           "Happy
         dragon story"          conclusion"
           /     \                  |
          /       \                 |
      Layer 1                       
      [6]         [7]             [8]
  "Princess    "Dragon        "Celebration
   with         conflict       and ending"
   friends"     resolved"
    / \          / \              / \
   /   \        /   \            /   \
Layer 0 (Leaves)
[0]   [1]     [2]   [3]        [4]   [5]
"Once  "She    "One   "The      "The   "They
upon   had     day    princess  kingdom all
a      three   dragon worked    cele-  lived
time"  loyal   appeared" together" brated" happily"
       friends"
```

## 8. Retrieval Data Flow

### Query Processing

```
+--------------------------------------------------------------------+
|                        INPUT: Question                              |
|                                                                     |
| "How did the princess defeat the dragon?"                          |
+--------------------------------------------------------------------+
                                    |
                                    v
+====================================================================+
|                      RETRIEVAL PROCESSING                           |
|                                                                     |
|  1. EMBED QUERY  -->  2. FIND SIMILAR  -->  3. BUILD CONTEXT       |
+====================================================================+
                                    |
                                    v
+--------------------------------------------------------------------+
|                        OUTPUT: Context + Answer                     |
|                                                                     |
| Context: "The princess and her friends worked together to defeat   |
|           the dragon and save everyone. She had three loyal        |
|           friends: a wise owl, a brave knight, and a magical       |
|           fairy."                                                  |
|                                                                     |
| Answer: "The princess defeated the dragon by working together      |
|          with her three loyal friends: a wise owl, a brave         |
|          knight, and a magical fairy."                             |
+--------------------------------------------------------------------+
```

### Collapsed Tree Mode (Default)

```python
# tree_retriever.py:158-195

# Step 1: Embed query
query = "How did the princess defeat the dragon?"
query_embedding = embedding_model.create_embedding(query)
# Shape: (1536,)

# Step 2: Get ALL node embeddings
all_nodes = [node0, node1, ..., node11]  # All 12 nodes
embeddings = [node.embeddings["OpenAI"] for node in all_nodes]
# Shape: (12, 1536)

# Step 3: Calculate distances
distances = [cosine_distance(query_embedding, emb) for emb in embeddings]
# Result: [0.45, 0.52, 0.31, 0.28, 0.67, 0.71, 0.42, 0.25, 0.58, 0.33, 0.61, 0.44]
#          n0    n1    n2    n3    n4    n5    n6    n7    n8    n9    n10   n11

# Step 4: Sort by distance (ascending = most similar first)
sorted_indices = [7, 3, 2, 9, 6, 11, 0, 1, 8, 10, 4, 5]
# Node 7 ("Dragon conflict resolved...") is most similar
# Node 3 ("The princess and her friends...") is second

# Step 5: Select nodes until max_tokens (3500 default)
selected_nodes = []
total_tokens = 0
for idx in sorted_indices[:top_k]:  # top_k=10
    node = all_nodes[idx]
    tokens = len(tokenizer.encode(node.text))
    if total_tokens + tokens > max_tokens:
        break
    selected_nodes.append(node)
    total_tokens += tokens

# Step 6: Build context
context = get_text(selected_nodes)
```

### Tree Traversal Mode

```python
# tree_retriever.py:197-250

# Start at top layer (layer 3)
current_nodes = layer_to_nodes[3]  # [node11]

for layer in range(num_layers):  # e.g., 3 layers
    # Find top_k most similar at this layer
    embeddings = get_embeddings(current_nodes, "OpenAI")
    distances = distances_from_embeddings(query_embedding, embeddings)
    
    # Select best nodes
    if selection_mode == "top_k":
        best_indices = indices[:top_k]
    elif selection_mode == "threshold":
        best_indices = [i for i in indices if distances[i] > threshold]
    
    selected_nodes.extend([current_nodes[i] for i in best_indices])
    
    # Get children for next layer
    if layer != num_layers - 1:
        child_indices = set()
        for idx in best_indices:
            child_indices.update(current_nodes[idx].children)
        current_nodes = [all_nodes[i] for i in child_indices]
```

## 9. Complete Data Transformation Summary

| Stage | Input | Output | Transformation |
|-------|-------|--------|----------------|
| 1. Chunking | `str` (1 doc) | `List[str]` (N chunks) | Split by sentences, max_tokens |
| 2. Embedding | `List[str]` | `Dict[int, Node]` | Add 1536-dim vectors |
| 3. Clustering | `List[Node]` | `List[List[Node]]` | UMAP + GMM grouping |
| 4. Summarizing | `List[List[Node]]` | `Dict[int, Node]` | LLM compress + embed |
| 5. Layer Build | Repeat 3-4 | Tree structure | Until stop condition |
| 6. Query | `str` question | `(1536,)` vector | Embed query |
| 7. Search | Query + Tree | `List[Node]` | Distance ranking |
| 8. Context | `List[Node]` | `str` | Concatenate text |
| 9. Answer | Context + Query | `str` | LLM QA generation |

## 10. Memory & Storage Footprint

```
Example: 10,000 word document

Chunking:
  - 10,000 words / ~75 words per chunk = ~133 chunks
  
Embedding:
  - 133 nodes * 1536 floats * 4 bytes = ~818 KB embeddings
  - Plus text storage: ~10,000 words * 5 chars = ~50 KB
  
Layer 1 (after clustering):
  - ~133/4 = ~33 summary nodes
  - ~33 * 1536 * 4 = ~203 KB embeddings
  - ~33 * 100 words = ~3.3 KB text
  
Layer 2:
  - ~33/4 = ~8 summary nodes
  - ~8 * 1536 * 4 = ~49 KB embeddings
  
Layer 3:
  - ~8/4 = ~2 summary nodes
  
Total nodes: 133 + 33 + 8 + 2 = 176 nodes
Total embeddings: ~1.1 MB
Total text: ~55 KB
Tree metadata: ~10 KB

TOTAL: ~1.2 MB for 10,000 word document
```

## 11. Timing Breakdown (Typical)

```
Operation                    | Time (approx)
-----------------------------|---------------
Text chunking                | <100ms
Leaf node creation (133)     | 5-10s (API calls)
Layer 1 clustering           | 1-2s
Layer 1 summarization (33)   | 15-30s (API calls)
Layer 2 clustering           | <1s
Layer 2 summarization (8)    | 5-10s
Layer 3 clustering           | <1s  
Layer 3 summarization (2)    | 2-4s
-----------------------------|---------------
Total indexing               | 30-60s

Query embedding              | 100-200ms
Distance calculation         | <50ms
Context building             | <10ms
QA generation               | 1-3s
-----------------------------|---------------
Total retrieval             | 1-4s
```
