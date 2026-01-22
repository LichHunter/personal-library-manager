# RAPTOR Retrieval Algorithms

> Detailed analysis of the retrieval strategies used in RAPTOR for querying the hierarchical tree.

## Overview

RAPTOR implements two distinct retrieval strategies:
1. **Collapsed Tree Retrieval** - Searches all nodes as a flat set
2. **Tree Traversal Retrieval** - Navigates tree layer by layer

Source file: `tree_retriever.py` (lines 1-328)

---

## Architecture Overview

```
                    Query
                      |
                      v
              +---------------+
              | create_embedding|
              +---------------+
                      |
                      v
         +------------------------+
         |    collapse_tree?      |
         +------------------------+
            |                |
           Yes               No
            |                |
            v                v
+------------------+  +------------------+
| Collapsed Tree   |  | Tree Traversal   |
| Retrieval        |  | Retrieval        |
+------------------+  +------------------+
            |                |
            v                v
    Search ALL nodes   Start at top layer
            |                |
            v                v
    Top-K by distance  Select top-K/threshold
            |                |
            |                v
            |          Traverse to children
            |                |
            |                v
            |          Repeat for num_layers
            |                |
            +-------+--------+
                    |
                    v
            Selected Nodes
                    |
                    v
            +---------------+
            |  get_text()   |
            +---------------+
                    |
                    v
              Context String
```

---

## 1. Distance Calculation

### Cosine Distance Function

```python
# utils.py:103-136
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances
```

### Cosine Distance Formula

Cosine distance measures the angle between two vectors:

$$d_{cosine}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$$

Where:
- $u \cdot v = \sum_{i=1}^{n} u_i v_i$ (dot product)
- $\|u\| = \sqrt{\sum_{i=1}^{n} u_i^2}$ (L2 norm)

**Range:** [0, 2] where:
- 0 = identical direction (most similar)
- 1 = orthogonal (unrelated)
- 2 = opposite direction (most dissimilar)

### Nearest Neighbor Sorting

```python
# utils.py:198-208
def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)
```

Returns indices sorted by ascending distance (nearest first).

---

## 2. Collapsed Tree Retrieval

### Concept

Collapsed tree retrieval treats the entire tree as a flat collection of nodes, ignoring the hierarchical structure. This is the **default mode** in RAPTOR.

```
         [Root Layer]
              |
    +----+----+----+
    |    |    |    |
[Layer 1 Summaries]
    |    |    |    |
+--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |
[   Leaf Nodes      ]

        ||
        vv (collapse)

[All Nodes in Single Pool]
     compare with query
     select top-K
```

### Implementation

```python
# tree_retriever.py:158-195
def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
    # Step 1: Embed the query
    query_embedding = self.create_embedding(query)

    selected_nodes = []

    # Step 2: Get ALL nodes from tree
    node_list = get_node_list(self.tree.all_nodes)

    # Step 3: Extract embeddings
    embeddings = get_embeddings(node_list, self.context_embedding_model)

    # Step 4: Calculate distances
    distances = distances_from_embeddings(query_embedding, embeddings)

    # Step 5: Sort by distance (ascending)
    indices = indices_of_nearest_neighbors_from_distances(distances)

    # Step 6: Select top-K within token budget
    total_tokens = 0
    for idx in indices[:top_k]:
        node = node_list[idx]
        node_tokens = len(self.tokenizer.encode(node.text))

        if total_tokens + node_tokens > max_tokens:
            break

        selected_nodes.append(node)
        total_tokens += node_tokens

    # Step 7: Build context string
    context = get_text(selected_nodes)
    return selected_nodes, context
```

### Pseudocode

```
function COLLAPSED_TREE_RETRIEVAL(query, tree, top_k, max_tokens):
    query_embedding = embed(query)
    
    // Get all nodes from all layers
    all_nodes = tree.all_nodes.values()
    
    // Calculate distance to each node
    distances = []
    for node in all_nodes:
        dist = cosine_distance(query_embedding, node.embedding)
        distances.append((node, dist))
    
    // Sort by distance (ascending = most similar first)
    sorted_nodes = sort_by_distance(distances)
    
    // Select nodes within budget
    selected = []
    total_tokens = 0
    
    for (node, dist) in sorted_nodes[:top_k]:
        tokens = count_tokens(node.text)
        if total_tokens + tokens > max_tokens:
            break
        selected.append(node)
        total_tokens += tokens
    
    // Build context
    context = concatenate([node.text for node in selected])
    return context
```

### Characteristics

| Aspect | Value |
|--------|-------|
| Search space | All nodes (leaves + summaries) |
| Complexity | O(N) where N = total nodes |
| Hierarchy awareness | None |
| Best for | Broad queries, unknown specificity |

---

## 3. Tree Traversal Retrieval

### Concept

Tree traversal starts at a specified layer (typically root) and navigates down through children of selected nodes.

```
Query
  |
  v
[Root Layer] -----> Select top-K
      |
      v
[Layer 1] ------> Get children of selected
      |            Select top-K from children
      v
[Leaf Layer] ---> Get children of selected
      |            Select top-K from children
      v
Selected Leaf Nodes
```

### Implementation

```python
# tree_retriever.py:197-250
def retrieve_information(
    self, current_nodes: List[Node], query: str, num_layers: int
) -> str:
    query_embedding = self.create_embedding(query)
    selected_nodes = []
    node_list = current_nodes

    for layer in range(num_layers):
        # Get embeddings for current layer
        embeddings = get_embeddings(node_list, self.context_embedding_model)
        
        # Calculate distances
        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        # Select based on mode
        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]
        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        # Add selected nodes
        nodes_to_add = [node_list[idx] for idx in best_indices]
        selected_nodes.extend(nodes_to_add)

        # Prepare next layer (except for last iteration)
        if layer != num_layers - 1:
            child_nodes = []
            for index in best_indices:
                child_nodes.extend(node_list[index].children)
            
            # Deduplicate
            child_nodes = list(dict.fromkeys(child_nodes))
            node_list = [self.tree.all_nodes[i] for i in child_nodes]

    context = get_text(selected_nodes)
    return selected_nodes, context
```

### Pseudocode

```
function TREE_TRAVERSAL_RETRIEVAL(query, tree, start_layer, num_layers, mode, k, threshold):
    query_embedding = embed(query)
    selected_nodes = []
    
    // Start at specified layer
    current_nodes = tree.layer_to_nodes[start_layer]
    
    for layer in range(num_layers):
        // Calculate distances for current layer
        distances = []
        for node in current_nodes:
            dist = cosine_distance(query_embedding, node.embedding)
            distances.append((node, dist))
        
        // Sort by distance
        sorted_pairs = sort_by_distance(distances)
        
        // Select based on mode
        if mode == "top_k":
            best_nodes = [node for (node, _) in sorted_pairs[:k]]
        else:  // threshold mode
            best_nodes = [node for (node, dist) in sorted_pairs if dist < threshold]
        
        // Add to results
        selected_nodes.extend(best_nodes)
        
        // Prepare next layer (get children)
        if layer < num_layers - 1:
            child_indices = set()
            for node in best_nodes:
                child_indices.update(node.children)
            current_nodes = [tree.all_nodes[i] for i in child_indices]
    
    context = concatenate([node.text for node in selected_nodes])
    return context
```

### Traversal Visualization

```
Example with top_k=2, num_layers=3:

Layer 2 (Root):    [R1, R2, R3]
                       |
                    top-2
                       v
                   [R1, R3] selected
                       |
                    children
                       v
Layer 1:           [S1, S2, S3, S4]
                       |
                    top-2
                       v
                   [S1, S4] selected
                       |
                    children
                       v
Layer 0 (Leaves):  [L1, L2, L5, L6, L7]
                       |
                    top-2
                       v
                   [L2, L6] selected

Final Selection: [R1, R3, S1, S4, L2, L6]
```

### Characteristics

| Aspect | Value |
|--------|-------|
| Search space | Subset of tree (traversal path) |
| Complexity | O(k * L) where k=top_k, L=num_layers |
| Hierarchy awareness | Full |
| Best for | Specific queries, known topic area |

---

## 4. Selection Modes

### Top-K Mode (Default)

Always selects exactly K nodes with smallest distances:

```python
# tree_retriever.py:231-232
elif self.selection_mode == "top_k":
    best_indices = indices[: self.top_k]
```

**Behavior:**
- Deterministic number of results
- May include low-relevance nodes if query is unusual
- Default: `top_k = 5`

### Threshold Mode

Selects nodes where distance is below a threshold:

```python
# tree_retriever.py:226-229
if self.selection_mode == "threshold":
    best_indices = [
        index for index in indices if distances[index] > self.threshold
    ]
```

**Note:** The code uses `> threshold` which appears to be a bug - cosine distance is lower for similar items, so it should likely be `< threshold` for similarity-based selection. However, this is the actual implementation.

**Behavior:**
- Variable number of results
- May return 0 nodes if nothing is relevant
- May return many nodes if query is broad
- Default: `threshold = 0.5`

### Comparison

```
Query: "machine learning algorithms"

Top-K (k=3):
  Node 1: dist=0.15 [selected]
  Node 2: dist=0.22 [selected]
  Node 3: dist=0.45 [selected]  <- may be irrelevant
  Node 4: dist=0.78

Threshold (t=0.3):
  Node 1: dist=0.15 [selected]
  Node 2: dist=0.22 [selected]
  Node 3: dist=0.45  <- not selected
  Node 4: dist=0.78
```

---

## 5. Context Building

### get_text Function

```python
# utils.py:181-195
def get_text(node_list: List[Node]) -> str:
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text
```

Concatenates node texts with double newlines as separators.

### Example Output

```
Input nodes: [
  Node(text="ML enables learning from data."),
  Node(text="Deep learning uses neural networks."),
  Node(text="AI encompasses multiple fields.")
]

Output context:
"ML enables learning from data.

Deep learning uses neural networks.

AI encompasses multiple fields.

"
```

---

## 6. Main Retrieve Interface

### Function Signature

```python
# tree_retriever.py:252-327
def retrieve(
    self,
    query: str,
    start_layer: int = None,
    num_layers: int = None,
    top_k: int = 10,
    max_tokens: int = 3500,
    collapse_tree: bool = True,
    return_layer_information: bool = False,
) -> str:
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | (required) | The search query string |
| `start_layer` | `tree.num_layers` | Layer to start traversal (ignored if collapse_tree=True) |
| `num_layers` | `tree.num_layers + 1` | Number of layers to traverse |
| `top_k` | 10 | Maximum nodes to select |
| `max_tokens` | 3500 | Token budget for context |
| `collapse_tree` | True | Use collapsed vs traversal retrieval |
| `return_layer_information` | False | Include layer metadata in response |

### Decision Flow

```python
# tree_retriever.py:302-311
if collapse_tree:
    logging.info(f"Using collapsed_tree")
    selected_nodes, context = self.retrieve_information_collapse_tree(
        query, top_k, max_tokens
    )
else:
    layer_nodes = self.tree.layer_to_nodes[start_layer]
    selected_nodes, context = self.retrieve_information(
        layer_nodes, query, num_layers
    )
```

### Complete Pseudocode

```
function RETRIEVE(query, config):
    // Validate inputs
    assert query is string
    assert max_tokens >= 1
    
    // Set defaults
    start_layer = config.start_layer or tree.num_layers
    num_layers = config.num_layers or tree.num_layers + 1
    
    // Validate layer configuration
    assert start_layer <= tree.num_layers
    assert num_layers <= start_layer + 1
    
    // Choose retrieval strategy
    if config.collapse_tree:
        nodes, context = COLLAPSED_TREE_RETRIEVAL(
            query, 
            tree, 
            config.top_k, 
            config.max_tokens
        )
    else:
        initial_nodes = tree.layer_to_nodes[start_layer]
        nodes, context = TREE_TRAVERSAL_RETRIEVAL(
            query,
            initial_nodes,
            num_layers,
            config.selection_mode,
            config.top_k,
            config.threshold
        )
    
    // Optional: include layer information
    if config.return_layer_information:
        layer_info = []
        for node in nodes:
            layer_info.append({
                "node_index": node.index,
                "layer_number": node_to_layer[node.index]
            })
        return context, layer_info
    
    return context
```

---

## 7. When to Use Each Strategy

### Collapsed Tree Retrieval

**Best for:**
- General/broad queries
- Unknown query specificity
- Maximum recall needed
- Simple implementation

**Example queries:**
- "What is this document about?"
- "Summarize the main themes"
- "Find all mentions of X"

### Tree Traversal Retrieval

**Best for:**
- Specific/detailed queries
- Hierarchical exploration
- Controlled granularity
- Memory-constrained environments

**Example queries:**
- "What specific algorithm does section 3 describe?"
- "Find the implementation details of feature X"
- "What are the sub-components of system Y?"

### Hybrid Approach

Not implemented in RAPTOR, but possible:
1. Start with collapsed tree to identify relevant areas
2. Use tree traversal to dive deeper into those areas

---

## 8. Configuration

### TreeRetrieverConfig

```python
# tree_retriever.py:19-103
class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,          # Default: tiktoken cl100k_base
        threshold=None,          # Default: 0.5
        top_k=None,              # Default: 5
        selection_mode=None,     # Default: "top_k"
        context_embedding_model=None,  # Default: "OpenAI"
        embedding_model=None,    # Default: OpenAIEmbeddingModel()
        num_layers=None,         # Default: tree.num_layers + 1
        start_layer=None,        # Default: tree.num_layers
    ):
```

### Key Constraints

```python
# tree_retriever.py:112-131
# num_layers cannot exceed tree depth + 1
if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
    raise ValueError(...)

# start_layer cannot exceed tree depth
if config.start_layer is not None and config.start_layer > tree.num_layers:
    raise ValueError(...)

# num_layers must be <= start_layer + 1
if self.num_layers > self.start_layer + 1:
    raise ValueError("num_layers must be less than or equal to start_layer + 1")
```

---

## 9. Source Code References

| Function | File | Lines |
|----------|------|-------|
| `distances_from_embeddings` | utils.py | 103-136 |
| `indices_of_nearest_neighbors_from_distances` | utils.py | 198-208 |
| `get_text` | utils.py | 181-195 |
| `TreeRetrieverConfig` | tree_retriever.py | 19-103 |
| `TreeRetriever.__init__` | tree_retriever.py | 106-144 |
| `retrieve_information_collapse_tree` | tree_retriever.py | 158-195 |
| `retrieve_information` (traversal) | tree_retriever.py | 197-250 |
| `retrieve` (main interface) | tree_retriever.py | 252-327 |
