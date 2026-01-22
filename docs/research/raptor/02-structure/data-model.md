# RAPTOR Data Model

## Core Data Structures

### Node Class

**Location:** `tree_structures.py:4`

The `Node` class represents a single node in the hierarchical tree structure. Nodes can be either leaf nodes (containing original text chunks) or parent nodes (containing summarized text from children).

```python
class Node:
    def __init__(
        self, 
        text: str,           # The text content of this node
        index: int,          # Unique identifier within the tree
        children: Set[int],  # Set of child node indices (empty for leaf nodes)
        embeddings           # Dict mapping model names to embedding vectors
    ) -> None
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The text content. For leaf nodes: original document chunk. For parent nodes: LLM-generated summary of children's text. |
| `index` | `int` | Unique integer identifier within the tree. Indices are sequential starting from 0. |
| `children` | `Set[int]` | Set of indices pointing to child nodes. Empty set `{}` for leaf nodes. |
| `embeddings` | `Dict[str, List[float]]` | Dictionary mapping embedding model names (e.g., "OpenAI") to embedding vectors. Supports multiple embedding models simultaneously. |

#### Node Types

1. **Leaf Nodes** (Layer 0)
   - Created from original text chunks
   - `children = set()` (empty)
   - `text` contains original document content

2. **Parent Nodes** (Layer 1+)
   - Created by summarizing clusters of child nodes
   - `children` contains indices of the nodes that were summarized
   - `text` contains the LLM-generated summary

### Tree Class

**Location:** `tree_structures.py:16`

The `Tree` class represents the complete hierarchical tree structure built from documents.

```python
class Tree:
    def __init__(
        self,
        all_nodes,          # Dict[int, Node] - all nodes in the tree
        root_nodes,         # Dict[int, Node] - top-level nodes
        leaf_nodes,         # Dict[int, Node] - bottom-level nodes
        num_layers,         # int - number of layers in the tree
        layer_to_nodes      # Dict[int, List[Node]] - layer number to nodes mapping
    ) -> None
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `all_nodes` | `Dict[int, Node]` | Complete dictionary of all nodes, keyed by node index. |
| `root_nodes` | `Dict[int, Node]` | Dictionary of top-level summary nodes (highest abstraction level). |
| `leaf_nodes` | `Dict[int, Node]` | Dictionary of leaf nodes (original text chunks). |
| `num_layers` | `int` | Total number of layers in the tree (0-indexed). Layer 0 is leaves. |
| `layer_to_nodes` | `Dict[int, List[Node]]` | Maps layer number to list of nodes at that layer. |

## Configuration Classes

### TreeBuilderConfig

**Location:** `tree_builder.py:24`

Configuration for the base TreeBuilder class.

| Parameter | Type | Default | Validation | Description |
|-----------|------|---------|------------|-------------|
| `tokenizer` | `tiktoken.Encoding` | `cl100k_base` | - | Tokenizer for text splitting |
| `max_tokens` | `int` | `100` | `>= 1` | Maximum tokens per text chunk |
| `num_layers` | `int` | `5` | `>= 1` | Maximum number of tree layers |
| `threshold` | `float` | `0.5` | `0 <= x <= 1` | Similarity threshold for node selection |
| `top_k` | `int` | `5` | `>= 1` | Number of top nodes to select |
| `selection_mode` | `str` | `"top_k"` | `"top_k"` or `"threshold"` | Node selection strategy |
| `summarization_length` | `int` | `100` | - | Max tokens for generated summaries |
| `summarization_model` | `BaseSummarizationModel` | `GPT3TurboSummarizationModel()` | Must be `BaseSummarizationModel` | Model for generating summaries |
| `embedding_models` | `Dict[str, BaseEmbeddingModel]` | `{"OpenAI": OpenAIEmbeddingModel()}` | All values must be `BaseEmbeddingModel` | Named embedding models |
| `cluster_embedding_model` | `str` | `"OpenAI"` | Must exist in `embedding_models` | Which embedding model to use for clustering |

### ClusterTreeConfig

**Location:** `cluster_tree_builder.py:17`

Extends `TreeBuilderConfig` with clustering-specific parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction_dimension` | `int` | `10` | Target dimensions for UMAP reduction |
| `clustering_algorithm` | `type` | `RAPTOR_Clustering` | Clustering algorithm class to use |
| `clustering_params` | `dict` | `{}` | Additional parameters passed to clustering |

### TreeRetrieverConfig

**Location:** `tree_retriever.py:19`

Configuration for the TreeRetriever class.

| Parameter | Type | Default | Validation | Description |
|-----------|------|---------|------------|-------------|
| `tokenizer` | `tiktoken.Encoding` | `cl100k_base` | - | Tokenizer for token counting |
| `threshold` | `float` | `0.5` | `0 <= x <= 1` | Similarity threshold |
| `top_k` | `int` | `5` | `>= 1` | Number of top nodes to retrieve |
| `selection_mode` | `str` | `"top_k"` | `"top_k"` or `"threshold"` | Selection strategy |
| `context_embedding_model` | `str` | `"OpenAI"` | Must be string | Name of embedding model for context |
| `embedding_model` | `BaseEmbeddingModel` | `OpenAIEmbeddingModel()` | Must be `BaseEmbeddingModel` | Model for query embedding |
| `num_layers` | `int \| None` | `None` | `>= 0` if set | Number of layers to traverse |
| `start_layer` | `int \| None` | `None` | `>= 0` if set | Layer to start traversal from |

### FaissRetrieverConfig

**Location:** `FaissRetriever.py:14`

Configuration for FAISS-based flat retrieval.

| Parameter | Type | Default | Validation | Description |
|-----------|------|---------|------------|-------------|
| `max_tokens` | `int` | `100` | `>= 1` | Max tokens per chunk |
| `max_context_tokens` | `int` | `3500` | `>= 1` or `None` | Max total context tokens |
| `use_top_k` | `bool` | `False` | - | Use top_k vs token limit |
| `embedding_model` | `BaseEmbeddingModel` | `OpenAIEmbeddingModel()` | Must be `BaseEmbeddingModel` | Embedding model for chunks |
| `question_embedding_model` | `BaseEmbeddingModel` | Same as `embedding_model` | Must be `BaseEmbeddingModel` | Embedding model for queries |
| `top_k` | `int` | `5` | `>= 1` | Number of results |
| `tokenizer` | `tiktoken.Encoding` | `cl100k_base` | - | Tokenizer |
| `embedding_model_string` | `str` | `"OpenAI"` | - | Key for node embeddings |

### RetrievalAugmentationConfig

**Location:** `RetrievalAugmentation.py:18`

Master configuration for the complete system.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tree_builder_config` | `TreeBuilderConfig` | Auto-created | Tree builder configuration |
| `tree_retriever_config` | `TreeRetrieverConfig` | Auto-created | Tree retriever configuration |
| `qa_model` | `BaseQAModel` | `GPT3TurboQAModel()` | Question-answering model |
| `embedding_model` | `BaseEmbeddingModel` | `None` | Shared embedding model (convenience) |
| `summarization_model` | `BaseSummarizationModel` | `None` | Shared summarization model (convenience) |
| `tree_builder_type` | `str` | `"cluster"` | Type of tree builder |
| `tr_*` | various | - | TreeRetriever config params |
| `tb_*` | various | - | TreeBuilder config params |

## Runtime Data Examples

### Example: Small Document Tree

Given input text of ~500 tokens, split into 5 chunks:

```
TREE STRUCTURE (num_layers = 2)
===============================

                    Layer 2 (Root)
                    +-------------+
                    | Node 7      |
                    | "Summary of |
                    | entire doc" |
                    | children:   |
                    | {5, 6}      |
                    +-------------+
                          |
           +--------------+--------------+
           |                             |
      Layer 1                       Layer 1
    +-------------+              +-------------+
    | Node 5      |              | Node 6      |
    | "Summary of |              | "Summary of |
    | nodes 0,1,2"|              | nodes 3,4"  |
    | children:   |              | children:   |
    | {0, 1, 2}   |              | {3, 4}      |
    +-------------+              +-------------+
           |                             |
    +------+------+               +------+------+
    |      |      |               |             |
 +----+ +----+ +----+          +----+       +----+
 |N 0 | |N 1 | |N 2 |          |N 3 |       |N 4 |
 |text| |text| |text|          |text|       |text|
 |"..." |"..." |"..." |        |"..." |     |"..." |
 |ch:{}| |ch:{}| |ch:{}|       |ch:{}|      |ch:{}|
 +----+ +----+ +----+          +----+       +----+
 
    Layer 0 (Leaves) - Original text chunks
```

### Example: Node Data

```python
# Leaf Node (Layer 0)
Node(
    text="The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    index=0,
    children=set(),  # Empty - this is a leaf
    embeddings={
        "OpenAI": [0.023, -0.041, 0.089, ..., 0.012],  # 1536 dimensions
        "SBert": [0.156, 0.023, -0.089, ..., 0.045]    # 768 dimensions (if configured)
    }
)

# Parent Node (Layer 1)
Node(
    text="This passage discusses a common pangram sentence and its properties.",
    index=5,
    children={0, 1, 2},  # References leaf nodes 0, 1, 2
    embeddings={
        "OpenAI": [0.045, -0.023, 0.067, ..., 0.034],
        "SBert": [0.189, 0.056, -0.012, ..., 0.078]
    }
)
```

### Example: Tree Data

```python
Tree(
    all_nodes={
        0: Node(text="chunk 0...", index=0, children=set(), embeddings=...),
        1: Node(text="chunk 1...", index=1, children=set(), embeddings=...),
        2: Node(text="chunk 2...", index=2, children=set(), embeddings=...),
        3: Node(text="chunk 3...", index=3, children=set(), embeddings=...),
        4: Node(text="chunk 4...", index=4, children=set(), embeddings=...),
        5: Node(text="summary of 0,1,2", index=5, children={0,1,2}, embeddings=...),
        6: Node(text="summary of 3,4", index=6, children={3,4}, embeddings=...),
        7: Node(text="root summary", index=7, children={5,6}, embeddings=...),
    },
    root_nodes={
        7: Node(text="root summary", index=7, children={5,6}, embeddings=...)
    },
    leaf_nodes={
        0: Node(text="chunk 0...", index=0, children=set(), embeddings=...),
        1: Node(text="chunk 1...", index=1, children=set(), embeddings=...),
        2: Node(text="chunk 2...", index=2, children=set(), embeddings=...),
        3: Node(text="chunk 3...", index=3, children=set(), embeddings=...),
        4: Node(text="chunk 4...", index=4, children=set(), embeddings=...),
    },
    num_layers=2,
    layer_to_nodes={
        0: [Node_0, Node_1, Node_2, Node_3, Node_4],  # Leaves
        1: [Node_5, Node_6],                          # First summaries
        2: [Node_7],                                  # Root
    }
)
```

### Example: Clustering Output

During tree construction, RAPTOR_Clustering produces cluster assignments:

```python
# Input: 5 leaf nodes with embeddings
# Output: List of clusters, each cluster is a list of nodes

clusters = [
    [Node_0, Node_1, Node_2],  # Cluster 0: semantically similar nodes
    [Node_3, Node_4],          # Cluster 1: another group
]

# Each cluster will be summarized into a single parent node
# Node 5 summarizes cluster 0
# Node 6 summarizes cluster 1
```

### Example: Retrieval Results

```python
# Query: "What is a pangram?"
# Using collapse_tree mode (searches all nodes)

selected_nodes = [
    Node_5,  # "This passage discusses a common pangram sentence..."
    Node_0,  # "The quick brown fox jumps over the lazy dog..."
    Node_7,  # Root summary mentioning pangrams
]

context = """
This passage discusses a common pangram sentence and its properties.

The quick brown fox jumps over the lazy dog. This sentence contains every letter...

[Root summary text...]
"""

layer_information = [
    {"node_index": 5, "layer_number": 1},
    {"node_index": 0, "layer_number": 0},
    {"node_index": 7, "layer_number": 2},
]
```

## Data Flow Through System

```
                    INPUT TEXT
                        |
                        v
            +------------------------+
            |     split_text()       |
            |  utils.py:22           |
            +------------------------+
                        |
                        v
            List[str] (text chunks)
                        |
                        v
            +------------------------+
            | create_node()          |
            | For each chunk:        |
            |  - Generate embeddings |
            |  - Create Node         |
            +------------------------+
                        |
                        v
            Dict[int, Node] (leaf_nodes)
                        |
                        v
    +-------------------------------------------+
    |          CLUSTERING LOOP                   |
    |                                           |
    |   For each layer until convergence:       |
    |   1. Extract embeddings from nodes        |
    |   2. UMAP dimensionality reduction        |
    |   3. GMM clustering                       |
    |   4. For each cluster:                    |
    |      - Concatenate node texts             |
    |      - Summarize with LLM                 |
    |      - Create parent Node                 |
    |   5. Update layer_to_nodes                |
    +-------------------------------------------+
                        |
                        v
                     Tree
                        |
            +-----------+-----------+
            |                       |
            v                       v
    +-------------+         +-------------+
    | pickle.dump |         | TreeRetriever|
    | (persist)   |         | (query)      |
    +-------------+         +-------------+
```

## Embedding Storage

Nodes store embeddings as a dictionary to support multiple embedding models:

```python
embeddings = {
    "OpenAI": [0.023, -0.041, ...],  # 1536 dims (text-embedding-ada-002)
    "SBert": [0.156, 0.023, ...],    # 768 dims (multi-qa-mpnet-base-cos-v1)
}
```

This allows:
1. Different models for clustering vs retrieval
2. Easy addition of new embedding models
3. Comparison of different embedding approaches

## Index Conventions

- **Node indices**: Sequential integers starting at 0
- **Layer indices**: 0 = leaf layer (bottom), higher = more abstract
- **Children references**: Stored as `Set[int]` of node indices, not direct references

This integer-based referencing:
1. Enables easy serialization (pickle)
2. Avoids circular reference issues
3. Allows O(1) node lookup via `all_nodes[index]`
