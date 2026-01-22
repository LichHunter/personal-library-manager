# RAPTOR Tree Building Algorithm

> Detailed analysis of the hierarchical tree construction process in RAPTOR.

## Overview

The tree building algorithm transforms raw text into a hierarchical abstraction tree by:
1. Splitting text into chunks
2. Creating leaf nodes with embeddings
3. Iteratively clustering and summarizing to build higher layers

Primary source files:
- `utils.py` - Text splitting (lines 22-100)
- `tree_builder.py` - Base tree construction (lines 133-370)
- `cluster_tree_builder.py` - Cluster-based construction (lines 41-151)

---

## High-Level Architecture

```
Raw Text Document
        |
        v
+------------------+
|   split_text()   |  <- Sentence-boundary splitting
+------------------+
        |
        v
[Chunk_0, Chunk_1, ..., Chunk_n]
        |
        v
+------------------+
| create_node()    |  <- Generate embeddings for each chunk
| (multithreaded)  |
+------------------+
        |
        v
Layer 0: [Node_0, Node_1, ..., Node_n]  (Leaf Nodes)
        |
        v
+------------------+
| construct_tree() |  <- Main loop
+------------------+
        |
   +----+----+
   |         |
   v         v
Cluster   Summarize
Nodes     Clusters
   |         |
   +----+----+
        |
        v
Layer 1: [Summary_0, Summary_1, ...]
        |
        v
    (repeat until stop condition)
        |
        v
Layer N: [Root Nodes]
```

---

## 1. Text Splitting Algorithm

### Function Signature

```python
# utils.py:22-100
def split_text(
    text: str, 
    tokenizer: tiktoken.get_encoding("cl100k_base"), 
    max_tokens: int, 
    overlap: int = 0
):
```

### Algorithm Description

The splitting algorithm respects sentence boundaries while ensuring chunks don't exceed `max_tokens`:

```
function SPLIT_TEXT(text, tokenizer, max_tokens, overlap=0):
    // Step 1: Split by sentence delimiters
    delimiters = [".", "!", "?", "\n"]
    sentences = regex_split(text, delimiters)
    
    // Step 2: Count tokens per sentence
    token_counts = [tokenizer.encode(" " + s).length for s in sentences]
    
    // Step 3: Build chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for (sentence, token_count) in zip(sentences, token_counts):
        if sentence.strip() == "":
            continue  // Skip empty sentences
        
        if token_count > max_tokens:
            // Handle oversized sentence - split by sub-delimiters
            sub_sentences = regex_split(sentence, [",", ";", ":"])
            chunks.extend(SPLIT_SUBSECTIONS(sub_sentences, max_tokens, overlap))
        
        else if current_length + token_count > max_tokens:
            // Current chunk is full - start new chunk
            chunks.append(join(current_chunk))
            current_chunk = last_n_elements(current_chunk, overlap)
            current_length = sum_tokens(current_chunk)
            current_chunk.append(sentence)
            current_length += token_count
        
        else:
            // Add to current chunk
            current_chunk.append(sentence)
            current_length += token_count
    
    // Don't forget the last chunk
    if current_chunk:
        chunks.append(join(current_chunk))
    
    return chunks
```

### Splitting Hierarchy

The algorithm uses a two-level delimiter hierarchy:

```
Level 1 Delimiters: . ! ? \n  (sentence boundaries)
         |
         v
    [Sentences]
         |
   If sentence > max_tokens:
         |
         v
Level 2 Delimiters: , ; :  (clause boundaries)
         |
         v
    [Sub-sentences]
```

### Example

```python
text = "Machine learning is powerful. It can analyze data quickly, make predictions, and learn patterns. NLP is a subfield."
max_tokens = 50  # hypothetical low limit

# Step 1: Split by sentence delimiters
sentences = [
    "Machine learning is powerful",
    " It can analyze data quickly, make predictions, and learn patterns",
    " NLP is a subfield"
]

# If middle sentence exceeds max_tokens, split further:
sub_sentences = [
    " It can analyze data quickly",
    " make predictions",
    " and learn patterns"
]
```

### Edge Cases

1. **Oversized sentences** (lines 55-81): Split by secondary delimiters
2. **Empty sentences** (lines 50-52): Skip whitespace-only strings
3. **Overlap handling** (lines 74, 86): Retain last N elements for context continuity

---

## 2. Node Creation

### Node Structure

```python
# tree_structures.py:4-13
class Node:
    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text        # The text content
        self.index = index      # Unique identifier
        self.children = children # Set of child node indices (empty for leaves)
        self.embeddings = embeddings  # Dict[model_name -> embedding_vector]
```

### Create Node Function

```python
# tree_builder.py:158-179
def create_node(
    self, index: int, text: str, children_indices: Optional[Set[int]] = None
) -> Tuple[int, Node]:
    if children_indices is None:
        children_indices = set()

    embeddings = {
        model_name: model.create_embedding(text)
        for model_name, model in self.embedding_models.items()
    }
    return (index, Node(text, index, children_indices, embeddings))
```

### Multithreaded Leaf Node Creation

```python
# tree_builder.py:238-258
def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
    with ThreadPoolExecutor() as executor:
        future_nodes = {
            executor.submit(self.create_node, index, text): (index, text)
            for index, text in enumerate(chunks)
        }

        leaf_nodes = {}
        for future in as_completed(future_nodes):
            index, node = future.result()
            leaf_nodes[index] = node

    return leaf_nodes
```

### Pseudocode

```
function CREATE_LEAF_NODES(chunks, embedding_models):
    leaf_nodes = {}
    
    // Parallel execution
    parallel for (index, text) in enumerate(chunks):
        embeddings = {}
        for (name, model) in embedding_models:
            embeddings[name] = model.create_embedding(text)
        
        node = Node(
            text=text,
            index=index,
            children=empty_set,
            embeddings=embeddings
        )
        leaf_nodes[index] = node
    
    return leaf_nodes
```

---

## 3. Tree Construction Loop

### Entry Point

```python
# tree_builder.py:260-295
def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
    # Split text into chunks
    chunks = split_text(text, self.tokenizer, self.max_tokens)
    
    # Create leaf nodes
    if use_multithreading:
        leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
    else:
        leaf_nodes = {i: create_node(i, t) for i, t in enumerate(chunks)}
    
    # Initialize layer tracking
    layer_to_nodes = {0: list(leaf_nodes.values())}
    all_nodes = copy.deepcopy(leaf_nodes)
    
    # Build tree hierarchy
    root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
    
    # Create final tree structure
    tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
    return tree
```

### ClusterTreeBuilder Construction

The main tree construction algorithm:

```python
# cluster_tree_builder.py:55-151
def construct_tree(
    self,
    current_level_nodes: Dict[int, Node],
    all_tree_nodes: Dict[int, Node],
    layer_to_nodes: Dict[int, List[Node]],
    use_multithreading: bool = False,
) -> Dict[int, Node]:
```

### Complete Algorithm Pseudocode

```
function CONSTRUCT_TREE(current_nodes, all_nodes, layer_to_nodes):
    next_node_index = len(all_nodes)
    
    for layer in range(num_layers):
        new_level_nodes = {}
        node_list = sorted_node_list(current_nodes)
        
        // STOP CONDITION 1: Not enough nodes for UMAP
        if len(node_list) <= reduction_dimension + 1:
            log("Stopping: Cannot create more layers")
            actual_num_layers = layer
            break
        
        // Step 1: Cluster current layer nodes
        clusters = RAPTOR_CLUSTERING(
            node_list,
            embedding_model_name,
            reduction_dimension
        )
        
        // Step 2: Summarize each cluster
        for cluster in clusters:
            // Get combined text from cluster
            combined_text = concatenate([node.text for node in cluster])
            
            // Generate summary via LLM
            summary = SUMMARIZE(combined_text, max_tokens=summarization_length)
            
            // Create parent node
            child_indices = {node.index for node in cluster}
            parent_node = CREATE_NODE(next_node_index, summary, child_indices)
            
            new_level_nodes[next_node_index] = parent_node
            next_node_index += 1
        
        // Step 3: Update tracking structures
        layer_to_nodes[layer + 1] = list(new_level_nodes.values())
        current_nodes = new_level_nodes
        all_nodes.update(new_level_nodes)
    
    return current_nodes  // Returns root nodes
```

### Cluster Processing Detail

```python
# cluster_tree_builder.py:66-86
def process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock):
    # Concatenate all node texts
    node_texts = get_text(cluster)

    # Generate summary via LLM
    summarized_text = self.summarize(
        context=node_texts,
        max_tokens=summarization_length,
    )

    # Create parent node with children references
    __, new_parent_node = self.create_node(
        next_node_index, summarized_text, {node.index for node in cluster}
    )

    # Thread-safe update
    with lock:
        new_level_nodes[next_node_index] = new_parent_node
```

---

## 4. Stop Conditions

The tree building loop terminates under these conditions:

### Condition 1: Maximum Layers Reached

```python
# cluster_tree_builder.py:87
for layer in range(self.num_layers):  # Default: num_layers = 5
```

The loop runs for at most `num_layers` iterations.

### Condition 2: Insufficient Nodes for Clustering

```python
# cluster_tree_builder.py:95-100
if len(node_list_current_layer) <= self.reduction_dimension + 1:
    self.num_layers = layer
    logging.info(
        f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
    )
    break
```

**Why `reduction_dimension + 1`?**
- UMAP requires at least `n_components + 2` data points to reduce dimensions
- With default `reduction_dimension = 10`, need at least 12 nodes
- Below this threshold, dimensionality reduction fails

### Condition 3: Implicit - Single Cluster

If clustering produces only one cluster containing all nodes, the next layer would be identical to current. In practice, the "insufficient nodes" condition catches this first.

### Stop Condition Flowchart

```
     Start Layer Construction
              |
              v
    +--------------------+
    | layer < num_layers?|
    +--------------------+
         |No        |Yes
         v          v
       STOP    +------------------+
               | nodes.count >    |
               | dim + 1?         |
               +------------------+
                  |No        |Yes
                  v          v
                STOP    Continue to
                        Clustering
```

---

## 5. Data Flow Example

### Input Document

```
"Machine learning enables computers to learn from data. 
Deep learning is a subset of ML using neural networks.
Natural language processing handles text and speech.
Computer vision analyzes images and video."
```

### Layer 0 (Leaf Nodes)

```
Node 0: "Machine learning enables computers to learn from data"
        embeddings: {OpenAI: [0.1, 0.2, ...]}
        children: {}

Node 1: "Deep learning is a subset of ML using neural networks"
        embeddings: {OpenAI: [0.15, 0.22, ...]}
        children: {}

Node 2: "Natural language processing handles text and speech"
        embeddings: {OpenAI: [0.3, 0.1, ...]}
        children: {}

Node 3: "Computer vision analyzes images and video"
        embeddings: {OpenAI: [0.25, 0.12, ...]}
        children: {}
```

### Clustering Result

```
Cluster 0: [Node 0, Node 1]  // ML-related
Cluster 1: [Node 2, Node 3]  // Application domains
```

### Layer 1 (Summaries)

```
Node 4: "Machine learning and deep learning enable computers 
         to learn patterns from data using neural networks."
        embeddings: {OpenAI: [0.12, 0.21, ...]}
        children: {0, 1}

Node 5: "NLP and computer vision are AI applications that
         process text, speech, images, and video."
        embeddings: {OpenAI: [0.27, 0.11, ...]}
        children: {2, 3}
```

### Layer 2 (Root)

```
Node 6: "AI encompasses machine learning techniques including
         deep learning, with applications in NLP and vision."
        embeddings: {OpenAI: [0.18, 0.16, ...]}
        children: {4, 5}
```

### Final Tree Structure

```
layer_to_nodes = {
    0: [Node_0, Node_1, Node_2, Node_3],  # Leaves
    1: [Node_4, Node_5],                   # First summaries
    2: [Node_6]                            # Root
}

all_nodes = {0: Node_0, 1: Node_1, ..., 6: Node_6}
leaf_nodes = {0: Node_0, 1: Node_1, 2: Node_2, 3: Node_3}
root_nodes = {6: Node_6}
num_layers = 2
```

---

## 6. Complete Algorithm Pseudocode

```
function BUILD_RAPTOR_TREE(text, config):
    //===========================================
    // PHASE 1: TEXT SPLITTING
    //===========================================
    chunks = SPLIT_TEXT(text, config.tokenizer, config.max_tokens)
    
    //===========================================
    // PHASE 2: LEAF NODE CREATION
    //===========================================
    leaf_nodes = {}
    
    parallel for (index, chunk) in enumerate(chunks):
        embeddings = {}
        for (name, model) in config.embedding_models:
            embeddings[name] = model.create_embedding(chunk)
        
        leaf_nodes[index] = Node(
            text=chunk,
            index=index,
            children=empty_set,
            embeddings=embeddings
        )
    
    //===========================================
    // PHASE 3: HIERARCHICAL CONSTRUCTION
    //===========================================
    all_nodes = copy(leaf_nodes)
    layer_to_nodes = {0: values(leaf_nodes)}
    current_level = leaf_nodes
    next_index = len(leaf_nodes)
    
    for layer in range(config.num_layers):
        node_list = sorted_values(current_level)
        
        // Check stop condition
        if len(node_list) <= config.reduction_dimension + 1:
            break
        
        // Cluster nodes
        clusters = RAPTOR_CLUSTERING(
            nodes=node_list,
            embedding_model=config.cluster_embedding_model,
            max_length=config.max_length_in_cluster,
            reduction_dimension=config.reduction_dimension,
            threshold=config.threshold
        )
        
        new_level = {}
        
        for cluster in clusters:
            // Combine cluster texts
            combined = concatenate(node.text for node in cluster)
            
            // Summarize via LLM
            summary = config.summarization_model.summarize(
                combined, 
                max_tokens=config.summarization_length
            )
            
            // Create embeddings for summary
            embeddings = {}
            for (name, model) in config.embedding_models:
                embeddings[name] = model.create_embedding(summary)
            
            // Create parent node
            parent = Node(
                text=summary,
                index=next_index,
                children={n.index for n in cluster},
                embeddings=embeddings
            )
            
            new_level[next_index] = parent
            next_index += 1
        
        // Update tracking
        layer_to_nodes[layer + 1] = values(new_level)
        all_nodes.update(new_level)
        current_level = new_level
    
    //===========================================
    // PHASE 4: TREE ASSEMBLY
    //===========================================
    return Tree(
        all_nodes=all_nodes,
        root_nodes=current_level,
        leaf_nodes=leaf_nodes,
        num_layers=len(layer_to_nodes) - 1,
        layer_to_nodes=layer_to_nodes
    )
```

---

## 7. Configuration Parameters

| Parameter | Default | Source | Impact |
|-----------|---------|--------|--------|
| `max_tokens` | 100 | TreeBuilderConfig:42-46 | Chunk size for splitting |
| `num_layers` | 5 | TreeBuilderConfig:48-52 | Maximum tree depth |
| `threshold` | 0.5 | TreeBuilderConfig:54-58 | Clustering soft threshold |
| `summarization_length` | 100 | TreeBuilderConfig:72-74 | Summary token limit |
| `reduction_dimension` | 10 | ClusterTreeConfig:20 | UMAP target dimensions |

---

## 8. Source Code References

| Component | File | Lines |
|-----------|------|-------|
| Text splitting | utils.py | 22-100 |
| Node creation | tree_builder.py | 158-179 |
| Multithreaded leaf creation | tree_builder.py | 238-258 |
| Main build entry | tree_builder.py | 260-295 |
| Tree construction loop | cluster_tree_builder.py | 55-151 |
| Cluster processing | cluster_tree_builder.py | 66-86 |
| Tree structure | tree_structures.py | 16-28 |
