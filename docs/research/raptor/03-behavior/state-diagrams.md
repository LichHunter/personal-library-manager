# RAPTOR State Diagrams

> System states and transitions

## 1. RetrievalAugmentation Instance States

The `RetrievalAugmentation` class can be in three primary states based on its internal components:

```
                           +------------------+
                           |    UNINITIALIZED |
                           |                  |
                           | tree = None      |
                           | retriever = None |
                           +--------+---------+
                                    |
                                    | __init__(config)
                                    | [config validation]
                                    | [tree_builder created]
                                    v
+------------------+       +------------------+
|  TREE_LOADED     |       |    INITIALIZED   |
|                  |       |                  |
| tree = Tree      |       | tree = None      |
| retriever = TR   |<------| retriever = None |
| [from file/obj]  |       | tree_builder = TB|
+--------+---------+  OR   +--------+---------+
         |                          |
         |                          | add_documents(text)
         |                          | [build_from_text]
         |                          v
         |                 +------------------+
         |                 |     READY        |
         |                 |                  |
         +---------------->| tree = Tree      |
                           | retriever = TR   |
                           | qa_model = QA    |
                           +--------+---------+
                                    |
                                    | save(path)
                                    v
                           +------------------+
                           |     PERSISTED    |
                           |                  |
                           | [Tree pickled    |
                           |  to disk]        |
                           +------------------+
```

### State Transition Table

| Current State | Event | Next State | Code Location |
|--------------|-------|------------|---------------|
| - | `__init__(config=None)` | INITIALIZED | RetrievalAugmentation.py:159-202 |
| - | `__init__(config, tree=path)` | TREE_LOADED | RetrievalAugmentation.py:174-181 |
| - | `__init__(config, tree=Tree)` | READY | RetrievalAugmentation.py:182-183 |
| INITIALIZED | `add_documents(text)` | READY | RetrievalAugmentation.py:204-220 |
| TREE_LOADED | `add_documents(text)` + "y" | READY | Overwrites existing |
| READY | `save(path)` | PERSISTED | RetrievalAugmentation.py:301-306 |
| READY | `retrieve(query)` | READY | Returns context |
| READY | `answer_question(q)` | READY | Returns answer |

### State Invariants

```
+------------------+--------------------------------------------+
| State            | Invariant Conditions                       |
+------------------+--------------------------------------------+
| UNINITIALIZED    | Does not exist as valid program state      |
+------------------+--------------------------------------------+
| INITIALIZED      | tree_builder != None                       |
|                  | qa_model != None                           |
|                  | tree == None                               |
|                  | retriever == None                          |
+------------------+--------------------------------------------+
| READY            | tree != None                               |
|                  | retriever != None                          |
|                  | tree_builder != None                       |
|                  | qa_model != None                           |
+------------------+--------------------------------------------+
```

## 2. Tree Building Process States

The tree building process goes through distinct phases during `build_from_text()`:

```
    +---------------+
    |    START      |
    | (raw text)    |
    +-------+-------+
            |
            | split_text()
            | (utils.py:22-100)
            v
    +---------------+
    |   CHUNKING    |
    |               |
    | Input: text   |
    | Output: chunks|
    | [max_tokens]  |
    +-------+-------+
            |
            | multithreaded_create_leaf_nodes()
            | (tree_builder.py:238-258)
            v
    +---------------+
    |  EMBEDDING    |
    |               |
    | Input: chunks |
    | Output: nodes |
    | [leaf_nodes]  |
    +-------+-------+
            |
            | layer_to_nodes[0] = leaf_nodes
            v
    +---------------+
    |  CLUSTERING   |<-----------+
    |               |            |
    | perform_      |            |
    | clustering()  |            |
    | UMAP + GMM    |            |
    +-------+-------+            |
            |                    |
            | clusters created   |
            v                    |
    +---------------+            |
    | SUMMARIZING   |            |
    |               |            |
    | For each      |            |
    | cluster:      |            |
    | - get_text()  |            |
    | - summarize() |            |
    | - create_node()|           |
    +-------+-------+            |
            |                    |
            | new_level_nodes    |
            v                    |
    +---------------+            |
    | LAYER_CHECK   |            |
    |               |            |
    | layer++       |            |
    | enough nodes? |            |
    +-------+-------+            |
            |                    |
      YES   |        NO          |
   (>dim+1) |    (<=dim+1)       |
            |        |           |
            |        v           |
            |  +-----------+     |
            |  |   DONE    |     |
            |  +-----------+     |
            |                    |
            +--------------------+
            (next layer)
```

### Phase Details

```
+---------------+--------------------------------------------------+
| Phase         | Description                                      |
+---------------+--------------------------------------------------+
| CHUNKING      | - Split text on sentence boundaries              |
|               | - Respect max_tokens limit (default: 100)        |
|               | - Handle long sentences via sub-splitting        |
|               | - Output: List[str] of text chunks               |
+---------------+--------------------------------------------------+
| EMBEDDING     | - Create Node for each chunk                     |
|               | - Generate embeddings via embedding_model        |
|               | - Parallel execution via ThreadPoolExecutor      |
|               | - Output: Dict[int, Node] leaf_nodes             |
+---------------+--------------------------------------------------+
| CLUSTERING    | - Extract embeddings from nodes                  |
|               | - UMAP dimensionality reduction (global)         |
|               | - GMM optimal cluster count via BIC              |
|               | - UMAP local clustering per global cluster       |
|               | - Validate cluster sizes (max_length_in_cluster) |
|               | - Recursive re-clustering if too large           |
|               | - Output: List[List[Node]] clusters              |
+---------------+--------------------------------------------------+
| SUMMARIZING   | - Concatenate text from cluster nodes            |
|               | - LLM summarization (summarization_length)       |
|               | - Create parent node with children indices       |
|               | - Generate embeddings for summary                |
|               | - Output: Dict[int, Node] new_level_nodes        |
+---------------+--------------------------------------------------+
| LAYER_CHECK   | - Check if enough nodes for another layer        |
|               | - Stop condition: len(nodes) <= reduction_dim+1  |
|               | - Update layer_to_nodes, all_tree_nodes          |
|               | - Either loop to CLUSTERING or proceed to DONE   |
+---------------+--------------------------------------------------+
```

## 3. Clustering Algorithm States (RAPTOR_Clustering)

The clustering algorithm itself has internal states:

```
    +------------------+
    |   INPUT_NODES    |
    |                  |
    | List[Node]       |
    | with embeddings  |
    +--------+---------+
             |
             | get embeddings from nodes
             | (cluster_utils.py:143)
             v
    +------------------+
    |  GLOBAL_REDUCE   |
    |                  |
    | UMAP reduction   |
    | n_neighbors =    |
    |   sqrt(n-1)      |
    | dim = min(dim,   |
    |         n-2)     |
    +--------+---------+
             |
             | reduced_embeddings_global
             v
    +------------------+
    |  GLOBAL_CLUSTER  |
    |                  |
    | GMM clustering   |
    | optimal_clusters |
    |   via BIC        |
    | threshold=0.1    |
    +--------+---------+
             |
             | n_global_clusters
             v
    +------------------+
    | LOCAL_ITERATION  |<---------+
    |                  |          |
    | For each global  |          |
    | cluster i:       |          |
    +--------+---------+          |
             |                    |
             | get cluster nodes  |
             v                    |
    +------------------+          |
    |  LOCAL_REDUCE    |          |
    |                  |          |
    | IF len > dim+1:  |          |
    |   UMAP local     |          |
    | ELSE:            |          |
    |   single cluster |          |
    +--------+---------+          |
             |                    |
             v                    |
    +------------------+          |
    |  LOCAL_CLUSTER   |          |
    |                  |          |
    | GMM on local     |          |
    | reduced emb      |          |
    +--------+---------+          |
             |                    |
             | assign labels      |
             |                    |
             +--------------------+
             (next global cluster)
             |
             | all clusters labeled
             v
    +------------------+
    |  SIZE_VALIDATE   |
    |                  |
    | For each cluster:|
    | check token count|
    +--------+---------+
             |
             |
      +------+------+
      |             |
  <= max_len    > max_len
      |             |
      v             v
    +--------+ +------------------+
    | ACCEPT | |   RECLUSTER      |
    |        | |                  |
    | Add to | | Recursive call   |
    | results| | with cluster     |
    +--------+ | nodes only       |
               +------------------+
                      |
                      v
               +------------------+
               |    OUTPUT        |
               |                  |
               | List[List[Node]] |
               | node_clusters    |
               +------------------+
```

## 4. Retrieval Mode State Machine

The retrieval system operates in one of two modes:

```
                    +------------------+
                    |  QUERY_RECEIVED  |
                    |                  |
                    | query string     |
                    +--------+---------+
                             |
                             | validate params
                             v
                    +------------------+
                    | MODE_SELECTION   |
                    |                  |
                    | collapse_tree?   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
          collapse_tree              collapse_tree
            = True                     = False
              |                             |
              v                             v
    +------------------+         +------------------+
    | COLLAPSED_MODE   |         | TRAVERSAL_MODE   |
    |                  |         |                  |
    | Search ALL nodes |         | Start at layer N |
    | Single flat query|         | Multi-layer      |
    +--------+---------+         +--------+---------+
             |                            |
             |                            |
             v                            v
    +------------------+         +------------------+
    | RANK_ALL_NODES   |         | LAYER_ITERATION  |<----+
    |                  |         |                  |     |
    | Embed query      |         | Current layer    |     |
    | Distance calc    |         | nodes            |     |
    | Sort by distance |         +--------+---------+     |
    +--------+---------+                  |               |
             |                            |               |
             v                            v               |
    +------------------+         +------------------+     |
    | SELECT_TOP_K     |         | SELECT_BEST      |     |
    |                  |         |                  |     |
    | Iterate sorted   |         | top_k OR         |     |
    | until max_tokens |         | threshold        |     |
    +--------+---------+         +--------+---------+     |
             |                            |               |
             |                            |               |
             |                   +--------+---------+     |
             |                   | IS_LAST_LAYER?   |     |
             |                   +--------+---------+     |
             |                            |               |
             |                  YES       |      NO       |
             |                    +-------+-------+       |
             |                    |               |       |
             |                    v               v       |
             |           +----------+   +------------------+
             |           |  DONE    |   | GET_CHILDREN     |
             |           +----------+   |                  |
             |                          | node.children    |
             |                          | -> next node_list|
             |                          +--------+---------+
             |                                   |
             |                                   +---------+
             |
             v
    +------------------+
    | CONTEXT_BUILD    |
    |                  |
    | Concatenate text |
    | from selected    |
    | nodes            |
    +--------+---------+
             |
             v
    +------------------+
    |    COMPLETE      |
    |                  |
    | Return context   |
    | + layer_info     |
    +------------------+
```

### Retrieval Mode Comparison

| Aspect | Collapsed Mode | Traversal Mode |
|--------|---------------|----------------|
| **Nodes Searched** | All nodes (all layers) | Subset per layer |
| **Queries Made** | 1 embedding query | 1 per layer traversed |
| **Token Budget** | max_tokens enforced | Via top_k/threshold |
| **Best For** | Quick, broad retrieval | Hierarchical exploration |
| **Code Path** | `retrieve_information_collapse_tree()` | `retrieve_information()` |
| **Default** | Yes (`collapse_tree=True`) | No |

## 5. Node State Lifecycle

Individual nodes have a simple lifecycle:

```
    +------------------+
    |   TEXT_CHUNK     |
    |   (raw string)   |
    +--------+---------+
             |
             | create_node()
             | tree_builder.py:158-179
             v
    +------------------+
    |   LEAF_NODE      |
    |                  |
    | text: chunk      |
    | index: int       |
    | children: {}     |
    | embeddings: {    |
    |   "OpenAI": [...]}
    +--------+---------+
             |
             | [Node is used in clustering]
             | [Node is summarized with siblings]
             v
    +------------------+
    |   CHILD_NODE     |
    |                  |
    | [Same as LEAF    |
    |  but referenced  |
    |  by parent node] |
    +--------+---------+

    ------------ Meanwhile, new parent is created ------------

    +------------------+
    |  SUMMARY_TEXT    |
    |  (LLM output)    |
    +--------+---------+
             |
             | create_node(children_indices)
             | cluster_tree_builder.py:80-82
             v
    +------------------+
    |  SUMMARY_NODE    |
    |                  |
    | text: summary    |
    | index: next_idx  |
    | children: {0,1,2}|
    | embeddings: {...}|
    +--------+---------+
             |
             | [May become child of higher layer]
             v
    +------------------+
    |   ROOT_NODE      |
    |                  |
    | [Top layer node  |
    |  no parent refs  |
    |  only children]  |
    +------------------+
```

### Node Data Structure States

```
Node (Leaf):
+------------------------------------------+
| text: "Once upon a time there was..."    |
| index: 0                                 |
| children: set()   # Empty for leaves     |
| embeddings: {                            |
|   "OpenAI": [0.023, -0.156, ..., 0.089]  |
| }                                        |
+------------------------------------------+

Node (Summary):
+------------------------------------------+
| text: "A fairy tale about a princess..." |
| index: 15                                |
| children: {0, 1, 2, 3}  # Leaf indices   |
| embeddings: {                            |
|   "OpenAI": [0.045, -0.112, ..., 0.134]  |
| }                                        |
+------------------------------------------+

Node (Root):
+------------------------------------------+
| text: "Complete story summary..."        |
| index: 42                                |
| children: {15, 16, 17}  # Summary nodes  |
| embeddings: {                            |
|   "OpenAI": [0.067, -0.098, ..., 0.156]  |
| }                                        |
+------------------------------------------+
```
