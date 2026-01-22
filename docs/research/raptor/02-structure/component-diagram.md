# RAPTOR Component Diagram

## High-Level Architecture

```
+===========================================================================+
|                          RAPTOR SYSTEM                                     |
+===========================================================================+
|                                                                            |
|  +---------------------------------------------------------------------+  |
|  |                     FACADE / ORCHESTRATION LAYER                     |  |
|  |                                                                      |  |
|  |  +----------------------------------------------------------------+  |  |
|  |  |                  RetrievalAugmentation                         |  |  |
|  |  |                                                                |  |  |
|  |  |  - add_documents(docs)     Coordinates all operations          |  |  |
|  |  |  - retrieve(question)      Entry point for users               |  |  |
|  |  |  - answer_question(q)      Combines retrieval + QA             |  |  |
|  |  |  - save(path) / load       Persistence                         |  |  |
|  |  +----------------------------------------------------------------+  |  |
|  +---------------------------------------------------------------------+  |
|           |                           |                    |              |
|           v                           v                    v              |
|  +------------------+    +---------------------+    +------------------+  |
|  |  INDEXING        |    |    RETRIEVAL        |    |   QA/RESPONSE    |  |
|  |  SUBSYSTEM       |    |    SUBSYSTEM        |    |   SUBSYSTEM      |  |
|  +------------------+    +---------------------+    +------------------+  |
|                                                                            |
+============================================================================+


                    DETAILED SUBSYSTEM BREAKDOWN
============================================================================

+--------------------------- INDEXING SUBSYSTEM ----------------------------+
|                                                                            |
|   Responsibility: Transform raw text into a hierarchical tree of          |
|                   summarized, embedded nodes                               |
|                                                                            |
|   +-----------------------------+    +-------------------------------+    |
|   |      Tree Builders          |    |      Clustering               |    |
|   |                             |    |                               |    |
|   |  +----------------------+   |    |  +--------------------------+ |    |
|   |  | TreeBuilder          |   |    |  | ClusteringAlgorithm (ABC)| |    |
|   |  | (Abstract Base)      |   |    |  +--------------------------+ |    |
|   |  +----------+-----------+   |    |             |                 |    |
|   |             |               |    |             v                 |    |
|   |             v               |    |  +--------------------------+ |    |
|   |  +----------------------+   |    |  | RAPTOR_Clustering        | |    |
|   |  | ClusterTreeBuilder   |<--+--->|  |                          | |    |
|   |  | (Concrete Impl)      |   |    |  | - UMAP reduction         | |    |
|   |  +----------------------+   |    |  | - GMM clustering         | |    |
|   |                             |    |  | - Recursive splitting    | |    |
|   +-----------------------------+    |  +--------------------------+ |    |
|                                      +-------------------------------+    |
|                                                                            |
|   Data Flow:                                                               |
|   text -> split_text() -> leaf_nodes -> cluster -> summarize -> parent    |
|                                           |                        |       |
|                                           +--- repeat until root --+       |
|                                                                            |
+----------------------------------------------------------------------------+


+--------------------------- RETRIEVAL SUBSYSTEM ---------------------------+
|                                                                            |
|   Responsibility: Navigate the tree to find relevant context for queries  |
|                                                                            |
|   +-----------------------------------------------------------------------+|
|   |                     Retrievers                                        ||
|   |                                                                       ||
|   |  +------------------------+         +-----------------------------+  ||
|   |  | BaseRetriever (ABC)    |         | TreeRetriever               |  ||
|   |  |                        |         |                             |  ||
|   |  | + retrieve(query)      |<--------|  Two retrieval strategies:  |  ||
|   |  +------------------------+         |                             |  ||
|   |             ^                       |  1. collapse_tree: search   |  ||
|   |             |                       |     ALL nodes flat          |  ||
|   |  +------------------------+         |                             |  ||
|   |  | FaissRetriever         |         |  2. tree_traversal: start   |  ||
|   |  |                        |         |     at root, descend via    |  ||
|   |  | - FAISS index          |         |     children                |  ||
|   |  | - flat vector search   |         +-----------------------------+  ||
|   |  +------------------------+                                          ||
|   +-----------------------------------------------------------------------+|
|                                                                            |
|   Selection Modes:                                                         |
|   - top_k: Return k most similar nodes                                     |
|   - threshold: Return nodes above similarity threshold                     |
|                                                                            |
+----------------------------------------------------------------------------+


+----------------------------- MODEL ADAPTERS ------------------------------+
|                                                                            |
|   Responsibility: Provide pluggable ML model interfaces                   |
|                                                                            |
|   +-------------------------+   +----------------------+   +-------------+ |
|   | Embedding Models        |   | Summarization Models |   | QA Models   | |
|   +-------------------------+   +----------------------+   +-------------+ |
|   |                         |   |                      |   |             | |
|   | BaseEmbeddingModel(ABC) |   | BaseSummarization-   |   | BaseQAModel | |
|   |  + create_embedding()   |   |   Model (ABC)        |   |   (ABC)     | |
|   |                         |   |  + summarize()       |   |  + answer_  | |
|   | Implementations:        |   |                      |   |    question | |
|   |  - OpenAIEmbedding      |   | Implementations:     |   |             | |
|   |  - SBertEmbedding       |   |  - GPT3Turbo         |   | Impls:      | |
|   |                         |   |  - GPT3              |   | - GPT3      | |
|   +-------------------------+   +----------------------+   | - GPT3Turbo | |
|                                                            | - GPT4      | |
|                                                            | - UnifiedQA | |
|                                                            +-------------+ |
|                                                                            |
+----------------------------------------------------------------------------+


+------------------------------- CORE DATA ---------------------------------+
|                                                                            |
|   Responsibility: Core data structures shared across subsystems           |
|                                                                            |
|   +-------------------------------+    +--------------------------------+ |
|   |            Node               |    |             Tree               | |
|   +-------------------------------+    +--------------------------------+ |
|   |  - text: str                  |    |  - all_nodes: Dict[int, Node]  | |
|   |  - index: int                 |    |  - root_nodes: Dict[int, Node] | |
|   |  - children: Set[int]         |    |  - leaf_nodes: Dict[int, Node] | |
|   |  - embeddings: Dict[str,vec]  |    |  - num_layers: int             | |
|   +-------------------------------+    |  - layer_to_nodes: Dict        | |
|                                        +--------------------------------+ |
|                                                                            |
+----------------------------------------------------------------------------+


+------------------------------- UTILITIES ---------------------------------+
|                                                                            |
|   Responsibility: Shared helper functions                                 |
|                                                                            |
|   +---------------------------------------------------------------------+ |
|   |                          utils.py                                    | |
|   |                                                                      | |
|   |  Text Processing:             Embedding Operations:                  | |
|   |   - split_text()               - distances_from_embeddings()         | |
|   |   - get_text()                 - indices_of_nearest_neighbors()      | |
|   |                                - get_embeddings()                    | |
|   |  Node Operations:                                                    | |
|   |   - get_node_list()           Tree Operations:                       | |
|   |   - get_children()             - reverse_mapping()                   | |
|   +---------------------------------------------------------------------+ |
|                                                                            |
+----------------------------------------------------------------------------+
```

## Component Interfaces

### RetrievalAugmentation (Facade)

The main entry point that users interact with. Coordinates between:
- Tree building (indexing)
- Tree retrieval (search)
- QA model (answer generation)

**Interface:**
```
add_documents(docs: str) -> None
retrieve(question: str, ...) -> str
answer_question(question: str, ...) -> str
save(path: str) -> None
```

### TreeBuilder -> ClusterTreeBuilder

**Responsibility:** Convert raw text into a hierarchical tree structure.

**Interface:**
```
build_from_text(text: str) -> Tree
construct_tree(...) -> Dict[int, Node]  # Abstract in TreeBuilder
create_node(index, text, children) -> Node
```

### TreeRetriever

**Responsibility:** Search the tree for relevant context given a query.

**Interface:**
```
retrieve(query, start_layer, num_layers, top_k, max_tokens, collapse_tree) -> str
retrieve_information(current_nodes, query, num_layers) -> (nodes, context)
retrieve_information_collapse_tree(query, top_k, max_tokens) -> (nodes, context)
```

### Model Adapters

All follow the same pattern: abstract base class with concrete implementations.

| Model Type | Base Class | Method Signature |
|------------|------------|------------------|
| Embedding | `BaseEmbeddingModel` | `create_embedding(text) -> List[float]` |
| Summarization | `BaseSummarizationModel` | `summarize(context, max_tokens) -> str` |
| QA | `BaseQAModel` | `answer_question(context, question) -> str` |

## Subsystem Boundaries

```
                           USER
                             |
                             v
+===========================[API BOUNDARY]================================+
|                     RetrievalAugmentation                               |
+==========================================================================+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
+--[INDEXING]--+    +--[RETRIEVAL]--+    +--[QA]--+
| TreeBuilder  |    | TreeRetriever |    | QAModel|
| Clustering   |    | FaissRetr.    |    +--------+
+--------------+    +---------------+
          |                  |
          +--------+---------+
                   |
          +--[MODELS]--+
          | Embedding  |
          | Summarize  |
          +------------+
                   |
          +--[DATA]----+
          | Node, Tree |
          +------------+
```

## Key Design Decisions

1. **Facade Pattern**: `RetrievalAugmentation` hides complexity of tree building/retrieval
2. **Strategy Pattern**: Pluggable models (embedding, summarization, QA)
3. **Template Method**: `TreeBuilder.construct_tree()` is abstract, `ClusterTreeBuilder` implements
4. **Configuration Objects**: Each component has a `*Config` class for parameters
5. **Separation of Concerns**: Indexing and retrieval are separate subsystems
