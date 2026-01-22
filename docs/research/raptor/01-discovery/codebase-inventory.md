# RAPTOR Codebase Inventory

> Complete file-by-file inventory of the RAPTOR source code

## Overview

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is a hierarchical retrieval system that builds a tree of document summaries for more context-aware information retrieval.

**Total Source Files:** 13  
**Total Lines of Code:** ~1,908 lines (excluding notebooks)  
**Primary Language:** Python 3.8+

---

## File Inventory

| File | Lines | Primary Purpose | Key Classes/Functions |
|------|-------|-----------------|----------------------|
| `__init__.py` | 17 | Public API exports | Exports all public classes |
| `tree_structures.py` | 29 | Core data structures | `Node`, `Tree` |
| `tree_builder.py` | 370 | Abstract tree building | `TreeBuilder`, `TreeBuilderConfig` |
| `cluster_tree_builder.py` | 152 | Clustering-based tree building | `ClusterTreeBuilder`, `ClusterTreeConfig` |
| `cluster_utils.py` | 186 | Clustering algorithms | `RAPTOR_Clustering`, `ClusteringAlgorithm`, GMM/UMAP functions |
| `tree_retriever.py` | 328 | Tree-based retrieval | `TreeRetriever`, `TreeRetrieverConfig` |
| `RetrievalAugmentation.py` | 307 | Main facade/orchestrator | `RetrievalAugmentation`, `RetrievalAugmentationConfig` |
| `EmbeddingModels.py` | 38 | Embedding model interfaces | `BaseEmbeddingModel`, `OpenAIEmbeddingModel`, `SBertEmbeddingModel` |
| `SummarizationModels.py` | 75 | Summarization model interfaces | `BaseSummarizationModel`, `GPT3TurboSummarizationModel`, `GPT3SummarizationModel` |
| `QAModels.py` | 186 | Question answering interfaces | `BaseQAModel`, `GPT3QAModel`, `GPT3TurboQAModel`, `GPT4QAModel`, `UnifiedQAModel` |
| `Retrievers.py` | 9 | Base retriever interface | `BaseRetriever` |
| `utils.py` | 209 | Utility functions | `split_text`, `distances_from_embeddings`, `get_text`, etc. |
| `FaissRetriever.py` | 202 | FAISS-based flat retrieval | `FaissRetriever`, `FaissRetrieverConfig` |

---

## Detailed File Descriptions

### `__init__.py` (17 lines)

**Purpose:** Package initialization and public API definition.

**Exports:**
```python
# Tree Building
ClusterTreeBuilder, ClusterTreeConfig
TreeBuilder, TreeBuilderConfig

# Data Structures
Node, Tree

# Retrieval
TreeRetriever, TreeRetrieverConfig
FaissRetriever, FaissRetrieverConfig
BaseRetriever

# Main Facade
RetrievalAugmentation, RetrievalAugmentationConfig

# Models
BaseEmbeddingModel, OpenAIEmbeddingModel, SBertEmbeddingModel
BaseSummarizationModel, GPT3SummarizationModel, GPT3TurboSummarizationModel
BaseQAModel, GPT3QAModel, GPT3TurboQAModel, GPT4QAModel, UnifiedQAModel
```

**Imports from:**
- `cluster_tree_builder`
- `EmbeddingModels`
- `FaissRetriever`
- `QAModels`
- `RetrievalAugmentation`
- `Retrievers`
- `SummarizationModels`
- `tree_builder`
- `tree_retriever`
- `tree_structures`

---

### `tree_structures.py` (29 lines)

**Purpose:** Defines core data structures for the hierarchical tree.

**Classes:**

| Class | Description | Attributes |
|-------|-------------|------------|
| `Node` | Single node in tree | `text: str`, `index: int`, `children: Set[int]`, `embeddings: dict` |
| `Tree` | Complete tree structure | `all_nodes`, `root_nodes`, `leaf_nodes`, `num_layers`, `layer_to_nodes` |

**Imports:**
- `typing` (Dict, List, Set)

**Notes:**
- Simple dataclasses without validation
- `Node.embeddings` is a dict mapping model names to embedding vectors
- `Tree.layer_to_nodes` maps layer index to list of nodes at that layer

---

### `tree_builder.py` (370 lines)

**Purpose:** Abstract base class for tree building with configuration.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `TreeBuilderConfig` | 24-131 | Configuration container with validation |
| `TreeBuilder` | 133-370 | Abstract tree builder with common functionality |

**TreeBuilderConfig Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokenizer` | tiktoken cl100k_base | Text tokenizer |
| `max_tokens` | 100 | Max tokens per text chunk |
| `num_layers` | 5 | Number of tree layers |
| `threshold` | 0.5 | Similarity threshold |
| `top_k` | 5 | Top-k selection |
| `selection_mode` | "top_k" | "top_k" or "threshold" |
| `summarization_length` | 100 | Max summary tokens |
| `summarization_model` | GPT3TurboSummarizationModel | Summarizer |
| `embedding_models` | {"OpenAI": OpenAIEmbeddingModel()} | Dict of embedding models |
| `cluster_embedding_model` | "OpenAI" | Which model to use for clustering |

**TreeBuilder Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `create_node()` | 158-179 | Creates a node with embeddings |
| `create_embedding()` | 181-193 | Generates embedding for text |
| `summarize()` | 195-206 | Generates summary using model |
| `get_relevant_nodes()` | 208-236 | Finds top-k similar nodes |
| `multithreaded_create_leaf_nodes()` | 238-258 | Parallel leaf node creation |
| `build_from_text()` | 260-295 | Main entry point for tree building |
| `construct_tree()` | 297-317 | Abstract method (must override) |

**Imports:**
- `copy`, `logging`, `os`
- `abc.abstractclassmethod`
- `concurrent.futures` (ThreadPoolExecutor, as_completed)
- `threading.Lock`
- `typing` (Dict, List, Optional, Set, Tuple)
- `openai`, `tiktoken`, `tenacity`
- Internal: `EmbeddingModels`, `SummarizationModels`, `tree_structures`, `utils`

**Notes:**
- Lines 319-369 contain commented-out transformer-like tree builder code
- Uses `@abstractclassmethod` (deprecated, should be `@abstractmethod`)

---

### `cluster_tree_builder.py` (152 lines)

**Purpose:** Concrete tree builder using clustering algorithms.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `ClusterTreeConfig` | 17-38 | Extends TreeBuilderConfig with clustering params |
| `ClusterTreeBuilder` | 41-152 | Tree builder using UMAP + GMM clustering |

**ClusterTreeConfig Additional Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reduction_dimension` | 10 | UMAP target dimension |
| `clustering_algorithm` | RAPTOR_Clustering | Clustering class to use |
| `clustering_params` | {} | Additional clustering params |

**ClusterTreeBuilder.construct_tree() Algorithm:**
1. For each layer (0 to num_layers):
   - Get current layer nodes
   - If too few nodes, stop (< reduction_dimension + 1)
   - Perform clustering using clustering_algorithm
   - For each cluster:
     - Concatenate node texts
     - Generate summary
     - Create parent node with children references
   - Update layer_to_nodes and all_tree_nodes

**Imports:**
- `logging`, `pickle`
- `concurrent.futures.ThreadPoolExecutor`
- `threading.Lock`
- `typing` (Dict, List, Set)
- Internal: `cluster_utils`, `tree_builder`, `tree_structures`, `utils`

---

### `cluster_utils.py` (186 lines)

**Purpose:** Implements the core RAPTOR clustering algorithm using UMAP and GMM.

**Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| `global_cluster_embeddings()` | 23-34 | UMAP dimensionality reduction (global) |
| `local_cluster_embeddings()` | 37-43 | UMAP dimensionality reduction (local) |
| `get_optimal_clusters()` | 46-57 | Find optimal cluster count via BIC |
| `GMM_cluster()` | 60-66 | Gaussian Mixture Model clustering |
| `perform_clustering()` | 69-123 | Two-stage global+local clustering |

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `ClusteringAlgorithm` | 126-129 | Abstract base for clustering |
| `RAPTOR_Clustering` | 132-185 | Main RAPTOR clustering implementation |

**RAPTOR_Clustering.perform_clustering() Algorithm:**
1. Extract embeddings from nodes
2. Call `perform_clustering()` function (UMAP + GMM)
3. For each unique cluster label:
   - Get nodes belonging to cluster
   - If single node, keep as-is
   - If total text tokens > max_length_in_cluster (3500):
     - Recursively recluster
   - Else add cluster to results

**Imports:**
- `logging`, `random`
- `abc` (ABC, abstractmethod)
- `typing` (List, Optional)
- `numpy`, `tiktoken`, `umap`
- `sklearn.mixture.GaussianMixture`
- Internal: `tree_structures`, `utils`

**Notes:**
- `RANDOM_SEED = 224` for reproducibility
- Uses soft clustering (nodes can belong to multiple clusters)

---

### `tree_retriever.py` (328 lines)

**Purpose:** Retrieves relevant information from a built tree.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `TreeRetrieverConfig` | 19-103 | Configuration for retrieval |
| `TreeRetriever` | 106-328 | Tree traversal and retrieval |

**TreeRetrieverConfig Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tokenizer` | tiktoken cl100k_base | Text tokenizer |
| `threshold` | 0.5 | Similarity threshold |
| `top_k` | 5 | Top-k selection |
| `selection_mode` | "top_k" | "top_k" or "threshold" |
| `context_embedding_model` | "OpenAI" | Model name for node embeddings |
| `embedding_model` | OpenAIEmbeddingModel() | Model for query embedding |
| `num_layers` | None | Layers to traverse |
| `start_layer` | None | Starting layer (default: top) |

**TreeRetriever Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `create_embedding()` | 146-156 | Embed query text |
| `retrieve_information_collapse_tree()` | 158-195 | Search ALL nodes (flat search) |
| `retrieve_information()` | 197-250 | Layer-by-layer tree traversal |
| `retrieve()` | 252-327 | Main retrieve method |

**Retrieval Strategies:**
1. **Collapsed Tree** (`collapse_tree=True`): Search all nodes regardless of layer
2. **Tree Traversal** (`collapse_tree=False`): Start at top layer, descend through children

**Imports:**
- `logging`, `os`
- `typing` (Dict, List, Set)
- `tiktoken`, `tenacity`
- Internal: `EmbeddingModels`, `Retrievers`, `tree_structures`, `utils`

---

### `RetrievalAugmentation.py` (307 lines)

**Purpose:** Main facade class that orchestrates building, retrieval, and QA.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `RetrievalAugmentationConfig` | 18-150 | Master configuration class |
| `RetrievalAugmentation` | 153-307 | Main user-facing class |

**RetrievalAugmentationConfig Parameters:**
| Parameter | Description |
|-----------|-------------|
| `tree_builder_config` | TreeBuilderConfig instance |
| `tree_retriever_config` | TreeRetrieverConfig instance |
| `qa_model` | QA model instance |
| `embedding_model` | Shared embedding model |
| `summarization_model` | Summarization model |
| `tree_builder_type` | "cluster" (only option) |
| `tr_*` | TreeRetriever config shortcuts |
| `tb_*` | TreeBuilder config shortcuts |

**RetrievalAugmentation Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `__init__()` | 159-202 | Initialize with config, optionally load tree |
| `add_documents()` | 204-220 | Build tree from text |
| `retrieve()` | 222-261 | Get relevant context |
| `answer_question()` | 263-299 | Retrieve + answer |
| `save()` | 301-306 | Pickle tree to file |

**Imports:**
- `logging`, `pickle`
- Internal: `cluster_tree_builder`, `EmbeddingModels`, `QAModels`, `SummarizationModels`, `tree_builder`, `tree_retriever`, `tree_structures`

**Notes:**
- `supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}` - only cluster builder supported
- Tree can be loaded from pickle file path or Tree instance

---

### `EmbeddingModels.py` (38 lines)

**Purpose:** Embedding model interfaces and implementations.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `BaseEmbeddingModel` | 11-14 | Abstract base class |
| `OpenAIEmbeddingModel` | 17-29 | OpenAI text-embedding-ada-002 |
| `SBertEmbeddingModel` | 32-37 | Sentence Transformers model |

**BaseEmbeddingModel Interface:**
```python
def create_embedding(self, text) -> List[float]:
    pass
```

**Imports:**
- `logging`
- `abc` (ABC, abstractmethod)
- `openai.OpenAI`
- `sentence_transformers.SentenceTransformer`
- `tenacity` (retry, stop_after_attempt, wait_random_exponential)

---

### `SummarizationModels.py` (75 lines)

**Purpose:** Summarization model interfaces and OpenAI implementations.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `BaseSummarizationModel` | 11-14 | Abstract base class |
| `GPT3TurboSummarizationModel` | 17-44 | gpt-3.5-turbo (chat) |
| `GPT3SummarizationModel` | 47-74 | text-davinci-003 (legacy) |

**BaseSummarizationModel Interface:**
```python
def summarize(self, context, max_tokens=150) -> str:
    pass
```

**Prompt Used:**
```
Write a summary of the following, including as many key details as possible: {context}:
```

**Imports:**
- `logging`, `os`
- `abc` (ABC, abstractmethod)
- `openai.OpenAI`
- `tenacity` (retry, stop_after_attempt, wait_random_exponential)

---

### `QAModels.py` (186 lines)

**Purpose:** Question-answering model interfaces and implementations.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `BaseQAModel` | 15-18 | Abstract base class |
| `GPT3QAModel` | 21-60 | text-davinci-003 (completions) |
| `GPT3TurboQAModel` | 63-112 | gpt-3.5-turbo (chat) |
| `GPT4QAModel` | 115-164 | gpt-4 (chat) |
| `UnifiedQAModel` | 167-185 | T5-based local model |

**BaseQAModel Interface:**
```python
def answer_question(self, context, question) -> str:
    pass
```

**Imports:**
- `logging`, `os`, `getpass`
- `abc` (ABC, abstractmethod)
- `openai.OpenAI`
- `torch`
- `tenacity` (retry, stop_after_attempt, wait_random_exponential)
- `transformers` (T5ForConditionalGeneration, T5Tokenizer)

**Notes:**
- `UnifiedQAModel` uses allenai/unifiedqa-v2-t5-3b-1363200 by default
- All OpenAI models use retry logic (6 attempts with exponential backoff)

---

### `Retrievers.py` (9 lines)

**Purpose:** Abstract base class for retrievers.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `BaseRetriever` | 5-8 | Abstract retriever interface |

**Interface:**
```python
def retrieve(self, query: str) -> str:
    pass
```

**Imports:**
- `abc` (ABC, abstractmethod)
- `typing.List`

---

### `utils.py` (209 lines)

**Purpose:** Shared utility functions.

**Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| `reverse_mapping()` | 14-19 | Invert layer_to_nodes to node_to_layer |
| `split_text()` | 22-100 | Split text into token-limited chunks |
| `distances_from_embeddings()` | 103-136 | Calculate embedding distances |
| `get_node_list()` | 139-151 | Convert node dict to sorted list |
| `get_embeddings()` | 154-165 | Extract embeddings from nodes |
| `get_children()` | 168-178 | Extract children from nodes |
| `get_text()` | 181-195 | Concatenate node texts |
| `indices_of_nearest_neighbors_from_distances()` | 198-208 | Sort by distance |

**split_text() Details:**
- Splits on: `.`, `!`, `?`, `\n`
- Long sentences split on: `,`, `;`, `:`
- Supports overlap between chunks
- Handles edge cases (empty chunks, very long sentences)

**Distance Metrics Supported:**
- `cosine` (default)
- `L1` (Manhattan/cityblock)
- `L2` (Euclidean)
- `Linf` (Chebyshev)

**Imports:**
- `logging`, `re`
- `typing` (Dict, List, Set)
- `numpy`, `tiktoken`
- `scipy.spatial`
- Internal: `tree_structures`

---

### `FaissRetriever.py` (202 lines)

**Purpose:** FAISS-based flat retrieval (non-hierarchical alternative).

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `FaissRetrieverConfig` | 14-79 | Configuration for FAISS retriever |
| `FaissRetriever` | 82-201 | FAISS index-based retriever |

**FaissRetrieverConfig Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 100 | Max tokens per chunk |
| `max_context_tokens` | 3500 | Max total context tokens |
| `use_top_k` | False | Use top-k vs token-limited |
| `embedding_model` | OpenAIEmbeddingModel() | Document embedding model |
| `question_embedding_model` | Same as embedding_model | Query embedding model |
| `top_k` | 5 | Number of results |
| `tokenizer` | tiktoken cl100k_base | Tokenizer |
| `embedding_model_string` | "OpenAI" | Model name key |

**FaissRetriever Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `build_from_text()` | 101-126 | Build index from text |
| `build_from_leaf_nodes()` | 128-145 | Build from existing nodes |
| `sanity_check()` | 147-164 | Verify embeddings |
| `retrieve()` | 166-201 | Find similar chunks |

**Imports:**
- `random`
- `concurrent.futures.ProcessPoolExecutor`
- `faiss`, `numpy`, `tiktoken`, `tqdm`
- Internal: `EmbeddingModels`, `Retrievers`, `utils`

**Notes:**
- Uses `faiss.IndexFlatIP` (inner product/cosine similarity)
- Supports parallel embedding creation with ProcessPoolExecutor

---

## Dependency Graph

```
                            __init__.py
                                 |
        +------------------------+------------------------+
        |            |           |           |            |
        v            v           v           v            v
RetrievalAugmentation    tree_retriever    FaissRetriever
        |                    |                   |
        +--------+-----------+                   |
                 |                               |
        +--------v---------+          +---------+
        |                  |          |
        v                  v          v
cluster_tree_builder   tree_builder  utils
        |                  |          ^
        |                  |          |
        v                  v          |
  cluster_utils     +------+----------+
        |           |      |
        v           v      v
    tree_structures    EmbeddingModels
                       SummarizationModels
                       QAModels
                       Retrievers
```

### Import Dependencies Matrix

| File | Imports From |
|------|--------------|
| `__init__.py` | All other modules |
| `tree_structures.py` | typing |
| `tree_builder.py` | EmbeddingModels, SummarizationModels, tree_structures, utils |
| `cluster_tree_builder.py` | cluster_utils, tree_builder, tree_structures, utils |
| `cluster_utils.py` | tree_structures, utils |
| `tree_retriever.py` | EmbeddingModels, Retrievers, tree_structures, utils |
| `RetrievalAugmentation.py` | cluster_tree_builder, EmbeddingModels, QAModels, SummarizationModels, tree_builder, tree_retriever, tree_structures |
| `EmbeddingModels.py` | (none internal) |
| `SummarizationModels.py` | (none internal) |
| `QAModels.py` | (none internal) |
| `Retrievers.py` | (none internal) |
| `utils.py` | tree_structures |
| `FaissRetriever.py` | EmbeddingModels, Retrievers, utils |

---

## Issues & Observations

### Code Quality Issues

1. **Deprecated decorator** (`tree_builder.py:297`): Uses `@abstractclassmethod` instead of `@abstractmethod`

2. **Missing `self` parameter** (`cluster_utils.py:133`): `RAPTOR_Clustering.perform_clustering()` is missing `self` - appears to be a static method but not decorated as such

3. **Inconsistent error handling**: Some methods return exceptions as values instead of raising them (e.g., `SummarizationModels.py:44`)

4. **Duplicate code**: `GPT3TurboSummarizationModel` and `GPT3SummarizationModel` are nearly identical

5. **Hardcoded prompt strings**: Summarization and QA prompts are embedded in model classes

6. **No type hints on some returns**: Many methods lack return type annotations

### Architectural Observations

1. **Single tree builder type**: Only "cluster" is supported despite infrastructure for multiple types

2. **Tight OpenAI coupling**: Default models assume OpenAI API availability

3. **Pickle-based persistence**: Tree serialization uses pickle (not portable/secure)

4. **No streaming support**: All operations are synchronous/blocking

5. **Limited testing**: No test files in source directory
