# RAPTOR Package Diagram

## Code Organization

The RAPTOR codebase is organized as a single flat Python package with no sub-packages.

```
raptor/
    |
    +-- __init__.py              # Public API exports (17 lines)
    |
    +-- tree_structures.py       # Core data classes (29 lines)
    |
    +-- tree_builder.py          # Abstract builder + config (370 lines)
    |
    +-- cluster_tree_builder.py  # Concrete builder impl (152 lines)
    |
    +-- cluster_utils.py         # Clustering algorithms (186 lines)
    |
    +-- tree_retriever.py        # Tree-based retrieval (328 lines)
    |
    +-- FaissRetriever.py        # Flat vector retrieval (202 lines)
    |
    +-- Retrievers.py            # Retriever base class (9 lines)
    |
    +-- RetrievalAugmentation.py # Main facade class (307 lines)
    |
    +-- EmbeddingModels.py       # Embedding adapters (38 lines)
    |
    +-- SummarizationModels.py   # Summarization adapters (75 lines)
    |
    +-- QAModels.py              # QA model adapters (186 lines)
    |
    +-- utils.py                 # Shared utilities (209 lines)

Total: ~2,108 lines of code
```

## Logical Layers

Although physically flat, the code can be understood in logical layers:

```
+==========================================================================+
|                         LAYER DIAGRAM                                     |
+==========================================================================+

+--------------------------------------------------------------------------+
|                          PUBLIC API LAYER                                 |
|                                                                          |
|  __init__.py - Exports the public interface                              |
|                                                                          |
|  Exports:                                                                |
|    - RetrievalAugmentation, RetrievalAugmentationConfig                  |
|    - TreeBuilder, TreeBuilderConfig                                      |
|    - ClusterTreeBuilder, ClusterTreeConfig                               |
|    - TreeRetriever, TreeRetrieverConfig                                  |
|    - FaissRetriever, FaissRetrieverConfig                                |
|    - Node, Tree                                                          |
|    - Base*Model classes (for extension)                                  |
|    - Concrete model implementations                                      |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                         FACADE LAYER                                      |
|                                                                          |
|  RetrievalAugmentation.py                                                |
|                                                                          |
|  - Orchestrates tree building and retrieval                              |
|  - Provides simple user-facing API                                       |
|  - Handles persistence (save/load)                                       |
+--------------------------------------------------------------------------+
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
+---------------------------+               +---------------------------+
|     INDEXING LAYER        |               |     RETRIEVAL LAYER       |
|                           |               |                           |
|  tree_builder.py          |               |  tree_retriever.py        |
|  cluster_tree_builder.py  |               |  FaissRetriever.py        |
|  cluster_utils.py         |               |  Retrievers.py            |
|                           |               |                           |
|  Responsibilities:        |               |  Responsibilities:        |
|  - Text chunking          |               |  - Query embedding        |
|  - Clustering             |               |  - Similarity search      |
|  - Summarization          |               |  - Context assembly       |
|  - Tree construction      |               |  - Layer traversal        |
+---------------------------+               +---------------------------+
              |                                           |
              +---------------------+---------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                       MODEL ADAPTERS LAYER                               |
|                                                                          |
|  EmbeddingModels.py                                                      |
|  SummarizationModels.py                                                  |
|  QAModels.py                                                             |
|                                                                          |
|  - Abstract base classes                                                 |
|  - Concrete implementations for OpenAI, HuggingFace, etc.                |
|  - Retry logic and error handling                                        |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                         CORE DATA LAYER                                  |
|                                                                          |
|  tree_structures.py                                                      |
|                                                                          |
|  - Node class                                                            |
|  - Tree class                                                            |
|  - Pure data containers with no business logic                           |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                         UTILITIES LAYER                                  |
|                                                                          |
|  utils.py                                                                |
|                                                                          |
|  - Text processing (split_text)                                          |
|  - Distance calculations                                                 |
|  - Node/embedding extraction helpers                                     |
+--------------------------------------------------------------------------+
```

## Import Dependencies

### Import Direction Graph

```
                    External Dependencies
                    ====================
                    
    +----------+  +--------+  +-------+  +----------+  +-------+
    |  openai  |  |tiktoken|  |tenacity| | sentence |  | faiss |
    +----------+  +--------+  +-------+  |transformers+-------+
         |            |           |      +----------+      |
         |            |           |           |            |
    +----+------------+-----------+-----------+------------+----+
    |                        Model Adapters                     |
    |  EmbeddingModels.py  SummarizationModels.py  QAModels.py |
    +----------------------------------------------------------+
    
    +----------+  +-------+  +------+
    |  numpy   |  | scipy |  | umap |
    +----------+  +-------+  +------+
         |            |          |
    +----+------------+----------+----+
    |        sklearn (GMM)            |
    +---------------------------------+
              |
    +---------+---------+
    |   cluster_utils   |
    +-------------------+


                    Internal Dependencies
                    ====================

    RetrievalAugmentation.py
        |
        +---> cluster_tree_builder (ClusterTreeBuilder, ClusterTreeConfig)
        +---> tree_builder (TreeBuilder, TreeBuilderConfig)
        +---> tree_retriever (TreeRetriever, TreeRetrieverConfig)
        +---> tree_structures (Node, Tree)
        +---> EmbeddingModels (BaseEmbeddingModel)
        +---> SummarizationModels (BaseSummarizationModel)
        +---> QAModels (BaseQAModel, GPT3TurboQAModel)

    tree_builder.py
        |
        +---> EmbeddingModels (BaseEmbeddingModel, OpenAIEmbeddingModel)
        +---> SummarizationModels (BaseSummarizationModel, GPT3TurboSummarizationModel)
        +---> tree_structures (Node, Tree)
        +---> utils (distances_from_embeddings, get_children, get_embeddings, 
        |           get_node_list, get_text, indices_of_nearest_neighbors, split_text)

    cluster_tree_builder.py
        |
        +---> cluster_utils (ClusteringAlgorithm, RAPTOR_Clustering)
        +---> tree_builder (TreeBuilder, TreeBuilderConfig)
        +---> tree_structures (Node, Tree)
        +---> utils (get_node_list, get_text)

    tree_retriever.py
        |
        +---> EmbeddingModels (BaseEmbeddingModel, OpenAIEmbeddingModel)
        +---> Retrievers (BaseRetriever)
        +---> tree_structures (Node, Tree)
        +---> utils (distances_from_embeddings, get_children, get_embeddings,
        |           get_node_list, get_text, indices_of_nearest_neighbors, reverse_mapping)

    FaissRetriever.py
        |
        +---> EmbeddingModels (BaseEmbeddingModel, OpenAIEmbeddingModel)
        +---> Retrievers (BaseRetriever)
        +---> utils (split_text)

    cluster_utils.py
        |
        +---> tree_structures (Node)
        +---> utils (get_embeddings)

    utils.py
        |
        +---> tree_structures (Node)

    tree_structures.py
        |
        +---> (no internal dependencies - leaf module)

    Retrievers.py
        |
        +---> (no internal dependencies - leaf module)

    EmbeddingModels.py, SummarizationModels.py, QAModels.py
        |
        +---> (no internal dependencies - leaf modules)
```

### Dependency Matrix

```
                  IMPORTS FROM (columns)
                  |
IMPORTED BY       | tree_  | tree_    | cluster_ | cluster_ | tree_     | Faiss   | Retriev | Embed  | Summar | QA     | utils  |
(rows)            | struct | builder  | tree_b   | utils    | retriever | Retr    | ers     | Models | Models | Models |        |
------------------|--------|----------|----------|----------|-----------|---------|---------|--------|--------|--------|--------|
tree_structures   |   -    |          |          |          |           |         |         |        |        |        |        |
tree_builder      |   X    |    -     |          |          |           |         |         |   X    |   X    |        |   X    |
cluster_tree_b    |   X    |    X     |    -     |    X     |           |         |         |        |        |        |   X    |
cluster_utils     |   X    |          |          |    -     |           |         |         |        |        |        |   X    |
tree_retriever    |   X    |          |          |          |     -     |         |    X    |   X    |        |        |   X    |
FaissRetriever    |        |          |          |          |           |    -    |    X    |   X    |        |        |   X    |
RetrievalAug      |   X    |    X     |    X     |          |     X     |         |         |   X    |   X    |   X    |        |
utils             |   X    |          |          |          |           |         |         |        |        |        |   -    |
__init__          |   X    |    X     |    X     |          |     X     |    X    |    X    |   X    |   X    |   X    |        |
```

## Cohesion Analysis

### High Cohesion Modules

1. **tree_structures.py** - Pure data classes only
2. **utils.py** - Pure utility functions
3. **Retrievers.py** - Single abstract class

### Mixed Cohesion Modules

1. **tree_builder.py** - Contains both `TreeBuilderConfig` (data) and `TreeBuilder` (behavior)
2. **cluster_tree_builder.py** - Contains both `ClusterTreeConfig` and `ClusterTreeBuilder`
3. **tree_retriever.py** - Contains both `TreeRetrieverConfig` and `TreeRetriever`

### Potential Improvements

```
CURRENT STRUCTURE:                    SUGGESTED STRUCTURE:
==================                    ====================

raptor/                               raptor/
  __init__.py                           __init__.py
  tree_structures.py                    
  tree_builder.py                       core/
  cluster_tree_builder.py                 __init__.py
  cluster_utils.py                        node.py
  tree_retriever.py                       tree.py
  FaissRetriever.py                     
  Retrievers.py                         config/
  RetrievalAugmentation.py                __init__.py
  EmbeddingModels.py                      tree_builder_config.py
  SummarizationModels.py                  tree_retriever_config.py
  QAModels.py                             retrieval_augmentation_config.py
  utils.py                              
                                        builders/
                                          __init__.py
                                          base.py
                                          cluster.py
                                        
                                        retrievers/
                                          __init__.py
                                          base.py
                                          tree.py
                                          faiss.py
                                        
                                        models/
                                          __init__.py
                                          embedding/
                                            base.py
                                            openai.py
                                            sbert.py
                                          summarization/
                                            base.py
                                            openai.py
                                          qa/
                                            base.py
                                            openai.py
                                            unified.py
                                        
                                        clustering/
                                          __init__.py
                                          base.py
                                          raptor.py
                                          umap_gmm.py
                                        
                                        utils/
                                          __init__.py
                                          text.py
                                          distance.py
                                          node_ops.py
                                        
                                        facade.py  # RetrievalAugmentation
```

## Circular Dependencies

The current codebase has **no circular dependencies**. The dependency graph is a DAG (Directed Acyclic Graph).

Dependency order (can be imported in this sequence):
1. `tree_structures.py` (no dependencies)
2. `Retrievers.py` (no dependencies)
3. `EmbeddingModels.py`, `SummarizationModels.py`, `QAModels.py` (no internal deps)
4. `utils.py` (depends on tree_structures)
5. `cluster_utils.py` (depends on tree_structures, utils)
6. `tree_builder.py` (depends on tree_structures, utils, models)
7. `cluster_tree_builder.py` (depends on tree_builder, cluster_utils)
8. `tree_retriever.py` (depends on tree_structures, utils, models, Retrievers)
9. `FaissRetriever.py` (depends on utils, models, Retrievers)
10. `RetrievalAugmentation.py` (depends on everything)

## External Dependencies

| Module | External Packages |
|--------|-------------------|
| tree_builder.py | openai, tiktoken, tenacity |
| cluster_utils.py | numpy, tiktoken, umap, sklearn.mixture |
| tree_retriever.py | tiktoken, tenacity |
| FaissRetriever.py | faiss, numpy, tiktoken, tqdm |
| EmbeddingModels.py | openai, sentence_transformers, tenacity |
| SummarizationModels.py | openai, tenacity |
| QAModels.py | openai, torch, tenacity, transformers |
| utils.py | numpy, tiktoken, scipy |

### Dependency Summary

```
REQUIRED:
  - openai          (API calls)
  - tiktoken        (tokenization)
  - numpy           (numerical operations)
  - scipy           (distance calculations)

OPTIONAL (based on features used):
  - faiss           (if using FaissRetriever)
  - umap-learn      (for clustering)
  - sklearn         (for GMM clustering)
  - sentence-transformers (if using SBert)
  - torch           (if using UnifiedQA)
  - transformers    (if using UnifiedQA)

RESILIENCE:
  - tenacity        (retry logic for API calls)
  - tqdm            (progress bars)
```
