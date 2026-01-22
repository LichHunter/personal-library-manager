# RAPTOR Configuration Reference

> Complete documentation of all configuration classes and their parameters

## Overview

RAPTOR uses a hierarchical configuration system with four main config classes:

```
RetrievalAugmentationConfig
    |-- TreeBuilderConfig (or ClusterTreeConfig)
    |-- TreeRetrieverConfig
    +-- QA Model reference
```

Additionally, there's a standalone `FaissRetrieverConfig` for the FAISS-based retriever.

---

## 1. TreeBuilderConfig

**Location:** `raptor/tree_builder.py:24-130`

Base configuration class for tree construction.

### Parameters

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `tokenizer` | Encoding | `tiktoken.get_encoding("cl100k_base")` | Any tiktoken encoding | Tokenizer for splitting text and counting tokens |
| `max_tokens` | int | `100` | >= 1 | Maximum tokens per text chunk (leaf node) |
| `num_layers` | int | `5` | >= 1 | Maximum number of tree layers to build |
| `threshold` | float | `0.5` | 0.0 - 1.0 | Similarity threshold for "threshold" selection mode |
| `top_k` | int | `5` | >= 1 | Number of similar nodes to select in "top_k" mode |
| `selection_mode` | str | `"top_k"` | `"top_k"`, `"threshold"` | How to select relevant nodes |
| `summarization_length` | int | `100` | Any positive int | Max tokens for generated summaries |
| `summarization_model` | BaseSummarizationModel | `GPT3TurboSummarizationModel()` | Any BaseSummarizationModel | Model used for summarization |
| `embedding_models` | Dict[str, BaseEmbeddingModel] | `{"OpenAI": OpenAIEmbeddingModel()}` | Dict of name -> model | Embedding models (can have multiple) |
| `cluster_embedding_model` | str | `"OpenAI"` | Key from embedding_models dict | Which embedding model to use for clustering |

### Validation Rules

```python
# From source code validation:
if not isinstance(max_tokens, int) or max_tokens < 1:
    raise ValueError("max_tokens must be an integer and at least 1")

if not isinstance(num_layers, int) or num_layers < 1:
    raise ValueError("num_layers must be an integer and at least 1")

if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
    raise ValueError("threshold must be a number between 0 and 1")

if not isinstance(top_k, int) or top_k < 1:
    raise ValueError("top_k must be an integer and at least 1")

if selection_mode not in ["top_k", "threshold"]:
    raise ValueError("selection_mode must be either 'top_k' or 'threshold'")

if not isinstance(summarization_model, BaseSummarizationModel):
    raise ValueError("summarization_model must be an instance of BaseSummarizationModel")

if not isinstance(embedding_models, dict):
    raise ValueError("embedding_models must be a dictionary of model_name: instance pairs")

for model in embedding_models.values():
    if not isinstance(model, BaseEmbeddingModel):
        raise ValueError("All embedding models must be an instance of BaseEmbeddingModel")

if cluster_embedding_model not in self.embedding_models:
    raise ValueError("cluster_embedding_model must be a key in the embedding_models dictionary")
```

### Parameter Effects

#### `max_tokens`
- Controls the granularity of leaf nodes
- **Lower values (50-100):** More granular chunking, better for detailed retrieval, more nodes
- **Higher values (200-500):** Larger chunks, fewer nodes, faster tree building

#### `num_layers`
- Sets the maximum depth of the tree
- **Note:** Tree building may stop early if there aren't enough nodes to cluster
- Each layer reduces the number of nodes through summarization

#### `selection_mode` and related parameters
- **`"top_k"`:** Always selects exactly `top_k` similar nodes
- **`"threshold"`:** Selects nodes with similarity above `threshold`
- Used internally during tree building for finding related nodes

#### `summarization_length`
- Controls how long summaries are at each tree level
- **Lower values:** More compression, potential information loss
- **Higher values:** More detailed summaries, larger tree

### Example Configuration

```python
from raptor import TreeBuilderConfig, SBertEmbeddingModel
import tiktoken

config = TreeBuilderConfig(
    tokenizer=tiktoken.get_encoding("cl100k_base"),
    max_tokens=150,           # Slightly larger chunks
    num_layers=4,             # Fewer layers
    threshold=0.7,            # Higher threshold
    top_k=3,                  # Fewer related nodes
    selection_mode="threshold",
    summarization_length=200, # Longer summaries
    embedding_models={
        "SBERT": SBertEmbeddingModel()
    },
    cluster_embedding_model="SBERT"
)
```

---

## 2. ClusterTreeConfig

**Location:** `raptor/cluster_tree_builder.py:17-38`

Extends `TreeBuilderConfig` with clustering-specific parameters. This is the default configuration type used by RAPTOR.

### Additional Parameters (Beyond TreeBuilderConfig)

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `reduction_dimension` | int | `10` | Positive int | Target dimensionality for UMAP reduction |
| `clustering_algorithm` | ClusteringAlgorithm | `RAPTOR_Clustering` | Any ClusteringAlgorithm class | Clustering algorithm to use |
| `clustering_params` | dict | `{}` | Dict | Additional parameters passed to clustering algorithm |

### Parameter Effects

#### `reduction_dimension`
- UMAP reduces embeddings to this dimensionality before clustering
- **Lower values (5-10):** More aggressive reduction, faster but may lose information
- **Higher values (15-30):** Preserves more information, but slower
- **Note:** If `len(nodes) <= reduction_dimension + 1`, tree building stops at that layer

#### `clustering_algorithm`
- Defines the clustering strategy
- Default `RAPTOR_Clustering` uses:
  1. Global UMAP + GMM clustering
  2. Local clustering within global clusters
  3. Recursive re-clustering for large clusters

#### `clustering_params`
- Passed as `**kwargs` to `clustering_algorithm.perform_clustering()`
- For `RAPTOR_Clustering`, available params:
  - `max_length_in_cluster` (int, default: 3500): Max tokens per cluster before re-clustering
  - `threshold` (float, default: 0.1): GMM probability threshold for cluster assignment
  - `verbose` (bool, default: False): Enable detailed logging

### Example Configuration

```python
from raptor import ClusterTreeConfig, RAPTOR_Clustering

config = ClusterTreeConfig(
    # Inherited from TreeBuilderConfig
    max_tokens=100,
    num_layers=5,
    summarization_length=150,
    
    # ClusterTreeConfig specific
    reduction_dimension=15,
    clustering_algorithm=RAPTOR_Clustering,
    clustering_params={
        "max_length_in_cluster": 5000,
        "threshold": 0.15,
        "verbose": True
    }
)
```

---

## 3. TreeRetrieverConfig

**Location:** `raptor/tree_retriever.py:19-103`

Configuration for the tree-based retriever.

### Parameters

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `tokenizer` | Encoding | `tiktoken.get_encoding("cl100k_base")` | Any tiktoken encoding | Tokenizer for counting tokens |
| `threshold` | float | `0.5` | 0.0 - 1.0 | Similarity threshold for "threshold" selection mode |
| `top_k` | int | `5` | >= 1 | Number of nodes to retrieve in "top_k" mode |
| `selection_mode` | str | `"top_k"` | `"top_k"`, `"threshold"` | Node selection strategy |
| `context_embedding_model` | str | `"OpenAI"` | Key from tree's embedding_models | Which stored embeddings to use for comparison |
| `embedding_model` | BaseEmbeddingModel | `OpenAIEmbeddingModel()` | Any BaseEmbeddingModel | Model to embed the query |
| `num_layers` | int or None | `None` | >= 0 or None | Number of layers to traverse (None = all) |
| `start_layer` | int or None | `None` | >= 0 or None | Layer to start retrieval from (None = top) |

### Validation Rules

```python
if not isinstance(threshold, float) or not (0 <= threshold <= 1):
    raise ValueError("threshold must be a float between 0 and 1")

if not isinstance(top_k, int) or top_k < 1:
    raise ValueError("top_k must be an integer and at least 1")

if selection_mode not in ["top_k", "threshold"]:
    raise ValueError("selection_mode must be a string and either 'top_k' or 'threshold'")

if not isinstance(context_embedding_model, str):
    raise ValueError("context_embedding_model must be a string")

if not isinstance(embedding_model, BaseEmbeddingModel):
    raise ValueError("embedding_model must be an instance of BaseEmbeddingModel")

if num_layers is not None:
    if not isinstance(num_layers, int) or num_layers < 0:
        raise ValueError("num_layers must be an integer and at least 0")

if start_layer is not None:
    if not isinstance(start_layer, int) or start_layer < 0:
        raise ValueError("start_layer must be an integer and at least 0")
```

### Additional Validation at Runtime

When `TreeRetriever` is initialized with a tree:

```python
if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
    raise ValueError("num_layers in config must be less than or equal to tree.num_layers + 1")

if config.start_layer is not None and config.start_layer > tree.num_layers:
    raise ValueError("start_layer in config must be less than or equal to tree.num_layers")

if self.num_layers > self.start_layer + 1:
    raise ValueError("num_layers must be less than or equal to start_layer + 1")
```

### Parameter Effects

#### `selection_mode`, `threshold`, `top_k`
- **`"top_k"`:** Retrieves exactly `top_k` most similar nodes at each layer
- **`"threshold"`:** Retrieves all nodes with similarity > `threshold`

#### `context_embedding_model` vs `embedding_model`
- **`context_embedding_model`:** String key to look up pre-computed node embeddings
- **`embedding_model`:** Model instance used to embed the query at runtime
- **Important:** These should be compatible (same embedding space)

#### `start_layer` and `num_layers`
For tree traversal mode (not collapsed):
- **`start_layer`:** Which layer to begin searching (0 = leaves, max = root)
- **`num_layers`:** How many layers to traverse downward
- **Default behavior:** Start at root, traverse all layers to leaves

### Example Configuration

```python
from raptor import TreeRetrieverConfig, SBertEmbeddingModel

config = TreeRetrieverConfig(
    threshold=0.6,
    top_k=10,
    selection_mode="top_k",
    context_embedding_model="OpenAI",  # Must match tree's embedding model key
    embedding_model=OpenAIEmbeddingModel(),
    start_layer=None,  # Start at top
    num_layers=None,   # Traverse all layers
)
```

---

## 4. RetrievalAugmentationConfig

**Location:** `raptor/RetrievalAugmentation.py:18-150`

The main configuration class that orchestrates all components.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tree_builder_config` | TreeBuilderConfig | None (auto-created) | Full tree builder configuration |
| `tree_retriever_config` | TreeRetrieverConfig | None (auto-created) | Full retriever configuration |
| `qa_model` | BaseQAModel | `GPT3TurboQAModel()` | Model for answering questions |
| `embedding_model` | BaseEmbeddingModel | None | Convenience param: sets both builder and retriever models |
| `summarization_model` | BaseSummarizationModel | None | Convenience param: sets tree builder summarization |
| `tree_builder_type` | str | `"cluster"` | Currently only `"cluster"` supported |

### Shortcut Parameters

These parameters provide shortcuts for common TreeRetrieverConfig settings:

| Parameter | Maps To | Default |
|-----------|---------|---------|
| `tr_tokenizer` | TreeRetrieverConfig.tokenizer | None |
| `tr_threshold` | TreeRetrieverConfig.threshold | 0.5 |
| `tr_top_k` | TreeRetrieverConfig.top_k | 5 |
| `tr_selection_mode` | TreeRetrieverConfig.selection_mode | "top_k" |
| `tr_context_embedding_model` | TreeRetrieverConfig.context_embedding_model | "OpenAI" |
| `tr_embedding_model` | TreeRetrieverConfig.embedding_model | None |
| `tr_num_layers` | TreeRetrieverConfig.num_layers | None |
| `tr_start_layer` | TreeRetrieverConfig.start_layer | None |

These parameters provide shortcuts for common TreeBuilderConfig settings:

| Parameter | Maps To | Default |
|-----------|---------|---------|
| `tb_tokenizer` | TreeBuilderConfig.tokenizer | None |
| `tb_max_tokens` | TreeBuilderConfig.max_tokens | 100 |
| `tb_num_layers` | TreeBuilderConfig.num_layers | 5 |
| `tb_threshold` | TreeBuilderConfig.threshold | 0.5 |
| `tb_top_k` | TreeBuilderConfig.top_k | 5 |
| `tb_selection_mode` | TreeBuilderConfig.selection_mode | "top_k" |
| `tb_summarization_length` | TreeBuilderConfig.summarization_length | 100 |
| `tb_summarization_model` | TreeBuilderConfig.summarization_model | None |
| `tb_embedding_models` | TreeBuilderConfig.embedding_models | None |
| `tb_cluster_embedding_model` | TreeBuilderConfig.cluster_embedding_model | "OpenAI" |

### Validation Rules

```python
if tree_builder_type not in supported_tree_builders:
    raise ValueError(f"tree_builder_type must be one of {list(supported_tree_builders.keys())}")

if qa_model is not None and not isinstance(qa_model, BaseQAModel):
    raise ValueError("qa_model must be an instance of BaseQAModel")

if embedding_model is not None and not isinstance(embedding_model, BaseEmbeddingModel):
    raise ValueError("embedding_model must be an instance of BaseEmbeddingModel")

# Can't specify both convenience param and full config
if embedding_model is not None and tb_embedding_models is not None:
    raise ValueError("Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both.")

if summarization_model is not None and not isinstance(summarization_model, BaseSummarizationModel):
    raise ValueError("summarization_model must be an instance of BaseSummarizationModel")

if summarization_model is not None and tb_summarization_model is not None:
    raise ValueError("Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both.")
```

### Configuration Precedence

When `embedding_model` convenience parameter is provided:
1. `tb_embedding_models` is set to `{"EMB": embedding_model}`
2. `tr_embedding_model` is set to `embedding_model`
3. `tb_cluster_embedding_model` is set to `"EMB"`
4. `tr_context_embedding_model` is set to `"EMB"`

When `summarization_model` convenience parameter is provided:
1. `tb_summarization_model` is set to `summarization_model`

### Example Configurations

#### Minimal Configuration (All Defaults)

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

# Uses OpenAI for everything
config = RetrievalAugmentationConfig()
RA = RetrievalAugmentation(config=config)
```

#### Custom Models via Convenience Parameters

```python
from raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig,
    SBertEmbeddingModel,
    GPT4QAModel
)

config = RetrievalAugmentationConfig(
    embedding_model=SBertEmbeddingModel(),  # Used everywhere
    qa_model=GPT4QAModel(),
    tb_max_tokens=150,
    tb_num_layers=4,
    tr_top_k=10,
)
RA = RetrievalAugmentation(config=config)
```

#### Full Custom Configuration

```python
from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    ClusterTreeConfig,
    TreeRetrieverConfig,
    SBertEmbeddingModel,
    OpenAIEmbeddingModel,
    GPT4QAModel
)

# Create tree builder config
tree_builder_config = ClusterTreeConfig(
    max_tokens=200,
    num_layers=6,
    summarization_length=150,
    embedding_models={
        "SBERT": SBertEmbeddingModel(),
        "OpenAI": OpenAIEmbeddingModel()
    },
    cluster_embedding_model="SBERT",
    reduction_dimension=12,
)

# Create retriever config
retriever_config = TreeRetrieverConfig(
    top_k=15,
    selection_mode="top_k",
    context_embedding_model="SBERT",
    embedding_model=SBertEmbeddingModel(),
)

# Combine into main config
config = RetrievalAugmentationConfig(
    tree_builder_config=tree_builder_config,
    tree_retriever_config=retriever_config,
    qa_model=GPT4QAModel(),
)

RA = RetrievalAugmentation(config=config)
```

---

## 5. FaissRetrieverConfig

**Location:** `raptor/FaissRetriever.py:14-79`

Configuration for the standalone FAISS-based retriever (alternative to tree-based retrieval).

### Parameters

| Parameter | Type | Default | Valid Values | Description |
|-----------|------|---------|--------------|-------------|
| `max_tokens` | int | `100` | >= 1 | Maximum tokens per chunk |
| `max_context_tokens` | int | `3500` | >= 1 or None | Max tokens in returned context |
| `use_top_k` | bool | `False` | True/False | Use top-k vs token-limited retrieval |
| `embedding_model` | BaseEmbeddingModel | `OpenAIEmbeddingModel()` | Any BaseEmbeddingModel | Model for document embeddings |
| `question_embedding_model` | BaseEmbeddingModel | Same as embedding_model | Any BaseEmbeddingModel | Model for query embeddings |
| `top_k` | int | `5` | >= 1 | Number of chunks to retrieve (if use_top_k=True) |
| `tokenizer` | Encoding | `tiktoken.get_encoding("cl100k_base")` | Any tiktoken encoding | Tokenizer for counting |
| `embedding_model_string` | str | `"OpenAI"` | Any string | Key name for embeddings (when using tree nodes) |

### Validation Rules

```python
if max_tokens < 1:
    raise ValueError("max_tokens must be at least 1")

if top_k < 1:
    raise ValueError("top_k must be at least 1")

if max_context_tokens is not None and max_context_tokens < 1:
    raise ValueError("max_context_tokens must be at least 1 or None")

if embedding_model is not None and not isinstance(embedding_model, BaseEmbeddingModel):
    raise ValueError("embedding_model must be an instance of BaseEmbeddingModel or None")

if question_embedding_model is not None and not isinstance(question_embedding_model, BaseEmbeddingModel):
    raise ValueError("question_embedding_model must be an instance of BaseEmbeddingModel or None")
```

### Retrieval Modes

#### Token-Limited Mode (`use_top_k=False`, default)
- Retrieves chunks until `max_context_tokens` is reached
- Number of chunks varies based on chunk sizes

#### Top-K Mode (`use_top_k=True`)
- Retrieves exactly `top_k` chunks
- Ignores `max_context_tokens`

### Example Configuration

```python
from raptor import FaissRetriever, FaissRetrieverConfig, SBertEmbeddingModel

config = FaissRetrieverConfig(
    max_tokens=150,
    max_context_tokens=4000,
    use_top_k=False,
    embedding_model=SBertEmbeddingModel(),
    top_k=10,
)

retriever = FaissRetriever(config)
retriever.build_from_text("Your document text...")
context = retriever.retrieve("What is the main topic?")
```

---

## Use Case Examples

### Use Case 1: Large Documents with Deep Summarization

```python
config = RetrievalAugmentationConfig(
    tb_max_tokens=150,          # Moderate chunk size
    tb_num_layers=7,            # Deep tree
    tb_summarization_length=200, # Detailed summaries
    tr_top_k=15,                # Retrieve more context
)
```

### Use Case 2: Fast Processing with Local Models

```python
from raptor import SBertEmbeddingModel

config = RetrievalAugmentationConfig(
    embedding_model=SBertEmbeddingModel("all-MiniLM-L6-v2"),  # Fast local model
    tb_max_tokens=200,          # Larger chunks = fewer nodes
    tb_num_layers=3,            # Shallow tree
    tb_summarization_length=100, # Shorter summaries
)
```

### Use Case 3: High Precision Retrieval

```python
config = RetrievalAugmentationConfig(
    tb_max_tokens=75,           # Fine-grained chunks
    tb_num_layers=5,
    tr_selection_mode="threshold",
    tr_threshold=0.7,           # High similarity threshold
)
```

### Use Case 4: Multiple Embedding Models

```python
from raptor import ClusterTreeConfig, OpenAIEmbeddingModel, SBertEmbeddingModel

tree_config = ClusterTreeConfig(
    embedding_models={
        "OpenAI": OpenAIEmbeddingModel(),
        "SBERT": SBertEmbeddingModel(),
    },
    cluster_embedding_model="SBERT",  # Use SBERT for clustering
)

# At retrieval time, can query using either embedding
```

---

## Configuration Summary Table

| Config Class | Key Parameters | When to Use |
|--------------|----------------|-------------|
| `TreeBuilderConfig` | max_tokens, num_layers, summarization_length | Base tree building |
| `ClusterTreeConfig` | + reduction_dimension, clustering_params | Default RAPTOR tree building |
| `TreeRetrieverConfig` | top_k, selection_mode, start_layer | Tree-based retrieval |
| `RetrievalAugmentationConfig` | Combines all + qa_model | Main entry point |
| `FaissRetrieverConfig` | max_context_tokens, use_top_k | Simple flat retrieval |
