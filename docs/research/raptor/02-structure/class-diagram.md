# RAPTOR Class Diagram

## Complete Class Hierarchy

```
+==========================================================================+
|                          CLASS HIERARCHY OVERVIEW                         |
+==========================================================================+

                              ABC (Abstract Base Class)
                                        |
        +---------------+---------------+---------------+---------------+
        |               |               |               |               |
        v               v               v               v               v
BaseEmbedding    BaseSummarization   BaseQA      BaseRetriever   Clustering
   Model              Model          Model                       Algorithm
        |               |               |               |               |
        |               |               |               |               |
   +----+----+     +----+----+     +---+---+     +-----+-----+          |
   |         |     |         |     |   |   |     |           |          |
OpenAI   SBert  GPT3Turbo  GPT3  GPT3 GPT4 Unified TreeRetr FaissRetr RAPTOR_
Embed    Embed  Summ       Summ  QA   QA   QA                         Clustering


                    TreeBuilderConfig
                           |
                           v
                    ClusterTreeConfig


                      TreeBuilder (abstract)
                           |
                           v
                    ClusterTreeBuilder


                    TreeRetrieverConfig


                    FaissRetrieverConfig


                RetrievalAugmentationConfig


                  RetrievalAugmentation (facade)


                     Node (data class)


                     Tree (data class)
```

## Detailed Class Definitions

### Core Data Classes

```
+-----------------------------------------------------------------------+
|                              Node                                      |
+-----------------------------------------------------------------------+
| tree_structures.py:4                                                  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - text: str                  # The text content of this node        |
|   - index: int                 # Unique identifier within the tree    |
|   - children: Set[int]         # Indices of child nodes (empty=leaf)  |
|   - embeddings: Dict[str, Any] # Model name -> embedding vector       |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(text, index, children, embeddings) -> None               |
+-----------------------------------------------------------------------+
| Notes:                                                                |
|   - Leaf nodes have empty children set                                |
|   - Parent nodes reference children by index                          |
|   - Supports multiple embedding models simultaneously                 |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                              Tree                                      |
+-----------------------------------------------------------------------+
| tree_structures.py:16                                                 |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - all_nodes: Dict[int, Node]      # All nodes indexed by id         |
|   - root_nodes: Dict[int, Node]     # Top-level summary nodes         |
|   - leaf_nodes: Dict[int, Node]     # Bottom-level text chunks        |
|   - num_layers: int                 # Number of hierarchy levels      |
|   - layer_to_nodes: Dict[int, List[Node]]  # Layer num -> nodes       |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(all_nodes, root_nodes, leaf_nodes, num_layers,           |
|              layer_to_nodes) -> None                                  |
+-----------------------------------------------------------------------+
| Notes:                                                                |
|   - Layer 0 = leaf nodes (original text chunks)                       |
|   - Higher layers = progressively more summarized content             |
|   - Can be serialized with pickle for persistence                     |
+-----------------------------------------------------------------------+
```

### Configuration Classes

```
+-----------------------------------------------------------------------+
|                       TreeBuilderConfig                                |
+-----------------------------------------------------------------------+
| tree_builder.py:24                                                    |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tokenizer: Encoding          # tiktoken, default cl100k_base      |
|   - max_tokens: int              # Max tokens per chunk (default 100) |
|   - num_layers: int              # Max tree layers (default 5)        |
|   - threshold: float             # Similarity threshold (default 0.5) |
|   - top_k: int                   # Top-k selection (default 5)        |
|   - selection_mode: str          # "top_k" | "threshold"              |
|   - summarization_length: int    # Max summary tokens (default 100)   |
|   - summarization_model: BaseSummarizationModel                       |
|   - embedding_models: Dict[str, BaseEmbeddingModel]                   |
|   - cluster_embedding_model: str # Key into embedding_models dict     |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(...) -> None        # With defaults and validation       |
|   + log_config() -> str          # Returns formatted config string    |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                       ClusterTreeConfig                                |
+-----------------------------------------------------------------------+
| cluster_tree_builder.py:17                   extends TreeBuilderConfig |
+-----------------------------------------------------------------------+
| Additional Attributes:                                                |
|   - reduction_dimension: int     # UMAP target dimensions (default 10)|
|   - clustering_algorithm: type   # ClusteringAlgorithm class          |
|   - clustering_params: dict      # Additional clustering parameters   |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(...) -> None        # Calls super().__init__()           |
|   + log_config() -> str          # Extends parent log                 |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      TreeRetrieverConfig                               |
+-----------------------------------------------------------------------+
| tree_retriever.py:19                                                  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tokenizer: Encoding          # tiktoken, default cl100k_base      |
|   - threshold: float             # Similarity threshold (default 0.5) |
|   - top_k: int                   # Top-k selection (default 5)        |
|   - selection_mode: str          # "top_k" | "threshold"              |
|   - context_embedding_model: str # Key name (default "OpenAI")        |
|   - embedding_model: BaseEmbeddingModel  # For query embedding        |
|   - num_layers: int | None       # Layers to traverse                 |
|   - start_layer: int | None      # Starting layer for traversal      |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(...) -> None        # With defaults and validation       |
|   + log_config() -> str          # Returns formatted config string    |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      FaissRetrieverConfig                              |
+-----------------------------------------------------------------------+
| FaissRetriever.py:14                                                  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - max_tokens: int              # Tokens per chunk (default 100)     |
|   - max_context_tokens: int      # Max context size (default 3500)    |
|   - use_top_k: bool              # Use top_k vs token limit           |
|   - embedding_model: BaseEmbeddingModel                               |
|   - question_embedding_model: BaseEmbeddingModel                      |
|   - top_k: int                   # Number of results (default 5)      |
|   - tokenizer: Encoding          # tiktoken                           |
|   - embedding_model_string: str  # Key name (default "OpenAI")        |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(...) -> None        # With validation                    |
|   + log_config() -> str          # Returns formatted config string    |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                   RetrievalAugmentationConfig                          |
+-----------------------------------------------------------------------+
| RetrievalAugmentation.py:18                                           |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tree_builder_config: TreeBuilderConfig | ClusterTreeConfig        |
|   - tree_retriever_config: TreeRetrieverConfig                        |
|   - qa_model: BaseQAModel        # Default GPT3TurboQAModel           |
|   - tree_builder_type: str       # "cluster" (only supported type)    |
|   + Many tb_* and tr_* params for nested config construction          |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(...) -> None        # Complex config assembly            |
|   + log_config() -> str          # Returns full config tree           |
+-----------------------------------------------------------------------+
```

### Builder Classes

```
+-----------------------------------------------------------------------+
|                         TreeBuilder                                    |
+-----------------------------------------------------------------------+
| tree_builder.py:133                                          ABSTRACT |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tokenizer: Encoding                                               |
|   - max_tokens: int                                                   |
|   - num_layers: int                                                   |
|   - top_k: int                                                        |
|   - threshold: float                                                  |
|   - selection_mode: str                                               |
|   - summarization_length: int                                         |
|   - summarization_model: BaseSummarizationModel                       |
|   - embedding_models: Dict[str, BaseEmbeddingModel]                   |
|   - cluster_embedding_model: str                                      |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(config: TreeBuilderConfig) -> None                       |
|   + create_node(index, text, children_indices) -> Tuple[int, Node]    |
|   + create_embedding(text) -> List[float]                             |
|   + summarize(context, max_tokens) -> str                             |
|   + get_relevant_nodes(current_node, list_nodes) -> List[Node]        |
|   + multithreaded_create_leaf_nodes(chunks) -> Dict[int, Node]        |
|   + build_from_text(text, use_multithreading) -> Tree                 |
|   @ construct_tree(...) -> Dict[int, Node]         # ABSTRACT METHOD  |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      ClusterTreeBuilder                                |
+-----------------------------------------------------------------------+
| cluster_tree_builder.py:41                       extends TreeBuilder  |
+-----------------------------------------------------------------------+
| Additional Attributes:                                                |
|   - reduction_dimension: int                                          |
|   - clustering_algorithm: type[ClusteringAlgorithm]                   |
|   - clustering_params: dict                                           |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(config: ClusterTreeConfig) -> None                       |
|   + construct_tree(current_level_nodes, all_tree_nodes,               |
|                    layer_to_nodes, use_multithreading) -> Dict        |
+-----------------------------------------------------------------------+
| Notes:                                                                |
|   - Implements clustering-based tree construction                     |
|   - Uses RAPTOR_Clustering by default                                 |
|   - Stops building when nodes <= reduction_dimension + 1              |
+-----------------------------------------------------------------------+
```

### Retriever Classes

```
+-----------------------------------------------------------------------+
|                        BaseRetriever                                   |
+-----------------------------------------------------------------------+
| Retrievers.py:5                                              ABSTRACT |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @ retrieve(query: str) -> str                    # ABSTRACT METHOD  |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                        TreeRetriever                                   |
+-----------------------------------------------------------------------+
| tree_retriever.py:106                           extends BaseRetriever |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tree: Tree                                                        |
|   - num_layers: int                                                   |
|   - start_layer: int                                                  |
|   - tokenizer: Encoding                                               |
|   - top_k: int                                                        |
|   - threshold: float                                                  |
|   - selection_mode: str                                               |
|   - embedding_model: BaseEmbeddingModel                               |
|   - context_embedding_model: str                                      |
|   - tree_node_index_to_layer: Dict[int, int]                          |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(config, tree) -> None                                    |
|   + create_embedding(text) -> List[float]                             |
|   + retrieve_information_collapse_tree(query, top_k, max_tokens)      |
|       -> Tuple[List[Node], str]                                       |
|   + retrieve_information(current_nodes, query, num_layers)            |
|       -> Tuple[List[Node], str]                                       |
|   + retrieve(query, start_layer, num_layers, top_k, max_tokens,       |
|              collapse_tree, return_layer_information) -> str | tuple  |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                        FaissRetriever                                  |
+-----------------------------------------------------------------------+
| FaissRetriever.py:82                            extends BaseRetriever |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - embedding_model: BaseEmbeddingModel                               |
|   - question_embedding_model: BaseEmbeddingModel                      |
|   - index: faiss.IndexFlatIP | None                                   |
|   - context_chunks: np.ndarray | None                                 |
|   - embeddings: np.ndarray                                            |
|   - max_tokens: int                                                   |
|   - max_context_tokens: int                                           |
|   - use_top_k: bool                                                   |
|   - tokenizer: Encoding                                               |
|   - top_k: int                                                        |
|   - embedding_model_string: str                                       |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(config) -> None                                          |
|   + build_from_text(doc_text) -> None                                 |
|   + build_from_leaf_nodes(leaf_nodes) -> None                         |
|   + sanity_check(num_samples) -> None                                 |
|   + retrieve(query) -> str                                            |
+-----------------------------------------------------------------------+
```

### Clustering Classes

```
+-----------------------------------------------------------------------+
|                     ClusteringAlgorithm                                |
+-----------------------------------------------------------------------+
| cluster_utils.py:126                                         ABSTRACT |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @ perform_clustering(embeddings, **kwargs) -> List[List[int]]       |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      RAPTOR_Clustering                                 |
+-----------------------------------------------------------------------+
| cluster_utils.py:132                   extends ClusteringAlgorithm    |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @staticmethod                                                       |
|   + perform_clustering(                                               |
|       nodes: List[Node],                                              |
|       embedding_model_name: str,                                      |
|       max_length_in_cluster: int = 3500,                              |
|       tokenizer = tiktoken,                                           |
|       reduction_dimension: int = 10,                                  |
|       threshold: float = 0.1,                                         |
|       verbose: bool = False                                           |
|     ) -> List[List[Node]]                                             |
+-----------------------------------------------------------------------+
| Notes:                                                                |
|   - Uses UMAP for dimensionality reduction                            |
|   - Uses GMM (Gaussian Mixture Model) for clustering                  |
|   - Recursively re-clusters if cluster too large                      |
+-----------------------------------------------------------------------+
```

### Model Adapter Classes

```
+-----------------------------------------------------------------------+
|                      BaseEmbeddingModel                                |
+-----------------------------------------------------------------------+
| EmbeddingModels.py:11                                        ABSTRACT |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @ create_embedding(text) -> List[float] | np.ndarray                |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                     OpenAIEmbeddingModel                               |
+-----------------------------------------------------------------------+
| EmbeddingModels.py:17                      extends BaseEmbeddingModel |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - client: OpenAI                                                    |
|   - model: str                   # default "text-embedding-ada-002"   |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + create_embedding(text) -> List[float]     # with @retry decorator |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      SBertEmbeddingModel                               |
+-----------------------------------------------------------------------+
| EmbeddingModels.py:32                      extends BaseEmbeddingModel |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: SentenceTransformer                                        |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model_name: str) -> None                                 |
|   + create_embedding(text) -> np.ndarray                              |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                    BaseSummarizationModel                              |
+-----------------------------------------------------------------------+
| SummarizationModels.py:11                                    ABSTRACT |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @ summarize(context, max_tokens) -> str                             |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                  GPT3TurboSummarizationModel                           |
+-----------------------------------------------------------------------+
| SummarizationModels.py:17               extends BaseSummarizationModel|
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: str                   # default "gpt-3.5-turbo"            |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + summarize(context, max_tokens, stop_sequence) -> str | Exception  |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                    GPT3SummarizationModel                              |
+-----------------------------------------------------------------------+
| SummarizationModels.py:47               extends BaseSummarizationModel|
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: str                   # default "text-davinci-003"         |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + summarize(context, max_tokens, stop_sequence) -> str | Exception  |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                         BaseQAModel                                    |
+-----------------------------------------------------------------------+
| QAModels.py:15                                               ABSTRACT |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   @ answer_question(context, question) -> str                         |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                        GPT3QAModel                                     |
+-----------------------------------------------------------------------+
| QAModels.py:21                                   extends BaseQAModel  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: str                   # default "text-davinci-003"         |
|   - client: OpenAI                                                    |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + answer_question(context, question, max_tokens, stop_sequence)     |
|       -> str                                     # uses completions   |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                      GPT3TurboQAModel                                  |
+-----------------------------------------------------------------------+
| QAModels.py:63                                   extends BaseQAModel  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: str                   # default "gpt-3.5-turbo"            |
|   - client: OpenAI                                                    |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + _attempt_answer_question(...) -> str        # private helper      |
|   + answer_question(context, question, ...) -> str | Exception        |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                         GPT4QAModel                                    |
+-----------------------------------------------------------------------+
| QAModels.py:115                                  extends BaseQAModel  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - model: str                   # default "gpt-4"                    |
|   - client: OpenAI                                                    |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model: str) -> None                                      |
|   + _attempt_answer_question(...) -> str        # private helper      |
|   + answer_question(context, question, ...) -> str | Exception        |
+-----------------------------------------------------------------------+


+-----------------------------------------------------------------------+
|                        UnifiedQAModel                                  |
+-----------------------------------------------------------------------+
| QAModels.py:167                                  extends BaseQAModel  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - device: torch.device                                              |
|   - model: T5ForConditionalGeneration                                 |
|   - tokenizer: T5Tokenizer                                            |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(model_name: str) -> None                                 |
|   + run_model(input_string, **generator_args) -> List[str]            |
|   + answer_question(context, question) -> str                         |
+-----------------------------------------------------------------------+
```

### Facade Class

```
+-----------------------------------------------------------------------+
|                     RetrievalAugmentation                              |
+-----------------------------------------------------------------------+
| RetrievalAugmentation.py:153                                  FACADE  |
+-----------------------------------------------------------------------+
| Attributes:                                                           |
|   - tree: Tree | None                                                 |
|   - tree_builder: TreeBuilder (ClusterTreeBuilder)                    |
|   - tree_retriever_config: TreeRetrieverConfig                        |
|   - qa_model: BaseQAModel                                             |
|   - retriever: TreeRetriever | None                                   |
+-----------------------------------------------------------------------+
| Methods:                                                              |
|   + __init__(config: RAConfig | None, tree: Tree | str | None)        |
|   + add_documents(docs: str) -> None                                  |
|   + retrieve(question, start_layer, num_layers, top_k, max_tokens,    |
|              collapse_tree, return_layer_information) -> str | tuple  |
|   + answer_question(question, top_k, start_layer, num_layers,         |
|                     max_tokens, collapse_tree,                        |
|                     return_layer_information) -> str | tuple          |
|   + save(path: str) -> None                                           |
+-----------------------------------------------------------------------+
```

## Relationship Diagram

```
                         INHERITANCE (extends)
                               |
  +----------------------------+----------------------------+
  |                            |                            |
BaseEmbedding             BaseQAModel              BaseSummarization
   Model                       |                       Model
     ^                    +----+----+                     ^
     |                    |    |    |                     |
+----+----+             GPT3  GPT4 Unified          +-----+-----+
|         |              QA    QA    QA             |           |
OpenAI   SBert                                   GPT3Turbo    GPT3
Embed    Embed                                     Summ       Summ


                 BaseRetriever              ClusteringAlgorithm
                      ^                            ^
                      |                            |
              +-------+-------+                    |
              |               |              RAPTOR_Clustering
         TreeRetriever   FaissRetriever


                   TreeBuilderConfig
                          ^
                          |
                   ClusterTreeConfig


                     TreeBuilder (abstract)
                          ^
                          |
                   ClusterTreeBuilder


              COMPOSITION (has-a)
              ===================

RetrievalAugmentation
    |
    +---> TreeBuilder (tree_builder)
    +---> TreeRetrieverConfig (tree_retriever_config)
    +---> BaseQAModel (qa_model)
    +---> TreeRetriever (retriever)
    +---> Tree (tree)


TreeBuilder
    |
    +---> Dict[str, BaseEmbeddingModel] (embedding_models)
    +---> BaseSummarizationModel (summarization_model)


TreeRetriever
    |
    +---> Tree (tree)
    +---> BaseEmbeddingModel (embedding_model)


Tree
    |
    +---> Dict[int, Node] (all_nodes, root_nodes, leaf_nodes)


Node
    |
    +---> Dict[str, List[float]] (embeddings)
    +---> Set[int] (children) --> references other Nodes by index


ClusterTreeBuilder
    |
    +---> ClusteringAlgorithm class (clustering_algorithm)


              DEPENDENCY (uses)
              =================

TreeBuilder ---------> utils.split_text()
                   \-> utils.get_text()
                   \-> utils.get_embeddings()

TreeRetriever ------> utils.distances_from_embeddings()
                  \-> utils.indices_of_nearest_neighbors()
                  \-> utils.get_text()
                  \-> utils.get_embeddings()
                  \-> utils.reverse_mapping()

ClusterTreeBuilder -> cluster_utils.RAPTOR_Clustering

RAPTOR_Clustering --> cluster_utils.perform_clustering()
                  \-> cluster_utils.global_cluster_embeddings()
                  \-> cluster_utils.GMM_cluster()
```
