# RAPTOR API Surface

> Documentation of the public API, entry points, and usage patterns

## Overview

RAPTOR exposes its functionality through a main facade class (`RetrievalAugmentation`) and several extensible base classes for customization. The API is designed for simplicity in common cases with extensive configuration options for advanced use.

---

## Public API (from `__init__.py`)

### Main Classes

| Class | Purpose | Primary Methods |
|-------|---------|-----------------|
| `RetrievalAugmentation` | Main facade - build, retrieve, answer | `add_documents()`, `retrieve()`, `answer_question()`, `save()` |
| `RetrievalAugmentationConfig` | Master configuration | Constructor with all options |

### Tree Building

| Class | Purpose |
|-------|---------|
| `TreeBuilder` | Abstract base for tree builders |
| `TreeBuilderConfig` | Tree builder configuration |
| `ClusterTreeBuilder` | Concrete clustering-based builder |
| `ClusterTreeConfig` | Cluster builder configuration |

### Retrieval

| Class | Purpose |
|-------|---------|
| `TreeRetriever` | Hierarchical tree retrieval |
| `TreeRetrieverConfig` | Tree retriever configuration |
| `FaissRetriever` | Flat FAISS-based retrieval |
| `FaissRetrieverConfig` | FAISS retriever configuration |
| `BaseRetriever` | Abstract retriever interface |

### Data Structures

| Class | Purpose |
|-------|---------|
| `Node` | Single node in the tree |
| `Tree` | Complete tree structure |

### Models (Extensible)

| Class | Purpose |
|-------|---------|
| `BaseEmbeddingModel` | Abstract embedding interface |
| `OpenAIEmbeddingModel` | OpenAI embeddings |
| `SBertEmbeddingModel` | Sentence-BERT embeddings |
| `BaseSummarizationModel` | Abstract summarization interface |
| `GPT3SummarizationModel` | GPT-3 (text-davinci-003) summarizer |
| `GPT3TurboSummarizationModel` | GPT-3.5-turbo summarizer |
| `BaseQAModel` | Abstract QA interface |
| `GPT3QAModel` | GPT-3 QA model |
| `GPT3TurboQAModel` | GPT-3.5-turbo QA model |
| `GPT4QAModel` | GPT-4 QA model |
| `UnifiedQAModel` | T5-based local QA model |

---

## Entry Points

### Primary Entry Point: RetrievalAugmentation

The main way users interact with RAPTOR:

```python
from raptor import RetrievalAugmentation

# Simple initialization (uses all defaults)
RA = RetrievalAugmentation()

# With custom config
from raptor import RetrievalAugmentationConfig
config = RetrievalAugmentationConfig(...)
RA = RetrievalAugmentation(config=config)

# Load existing tree
RA = RetrievalAugmentation(tree="path/to/saved/tree")
```

### Secondary Entry Point: Direct Component Usage

For advanced users who want fine-grained control:

```python
from raptor import (
    ClusterTreeBuilder, ClusterTreeConfig,
    TreeRetriever, TreeRetrieverConfig
)

# Build tree directly
config = ClusterTreeConfig(...)
builder = ClusterTreeBuilder(config)
tree = builder.build_from_text(text)

# Create retriever
retriever_config = TreeRetrieverConfig(...)
retriever = TreeRetriever(retriever_config, tree)
context = retriever.retrieve(query)
```

---

## Class Documentation

### RetrievalAugmentation

**Location:** `RetrievalAugmentation.py:153-307`

**Purpose:** Main facade that orchestrates tree building, retrieval, and question answering.

#### Constructor

```python
def __init__(self, config=None, tree=None):
    """
    Args:
        config (RetrievalAugmentationConfig, optional): Configuration object.
            Defaults to RetrievalAugmentationConfig() with all defaults.
        tree (str | Tree | None, optional): Either:
            - Path to a pickled Tree file
            - Tree instance
            - None (tree will be built later)
    """
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_documents` | `(docs: str) -> None` | Build tree from text documents |
| `retrieve` | `(question, ...) -> str \| tuple` | Get relevant context for a question |
| `answer_question` | `(question, ...) -> str \| tuple` | Retrieve context and answer question |
| `save` | `(path: str) -> None` | Pickle tree to file |

#### add_documents()

```python
def add_documents(self, docs: str) -> None:
    """
    Builds a tree from the input text documents.
    
    Args:
        docs (str): The input text to index.
    
    Note:
        - If tree already exists, prompts for confirmation
        - Creates TreeRetriever after building
    """
```

#### retrieve()

```python
def retrieve(
    self,
    question: str,
    start_layer: int = None,      # Default: top layer
    num_layers: int = None,       # Default: all layers
    top_k: int = 10,              # Results per layer
    max_tokens: int = 3500,       # Max context tokens
    collapse_tree: bool = True,   # Use flat search
    return_layer_information: bool = True
) -> str | tuple[str, list]:
    """
    Retrieves relevant context for a question.
    
    Returns:
        If return_layer_information=False: str (context)
        If return_layer_information=True: tuple(context, layer_info)
        
        layer_info is list of dicts: [{"node_index": int, "layer_number": int}, ...]
    """
```

#### answer_question()

```python
def answer_question(
    self,
    question: str,
    top_k: int = 10,
    start_layer: int = None,
    num_layers: int = None,
    max_tokens: int = 3500,
    collapse_tree: bool = True,
    return_layer_information: bool = False
) -> str | tuple[str, list]:
    """
    Retrieves context and answers the question using the QA model.
    
    Returns:
        If return_layer_information=False: str (answer)
        If return_layer_information=True: tuple(answer, layer_info)
    """
```

---

### RetrievalAugmentationConfig

**Location:** `RetrievalAugmentation.py:18-150`

**Purpose:** Master configuration object that configures all components.

#### Constructor Parameters

```python
def __init__(
    self,
    # Pre-built configs (override individual params if provided)
    tree_builder_config=None,      # ClusterTreeConfig instance
    tree_retriever_config=None,    # TreeRetrieverConfig instance
    
    # Model instances
    qa_model=None,                 # BaseQAModel (default: GPT3TurboQAModel)
    embedding_model=None,          # Shared BaseEmbeddingModel
    summarization_model=None,      # BaseSummarizationModel
    
    # Builder type
    tree_builder_type="cluster",   # Only "cluster" supported
    
    # TreeRetriever shortcuts (tr_*)
    tr_tokenizer=None,
    tr_threshold=0.5,
    tr_top_k=5,
    tr_selection_mode="top_k",
    tr_context_embedding_model="OpenAI",
    tr_embedding_model=None,
    tr_num_layers=None,
    tr_start_layer=None,
    
    # TreeBuilder shortcuts (tb_*)
    tb_tokenizer=None,
    tb_max_tokens=100,
    tb_num_layers=5,
    tb_threshold=0.5,
    tb_top_k=5,
    tb_selection_mode="top_k",
    tb_summarization_length=100,
    tb_summarization_model=None,
    tb_embedding_models=None,
    tb_cluster_embedding_model="OpenAI",
):
```

#### Configuration Hierarchy

```
RetrievalAugmentationConfig
├── tree_builder_config (ClusterTreeConfig)
│   ├── tokenizer
│   ├── max_tokens
│   ├── num_layers
│   ├── threshold
│   ├── top_k
│   ├── selection_mode
│   ├── summarization_length
│   ├── summarization_model
│   ├── embedding_models (dict)
│   ├── cluster_embedding_model
│   ├── reduction_dimension      # Cluster-specific
│   ├── clustering_algorithm     # Cluster-specific
│   └── clustering_params        # Cluster-specific
├── tree_retriever_config (TreeRetrieverConfig)
│   ├── tokenizer
│   ├── threshold
│   ├── top_k
│   ├── selection_mode
│   ├── context_embedding_model
│   ├── embedding_model
│   ├── num_layers
│   └── start_layer
└── qa_model
```

---

### Node

**Location:** `tree_structures.py:4-13`

**Purpose:** Represents a single node in the hierarchical tree.

```python
class Node:
    def __init__(
        self, 
        text: str,              # Text content (original or summary)
        index: int,             # Unique node identifier
        children: Set[int],     # Indices of child nodes (empty for leaves)
        embeddings: dict        # {"model_name": embedding_vector, ...}
    ) -> None:
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The text content of the node |
| `index` | `int` | Unique identifier for this node |
| `children` | `Set[int]` | Set of child node indices (empty for leaf nodes) |
| `embeddings` | `dict` | Dictionary mapping model names to embedding vectors |

---

### Tree

**Location:** `tree_structures.py:16-28`

**Purpose:** Represents the complete hierarchical tree structure.

```python
class Tree:
    def __init__(
        self,
        all_nodes: Dict[int, Node],      # All nodes by index
        root_nodes: Dict[int, Node],     # Top-level summary nodes
        leaf_nodes: Dict[int, Node],     # Bottom-level original chunks
        num_layers: int,                 # Total number of layers
        layer_to_nodes: Dict[int, List[Node]]  # Nodes by layer
    ) -> None:
```

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `all_nodes` | `Dict[int, Node]` | Complete mapping of all nodes |
| `root_nodes` | `Dict[int, Node]` | Highest-level summary nodes |
| `leaf_nodes` | `Dict[int, Node]` | Original text chunk nodes |
| `num_layers` | `int` | Number of layers (0 = leaves) |
| `layer_to_nodes` | `Dict[int, List[Node]]` | Nodes organized by layer |

---

### BaseEmbeddingModel

**Location:** `EmbeddingModels.py:11-14`

**Purpose:** Abstract base class for embedding models.

```python
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for input text.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        pass
```

#### Implementations

**OpenAIEmbeddingModel** (`EmbeddingModels.py:17-29`)
```python
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model
```

**SBertEmbeddingModel** (`EmbeddingModels.py:32-37`)
```python
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)
```

---

### BaseSummarizationModel

**Location:** `SummarizationModels.py:11-14`

**Purpose:** Abstract base class for summarization models.

```python
class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """
        Generate summary of input context.
        
        Args:
            context (str): Text to summarize
            max_tokens (int): Maximum tokens in summary
            
        Returns:
            str: Summary text
        """
        pass
```

#### Implementations

**GPT3TurboSummarizationModel** (`SummarizationModels.py:17-44`)
```python
class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
```

**GPT3SummarizationModel** (`SummarizationModels.py:47-74`)
```python
class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model
```

---

### BaseQAModel

**Location:** `QAModels.py:15-18`

**Purpose:** Abstract base class for question-answering models.

```python
class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer a question given context.
        
        Args:
            context (str): Retrieved context
            question (str): User question
            
        Returns:
            str: Answer text
        """
        pass
```

#### Implementations

| Class | Model | Type |
|-------|-------|------|
| `GPT3QAModel` | text-davinci-003 | OpenAI Completion |
| `GPT3TurboQAModel` | gpt-3.5-turbo | OpenAI Chat |
| `GPT4QAModel` | gpt-4 | OpenAI Chat |
| `UnifiedQAModel` | allenai/unifiedqa-v2-t5-3b | Local T5 |

---

### BaseRetriever

**Location:** `Retrievers.py:5-8`

**Purpose:** Abstract base class for retrievers.

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query (str): Search query
            
        Returns:
            str: Retrieved context
        """
        pass
```

---

## Usage Examples

### Basic Usage (from demo.ipynb)

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"

from raptor import RetrievalAugmentation

# Initialize
RA = RetrievalAugmentation()

# Load document
with open('sample.txt', 'r') as file:
    text = file.read()

# Build tree
RA.add_documents(text)

# Ask question
question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer:", answer)

# Save tree
RA.save("path/to/tree")

# Load tree later
RA = RetrievalAugmentation(tree="path/to/tree")
```

### Custom Models (from demo.ipynb)

```python
from raptor import (
    BaseSummarizationModel, 
    BaseQAModel, 
    BaseEmbeddingModel,
    RetrievalAugmentationConfig,
    RetrievalAugmentation
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import torch

# Custom Summarization Model
class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def summarize(self, context, max_tokens=150):
        messages = [
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        return outputs[0]["generated_text"].strip()

# Custom QA Model
class GEMMAQAModel(BaseQAModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def answer_question(self, context, question):
        messages = [
            {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        return outputs[0]["generated_text"][len(prompt):]

# Custom Embedding Model
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

# Create config with custom models
config = RetrievalAugmentationConfig(
    summarization_model=GEMMASummarizationModel(),
    qa_model=GEMMAQAModel(),
    embedding_model=SBertEmbeddingModel()
)

# Initialize with custom config
RA = RetrievalAugmentation(config=config)

# Use as normal
RA.add_documents(text)
answer = RA.answer_question(question="Your question here")
```

### Advanced Configuration

```python
from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    OpenAIEmbeddingModel,
    GPT4QAModel,
    GPT3TurboSummarizationModel
)

config = RetrievalAugmentationConfig(
    # Models
    qa_model=GPT4QAModel(),
    summarization_model=GPT3TurboSummarizationModel(),
    
    # Tree Builder settings
    tb_max_tokens=200,           # Larger chunks
    tb_num_layers=3,             # Fewer layers
    tb_summarization_length=150, # Longer summaries
    
    # Tree Retriever settings
    tr_top_k=10,                 # More results per layer
    tr_selection_mode="threshold",
    tr_threshold=0.3,            # Lower threshold = more results
)

RA = RetrievalAugmentation(config=config)
```

### Retrieval Strategies

```python
# Strategy 1: Collapsed Tree (flat search across all nodes)
context = RA.retrieve(
    question="Your question",
    collapse_tree=True,  # Default
    top_k=10,
    max_tokens=3500
)

# Strategy 2: Tree Traversal (hierarchical search)
context = RA.retrieve(
    question="Your question",
    collapse_tree=False,
    start_layer=2,       # Start from layer 2
    num_layers=3,        # Traverse 3 layers down
    top_k=5              # Select top-5 at each layer
)

# Get layer information
context, layer_info = RA.retrieve(
    question="Your question",
    return_layer_information=True
)
# layer_info = [{"node_index": 42, "layer_number": 1}, ...]
```

### Direct Tree Building

```python
from raptor import ClusterTreeBuilder, ClusterTreeConfig

config = ClusterTreeConfig(
    max_tokens=100,
    num_layers=5,
    reduction_dimension=10,
    threshold=0.1,
)

builder = ClusterTreeBuilder(config)
tree = builder.build_from_text(text)

# Access tree structure
print(f"Total nodes: {len(tree.all_nodes)}")
print(f"Leaf nodes: {len(tree.leaf_nodes)}")
print(f"Root nodes: {len(tree.root_nodes)}")
print(f"Layers: {tree.num_layers}")

# Inspect a node
node = tree.all_nodes[0]
print(f"Text: {node.text[:100]}...")
print(f"Children: {node.children}")
print(f"Embedding models: {list(node.embeddings.keys())}")
```

### Using FaissRetriever (Non-Hierarchical)

```python
from raptor import FaissRetriever, FaissRetrieverConfig

config = FaissRetrieverConfig(
    max_tokens=100,
    max_context_tokens=3500,
    top_k=10,
)

retriever = FaissRetriever(config)
retriever.build_from_text(document_text)

# Or build from existing leaf nodes
# retriever.build_from_leaf_nodes(tree.leaf_nodes.values())

context = retriever.retrieve("Your query here")
```

---

## Configuration Reference

### Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_tokens` | 100 | 1+ | Tokens per text chunk |
| `num_layers` | 5 | 1+ | Number of tree layers |
| `top_k` | 5 | 1+ | Nodes to select |
| `threshold` | 0.5 | 0-1 | Similarity threshold |
| `selection_mode` | "top_k" | "top_k", "threshold" | Selection strategy |
| `summarization_length` | 100 | 1+ | Max summary tokens |
| `reduction_dimension` | 10 | 1+ | UMAP target dims |
| `max_context_tokens` | 3500 | 1+ | Total context limit |

### Selection Modes

**top_k:** Select the k most similar nodes regardless of similarity score.

**threshold:** Select all nodes with similarity above threshold.

### Retrieval Strategies

**collapse_tree=True (Collapsed):**
- Search all nodes in the tree
- Faster for small trees
- May miss hierarchical context

**collapse_tree=False (Tree Traversal):**
- Start at specified layer (default: top)
- At each layer, select top-k nodes
- Descend to children of selected nodes
- Better for understanding document structure

---

## Error Handling

### Common Exceptions

```python
# Missing API key
# ValueError: OPENAI_API_KEY environment variable not set

# Invalid configuration
ValueError("max_tokens must be an integer and at least 1")
ValueError("threshold must be a float between 0 and 1")
ValueError("selection_mode must be either 'top_k' or 'threshold'")

# Missing tree
ValueError("The TreeRetriever instance has not been initialized. Call 'add_documents' first.")

# Invalid tree file
ValueError(f"Failed to load tree from {tree}: {e}")
ValueError("The loaded object is not an instance of Tree")

# Layer configuration
ValueError("num_layers must be less than or equal to start_layer + 1")
ValueError("start_layer must be less than or equal to tree.num_layers")
```

### Retry Behavior

All OpenAI API calls use tenacity retry logic:
- Wait: Exponential backoff (1-20 seconds)
- Stop: After 6 attempts
- Handles rate limits and transient failures
