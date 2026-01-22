# RAPTOR Extension Points

> Complete guide to extending and customizing RAPTOR components

## Overview

RAPTOR is designed with extensibility in mind through abstract base classes. The following components can be customized:

| Extension Point | Base Class | Purpose |
|-----------------|------------|---------|
| Embedding Models | `BaseEmbeddingModel` | Convert text to vector representations |
| Summarization Models | `BaseSummarizationModel` | Generate summaries of text clusters |
| QA Models | `BaseQAModel` | Answer questions given context |
| Retrievers | `BaseRetriever` | Retrieve relevant context for queries |
| Clustering Algorithms | `ClusteringAlgorithm` | Group similar nodes together |

---

## 1. BaseEmbeddingModel

**Location:** `raptor/EmbeddingModels.py:11-14`

### Interface Definition

```python
from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        """
        Convert text to a vector embedding.
        
        Args:
            text: The input text string to embed
            
        Returns:
            List[float] or np.ndarray: The embedding vector
        """
        pass
```

### Required Methods

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `create_embedding` | `(self, text: str)` | `List[float]` or `np.ndarray` | Generate embedding for input text |

### Built-in Implementations

#### OpenAIEmbeddingModel

```python
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
```

**Features:**
- Uses OpenAI's embedding API
- Default model: `text-embedding-ada-002`
- Automatic retry with exponential backoff (6 attempts)
- Replaces newlines with spaces before embedding

#### SBertEmbeddingModel

```python
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
```

**Features:**
- Uses Sentence-Transformers library
- Default model: `multi-qa-mpnet-base-cos-v1`
- No retry logic (local model)

### Custom Implementation Example

```python
from raptor import BaseEmbeddingModel
import cohere

class CohereEmbeddingModel(BaseEmbeddingModel):
    """Custom embedding model using Cohere's API."""
    
    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
    
    def create_embedding(self, text: str):
        """Generate embedding using Cohere API."""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings[0]


# Usage with RAPTOR
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

cohere_embeddings = CohereEmbeddingModel(api_key="your-api-key")
config = RetrievalAugmentationConfig(embedding_model=cohere_embeddings)
RA = RetrievalAugmentation(config=config)
```

### Embedding Model with Local Transformers

```python
from raptor import BaseEmbeddingModel
from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """Custom embedding model using any HuggingFace model."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def create_embedding(self, text: str):
        """Generate embedding using mean pooling."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy().tolist()
```

---

## 2. BaseSummarizationModel

**Location:** `raptor/SummarizationModels.py:11-14`

### Interface Definition

```python
from abc import ABC, abstractmethod

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        """
        Generate a summary of the input context.
        
        Args:
            context: The text to summarize
            max_tokens: Maximum tokens in the summary (default: 150)
            
        Returns:
            str: The generated summary
        """
        pass
```

### Required Methods

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `summarize` | `(self, context: str, max_tokens: int = 150)` | `str` | Generate summary of input text |

### Built-in Implementations

#### GPT3TurboSummarizationModel

```python
class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return e  # Returns exception object on failure
```

**Features:**
- Uses OpenAI Chat Completions API
- System prompt: "You are a helpful assistant."
- User prompt: "Write a summary of the following, including as many key details as possible: {context}:"
- Retry logic with exponential backoff (6 attempts)
- **Warning:** Returns exception object on failure, not string

#### GPT3SummarizationModel

```python
class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model
    # Same implementation as GPT3TurboSummarizationModel
```

**Note:** Despite the class name suggesting completions API, it actually uses chat completions (see source code).

### Custom Implementation Example

```python
from raptor import BaseSummarizationModel
from anthropic import Anthropic

class ClaudeSummarizationModel(BaseSummarizationModel):
    """Custom summarization model using Anthropic Claude."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def summarize(self, context: str, max_tokens: int = 500):
        """Generate summary using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": f"Write a concise summary of the following text, including key details:\n\n{context}"
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Summarization error: {e}")
            return f"Error: {str(e)}"


# Usage
claude_summarizer = ClaudeSummarizationModel(api_key="your-api-key")
config = RetrievalAugmentationConfig(summarization_model=claude_summarizer)
RA = RetrievalAugmentation(config=config)
```

### Local LLM Summarization (Ollama)

```python
from raptor import BaseSummarizationModel
import requests

class OllamaSummarizationModel(BaseSummarizationModel):
    """Custom summarization using local Ollama instance."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def summarize(self, context: str, max_tokens: int = 500):
        """Generate summary using Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"Write a summary of the following, including as many key details as possible:\n\n{context}",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }
            )
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama error: {e}")
            return str(e)
```

---

## 3. BaseQAModel

**Location:** `raptor/QAModels.py:15-18`

### Interface Definition

```python
from abc import ABC, abstractmethod

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        """
        Answer a question given the context.
        
        Args:
            context: The context containing relevant information
            question: The question to answer
            
        Returns:
            str: The answer to the question
        """
        pass
```

### Required Methods

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `answer_question` | `(self, context: str, question: str)` | `str` | Answer question using context |

### Built-in Implementations

#### GPT3TurboQAModel (Default)

```python
class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        try:
            return self._attempt_answer_question(context, question, max_tokens, stop_sequence)
        except Exception as e:
            print(e)
            return e  # Returns exception object on failure
```

**Features:**
- System prompt: "You are Question Answering Portal"
- User prompt: "Given Context: {context} Give the best full answer amongst the option to question {question}"
- Temperature: 0 (deterministic)
- Double retry decorator (both methods)

#### GPT4QAModel

Same implementation as GPT3TurboQAModel but with `model="gpt-4"` default.

#### GPT3QAModel (Legacy Completions API)

```python
class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(e)
            return ""  # Returns empty string on failure
```

**Note:** This uses the legacy completions API and has a typo ("folloing" instead of "following").

#### UnifiedQAModel (Local T5)

```python
class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]
```

**Features:**
- Fully local, no API calls
- Uses Allen AI's UnifiedQA T5 model
- GPU support with automatic fallback to CPU

### Custom Implementation Example

```python
from raptor import BaseQAModel
from anthropic import Anthropic

class ClaudeQAModel(BaseQAModel):
    """Custom QA model using Anthropic Claude."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def answer_question(self, context: str, question: str):
        """Answer question using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
                    }
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"QA error: {e}")
            return f"Error: {str(e)}"


# Usage
claude_qa = ClaudeQAModel(api_key="your-api-key")
config = RetrievalAugmentationConfig(qa_model=claude_qa)
RA = RetrievalAugmentation(config=config)
```

---

## 4. BaseRetriever

**Location:** `raptor/Retrievers.py:5-8`

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The search query
            
        Returns:
            str: The retrieved context
        """
        pass
```

### Required Methods

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `retrieve` | `(self, query: str)` | `str` | Retrieve context for query |

### Built-in Implementations

#### TreeRetriever

The main retriever implementation that traverses the RAPTOR tree structure. Supports two modes:
- **Collapsed Tree:** Searches all nodes at once
- **Tree Traversal:** Traverses from top to bottom

See `tree_retriever.py` for full implementation.

#### FaissRetriever

A simpler flat retriever using FAISS for similarity search:

```python
class FaissRetriever(BaseRetriever):
    def __init__(self, config):
        self.embedding_model = config.embedding_model
        self.question_embedding_model = config.question_embedding_model
        self.index = None
        self.context_chunks = None
        # ... additional initialization
    
    def build_from_text(self, doc_text):
        """Build FAISS index from text."""
        self.context_chunks = np.array(split_text(doc_text, self.tokenizer, self.max_tokens))
        # Generate embeddings with parallel processing
        # Build FAISS index
    
    def retrieve(self, query: str) -> str:
        """Retrieve similar chunks using FAISS."""
        query_embedding = self.question_embedding_model.create_embedding(query)
        # Search FAISS index
        # Return concatenated context
```

### Custom Implementation Example

```python
from raptor import BaseRetriever
from chromadb import Client

class ChromaDBRetriever(BaseRetriever):
    """Custom retriever using ChromaDB."""
    
    def __init__(self, collection_name: str = "raptor"):
        self.client = Client()
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def add_documents(self, texts: list, embeddings: list, ids: list):
        """Add documents to ChromaDB."""
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )
    
    def retrieve(self, query: str) -> str:
        """Retrieve relevant documents from ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )
        return "\n\n".join(results['documents'][0])
```

---

## 5. ClusteringAlgorithm

**Location:** `raptor/cluster_utils.py:126-129`

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        """
        Perform clustering on embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            **kwargs: Additional clustering parameters
            
        Returns:
            List[List[int]]: Cluster assignments for each embedding
        """
        pass
```

### Required Methods

| Method | Signature | Return Type | Description |
|--------|-----------|-------------|-------------|
| `perform_clustering` | `(embeddings, **kwargs)` | `List[List[int]]` | Cluster embeddings into groups |

### Built-in Implementation: RAPTOR_Clustering

```python
class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        RAPTOR's hierarchical clustering algorithm.
        
        1. Extract embeddings from nodes
        2. Global clustering with UMAP + GMM
        3. Local clustering within each global cluster
        4. Recursive re-clustering if clusters exceed max_length
        """
        # Get embeddings
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        
        # Perform clustering
        clusters = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)
        
        # Process clusters with token limit enforcement
        node_clusters = []
        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [nodes[i] for i in indices]
            
            # Base case
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue
            
            # Check token limit
            total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])
            
            if total_length > max_length_in_cluster:
                # Recursive re-clustering
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)
        
        return node_clusters
```

**Key Algorithm Steps:**

1. **UMAP Dimensionality Reduction** (global_cluster_embeddings):
   - Reduces embedding dimensions using UMAP
   - `n_neighbors = sqrt(len(embeddings) - 1)`
   - Metric: cosine similarity

2. **Gaussian Mixture Model Clustering** (GMM_cluster):
   - Determines optimal cluster count via BIC
   - Soft assignment with probability threshold

3. **Two-level Clustering:**
   - Global clusters on full dataset
   - Local clusters within each global cluster

4. **Token Limit Enforcement:**
   - Recursive re-clustering if cluster exceeds `max_length_in_cluster`

### Custom Implementation Example

```python
from raptor.cluster_utils import ClusteringAlgorithm
from sklearn.cluster import KMeans
import numpy as np
from typing import List

class KMeansClusteringAlgorithm(ClusteringAlgorithm):
    """Simple K-Means clustering for RAPTOR."""
    
    def perform_clustering(
        self,
        nodes,
        embedding_model_name: str,
        n_clusters: int = 10,
        max_length_in_cluster: int = 3500,
        tokenizer=None,
        **kwargs
    ) -> List[List]:
        """Cluster nodes using K-Means."""
        import tiktoken
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Extract embeddings
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        
        # Adjust n_clusters if needed
        n_clusters = min(n_clusters, len(nodes))
        
        # Perform K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(nodes[idx])
        
        return list(clusters.values())


# Usage with ClusterTreeConfig
from raptor import ClusterTreeConfig, ClusterTreeBuilder

custom_clustering = KMeansClusteringAlgorithm()
config = ClusterTreeConfig(
    clustering_algorithm=custom_clustering,
    clustering_params={"n_clusters": 15}
)
builder = ClusterTreeBuilder(config)
```

### Hierarchical Agglomerative Clustering Example

```python
from raptor.cluster_utils import ClusteringAlgorithm
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

class HierarchicalClusteringAlgorithm(ClusteringAlgorithm):
    """Hierarchical Agglomerative Clustering for RAPTOR."""
    
    def perform_clustering(
        self,
        nodes,
        embedding_model_name: str,
        distance_threshold: float = 0.5,
        linkage_method: str = "ward",
        **kwargs
    ):
        """Cluster using hierarchical agglomerative clustering."""
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        
        if len(nodes) <= 1:
            return [[node] for node in nodes]
        
        # Compute linkage matrix
        Z = linkage(embeddings, method=linkage_method)
        
        # Form flat clusters
        labels = fcluster(Z, t=distance_threshold, criterion='distance')
        
        # Group nodes by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(nodes[idx])
        
        return list(clusters.values())
```

---

## Complete Integration Example

Here's a full example showing all custom components together:

```python
from raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig,
    BaseEmbeddingModel,
    BaseSummarizationModel,
    BaseQAModel
)

# Custom Embedding Model
class MyEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        # Your implementation
        pass

# Custom Summarization Model
class MySummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        # Your implementation
        pass

# Custom QA Model
class MyQAModel(BaseQAModel):
    def answer_question(self, context, question):
        # Your implementation
        pass

# Create instances
my_embedding = MyEmbeddingModel()
my_summarizer = MySummarizationModel()
my_qa = MyQAModel()

# Configure RAPTOR with all custom models
config = RetrievalAugmentationConfig(
    embedding_model=my_embedding,
    summarization_model=my_summarizer,
    qa_model=my_qa,
    # Tree builder settings
    tb_max_tokens=100,
    tb_num_layers=5,
    tb_summarization_length=100,
    # Tree retriever settings
    tr_top_k=5,
    tr_selection_mode="top_k",
)

# Initialize and use
RA = RetrievalAugmentation(config=config)
RA.add_documents("Your document text here...")
answer = RA.answer_question("What is the main topic?")
```

---

## Notes on Method Signatures

### Extending Signature Compatibility

The base classes define minimal interfaces. Implementations can extend the signature with additional optional parameters:

```python
# Base definition
def summarize(self, context, max_tokens=150):
    pass

# Implementation can add parameters (with defaults)
def summarize(self, context, max_tokens=500, stop_sequence=None):
    # Extended implementation
```

This is safe because:
1. Additional parameters have defaults
2. RAPTOR calls methods with the base signature
3. Extra parameters are ignored if not provided

### Static vs Instance Methods

**Important:** `RAPTOR_Clustering.perform_clustering` is defined without `self` parameter, making it effectively a static method despite missing the `@staticmethod` decorator:

```python
class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(  # Note: no 'self' parameter
        nodes: List[Node],
        embedding_model_name: str,
        ...
    )
```

When implementing custom clustering, follow the same pattern or use `@staticmethod` explicitly.
