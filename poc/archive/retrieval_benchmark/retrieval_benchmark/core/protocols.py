"""Protocol definitions (abstract interfaces) for the retrieval benchmark."""

from abc import ABC, abstractmethod
from typing import Optional

from .types import Document, SearchResponse, IndexStats, SearchHit


class Embedder(ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        ...


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name/identifier."""
        ...
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection for vectors."""
        ...
    
    @abstractmethod
    def insert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert vectors into a collection."""
        ...
    
    @abstractmethod
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filter: Optional[dict] = None,
    ) -> list[SearchHit]:
        """Search for similar vectors."""
        ...
    
    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of vectors in a collection."""
        ...
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection and cleanup resources."""
        ...


class LLM(ABC):
    """Abstract base class for language models."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> str:
        """Generate text from a prompt."""
        ...


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...
    
    @property
    @abstractmethod
    def requires_llm(self) -> bool:
        """Return whether this strategy requires an LLM."""
        ...
    
    @abstractmethod
    def configure(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: Optional[LLM] = None,
    ) -> None:
        """Configure the strategy with required components."""
        ...
    
    @abstractmethod
    def index(self, documents: list[Document]) -> IndexStats:
        """Index a list of documents. Returns indexing statistics."""
        ...
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        """Search for relevant documents/chunks."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the index and free resources."""
        ...
