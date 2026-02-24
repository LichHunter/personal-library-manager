"""Abstract base class for sparse retrievers.

Defines the interface that all sparse retrieval implementations must follow,
enabling seamless switching between BM25 and SPLADE (or future methods).

The interface is designed to be minimal and consistent with how HybridRetriever
interacts with sparse search components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class SparseRetriever(ABC):
    """Abstract base class for sparse retrieval methods.

    Implementations must provide:
    - index(): Build the sparse index from text documents
    - search(): Retrieve top-k results for a query
    - save(): Persist the index to disk
    - load(): Load a persisted index from disk

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.index(['kubernetes pod', 'docker container'])
        >>> results = retriever.search('kubernetes', k=2)
        >>> retriever.save('/path/to/index')
        >>> loaded = BM25Retriever.load('/path/to/index')
    """

    @abstractmethod
    def index(self, documents: list[str]) -> None:
        """Build the sparse index from a list of text documents.

        Args:
            documents: List of text strings to index. Order is preserved
                and indices in search results correspond to document positions.
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[dict]:
        """Search the index for the top-k results matching the query.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of dicts with keys:
                - 'index' (int): Position in the indexed corpus
                - 'score' (float): Relevance score (higher is better)
                - 'content' (str): The original document text

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index to disk.

        Args:
            path: Directory path where the index files will be saved.
                Directory is created if it doesn't exist.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "SparseRetriever":
        """Load a previously saved index from disk.

        Args:
            path: Directory path where the index was saved.

        Returns:
            A SparseRetriever instance with the loaded index.

        Raises:
            FileNotFoundError: If the index directory doesn't exist.
        """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the index is built and ready for search.

        Returns:
            True if the index is ready for search queries.
        """
        ...

    @property
    @abstractmethod
    def document_count(self) -> int:
        """Get the number of indexed documents.

        Returns:
            Number of documents in the index, or 0 if not indexed.
        """
        ...
