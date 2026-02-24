"""BM25 implementation of SparseRetriever interface.

Wraps the existing BM25Index to conform to the SparseRetriever interface,
enabling seamless switching between BM25 and SPLADE in HybridRetriever.
"""

from __future__ import annotations

from pathlib import Path

from plm.search.components.bm25 import BM25Index
from plm.search.components.sparse.base import SparseRetriever


class BM25Retriever(SparseRetriever):
    """BM25 sparse retriever using bm25s library.

    This is a thin wrapper around BM25Index that implements the
    SparseRetriever interface for use with the retriever factory.
    """

    def __init__(self) -> None:
        self._index: BM25Index | None = None

    def index(self, documents: list[str]) -> None:
        self._index = BM25Index()
        self._index.index(documents)

    def search(self, query: str, k: int) -> list[dict]:
        if self._index is None:
            raise RuntimeError("Index has not been built. Call index() or load() first.")
        return self._index.search(query, k)

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built. Call index() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        instance = cls()
        instance._index = BM25Index.load(str(path))
        return instance

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    @property
    def document_count(self) -> int:
        if self._index is None or not hasattr(self._index, "_corpus"):
            return 0
        return len(self._index._corpus)
