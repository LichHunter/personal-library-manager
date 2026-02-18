"""BM25 lexical retrieval index component.

Wraps the bm25s library to provide a persistent BM25 index for lexical
(keyword-based) retrieval. Supports indexing, saving, loading, and searching.

Example:
    >>> index = BM25Index()
    >>> index.index(['kubernetes pod definition', 'docker container'])
    >>> results = index.search('kubernetes pod', k=2)
    >>> results[0]['content']
    'kubernetes pod definition'
"""

from __future__ import annotations

import bm25s


class BM25Index:
    """BM25 index wrapping bm25s.BM25 for lexical retrieval.

    Supports building an index from a list of text chunks, persisting to disk,
    loading from disk, and searching with a query string.

    Tokenization uses simple lowercase whitespace splitting for POC parity.

    Attributes:
        _bm25: The underlying bm25s.BM25 instance (None until indexed).
        _corpus: The list of chunk strings used to build the index.
    """

    def __init__(self) -> None:
        """Initialize an empty BM25Index."""
        self._bm25: bm25s.BM25 | None = None
        self._corpus: list[str] = []

    def index(self, chunks: list[str]) -> None:
        """Build the BM25 index from a list of text chunks.

        Args:
            chunks: List of text strings to index.

        Example:
            >>> idx = BM25Index()
            >>> idx.index(['kubernetes pod', 'docker container'])
        """
        self._corpus = list(chunks)
        tokenized = [chunk.lower().split() for chunk in self._corpus]
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokenized, show_progress=False)

    def save(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Directory path where the index files will be saved.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError("Index has not been built. Call index() first.")
        self._bm25.save(path, corpus=self._corpus)

    @classmethod
    def load(cls, path: str, mmap: bool = False) -> "BM25Index":
        """Load a previously saved index from disk.

        Args:
            path: Directory path where the index was saved.
            mmap: If True, use memory-mapped arrays (reduces RAM usage).

        Returns:
            A BM25Index instance with the loaded index and corpus.
        """
        instance = cls()
        instance._bm25 = bm25s.BM25.load(path, load_corpus=True, mmap=mmap)
        raw_corpus = instance._bm25.corpus
        if raw_corpus is not None:
            instance._corpus = [
                item["text"] if isinstance(item, dict) else str(item)
                for item in raw_corpus
            ]
        # Clear internal corpus so retrieve() returns raw int indices
        # instead of corpus dicts. We store corpus separately in _corpus.
        instance._bm25.corpus = None
        return instance

    def search(self, query: str, k: int) -> list[dict]:
        """Search the index for the top-k results matching the query.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of dicts with keys:
                - 'content' (str): The chunk text.
                - 'score' (float): BM25 relevance score.
                - 'index' (int): Original position in the indexed corpus.

        Raises:
            RuntimeError: If the index has not been built yet.

        Example:
            >>> idx = BM25Index()
            >>> idx.index(['kubernetes pod', 'docker container'])
            >>> results = idx.search('kubernetes', k=1)
            >>> results[0]['content']
            'kubernetes pod'
        """
        if self._bm25 is None:
            raise RuntimeError("Index has not been built. Call index() or load() first.")

        query_tokens = [query.lower().split()]
        actual_k = min(k, len(self._corpus))

        # Retrieve raw integer indices (NOT corpus strings) to avoid
        # list.index() collisions when duplicate content exists.
        indices, scores = self._bm25.retrieve(
            query_tokens,
            k=actual_k,
            show_progress=False,
        )

        # indices shape: (n_queries, k), scores shape: (n_queries, k)
        output = []
        for i in range(actual_k):
            idx = int(indices[0, i])
            score = float(scores[0, i])
            content = self._corpus[idx] if idx < len(self._corpus) else ""
            output.append({
                "content": content,
                "score": score,
                "index": idx,
            })

        return output
