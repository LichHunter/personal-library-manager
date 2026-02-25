"""Cross-encoder reranking component for modular retrieval pipeline.

This module implements a reranker component that uses cross-encoder models to
refine initial retrieval results. Cross-encoders score query-chunk pairs directly,
providing more accurate relevance scores than semantic embeddings alone.

The reranker is typically the final step in a retrieval pipeline, reranking the
top-k results from earlier stages (BM25, semantic, or fused) for maximum accuracy.

Example:
    >>> reranker = Reranker(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
    >>> data = {
    ...     'query': 'kubernetes pod',
    ...     'chunks': [
    ...         ScoredChunk('0', 'kubernetes pod definition', 0.8, 'rrf', 1, ()),
    ...         ScoredChunk('1', 'docker container', 0.7, 'rrf', 2, ())
    ...     ]
    ... }
    >>> results = reranker.process(data)
    >>> # results = [
    >>> #     ScoredChunk('0', 'kubernetes pod definition', 0.95, 'rrf', 1, ()),
    >>> #     ScoredChunk('1', 'docker container', 0.3, 'rrf', 2, ())
    >>> # ]
"""

from typing import Any
import numpy as np

from ..base import Component
from ..types import ScoredChunk
from ..utils.logger import get_logger


class Reranker(Component):
    """Cross-encoder reranking component.

    Wraps a cross-encoder model to rerank chunks based on their relevance to
    the query. Cross-encoders score query-chunk pairs directly, providing more
    accurate relevance scores than embedding-based approaches.

    The model is loaded lazily in process() (not in __init__) to maintain
    statelessness and avoid loading expensive models unnecessarily.

    Attributes:
        model_name (str): Name of the cross-encoder model to use.
            Default: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        _model: Lazy-loaded cross-encoder model (None until first process() call)

    Example:
        >>> reranker = Reranker()
        >>> data = {
        ...     'query': 'kubernetes',
        ...     'chunks': [
        ...         ScoredChunk('0', 'kubernetes pod', 0.8, 'semantic', 1, ()),
        ...         ScoredChunk('1', 'docker container', 0.7, 'semantic', 2, ())
        ...     ]
        ... }
        >>> results = reranker.process(data)
        >>> len(results)
        2
        >>> results[0].score > results[1].score
        True
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize Reranker component.

        Args:
            model: Name of the cross-encoder model to use.
                Default: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                Other options: 'cross-encoder/qnli-distilroberta-base',
                              'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        """
        self.model_name = model
        self._model = None
        self._log = get_logger()
        self._log.debug(f"[Reranker] Initialized with model: {model}")

    def _load_model(self) -> Any:
        """Load cross-encoder model lazily.

        Loads the model on first use to avoid unnecessary initialization.
        Subsequent calls return the cached model.

        Returns:
            Loaded cross-encoder model

        Raises:
            ImportError: If sentence-transformers is not installed
            OSError: If model cannot be downloaded
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for Reranker. "
                "Install with: pip install sentence-transformers"
            )

        self._model = CrossEncoder(self.model_name)
        return self._model

    def process(self, data: dict[str, Any]) -> list[ScoredChunk]:
        """Rerank chunks using cross-encoder model.

        Takes a query and list of ScoredChunk objects, scores each query-chunk
        pair using the cross-encoder, and returns the chunks reranked by the
        new scores.

        The cross-encoder scores are normalized to [0, 1] range for consistency
        with other scoring methods.

        Args:
            data: Dictionary with required fields:
                - 'query' (str): The search query
                - 'chunks' (list[ScoredChunk]): List of ScoredChunk objects to rerank

        Returns:
            List of ScoredChunk objects reranked by cross-encoder scores.
            Each chunk has:
            - chunk_id: Original chunk ID (preserved)
            - content: Original chunk content (preserved)
            - score: New cross-encoder relevance score (0.0 to 1.0)
            - source: Original source (preserved)
            - rank: New rank based on cross-encoder scores (1-based)
            - metadata: Original metadata (preserved)

        Raises:
            KeyError: If 'query' or 'chunks' field is missing
            ValueError: If chunks list is empty
            TypeError: If query is not a string or chunks are not ScoredChunk objects

        Example:
            >>> reranker = Reranker()
            >>> data = {
            ...     'query': 'kubernetes pod',
            ...     'chunks': [
            ...         ScoredChunk('0', 'pod definition', 0.8, 'semantic', 1, ()),
            ...         ScoredChunk('1', 'container info', 0.7, 'semantic', 2, ())
            ...     ]
            ... }
            >>> results = reranker.process(data)
            >>> results[0].score > results[1].score
            True
        """
        self._log.debug(f"[Reranker] Reranking chunks for query")

        # Validate input
        if "query" not in data:
            raise KeyError("Input dict must have 'query' field")
        if "chunks" not in data:
            raise KeyError("Input dict must have 'chunks' field")

        query = data["query"]
        chunks = data["chunks"]

        if not isinstance(query, str):
            raise TypeError(f"Query must be string, got {type(query).__name__}")

        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Validate chunks are ScoredChunk objects
        for chunk in chunks:
            if not isinstance(chunk, ScoredChunk):
                raise TypeError(
                    f"Chunks must be ScoredChunk objects, got {type(chunk).__name__}"
                )

        # Load model (lazy loading)
        model = self._load_model()

        # Create query-chunk pairs for cross-encoder
        pairs = [[query, chunk.content] for chunk in chunks]

        # Score pairs using cross-encoder
        scores = model.predict(pairs)
        self._log.trace(
            f"[Reranker] Computed cross-encoder scores for {len(scores)} chunks"
        )

        # Normalize scores to [0, 1] range
        # Cross-encoder outputs are typically in [-inf, inf], so we use sigmoid
        # to normalize to [0, 1]
        normalized_scores = 1 / (1 + np.exp(-scores))

        # Sort by score (descending)
        sorted_indices = np.argsort(normalized_scores)[::-1]

        # Create reranked ScoredChunk objects with new scores and ranks
        reranked_chunks = []
        for rank, idx in enumerate(sorted_indices, start=1):
            original_chunk = chunks[idx]
            reranked_chunks.append(
                ScoredChunk(
                    chunk_id=original_chunk.chunk_id,
                    content=original_chunk.content,
                    score=float(normalized_scores[idx]),
                    source=original_chunk.source,
                    rank=rank,
                    metadata=original_chunk.metadata,
                )
            )

        self._log.debug(
            f"[Reranker] Completed reranking: {len(reranked_chunks)} chunks reranked"
        )
        return reranked_chunks
