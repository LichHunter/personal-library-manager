"""BM25 lexical retrieval scorer component.

This module implements BM25Okapi scoring for lexical (keyword-based) retrieval.
It wraps the rank_bm25 library to provide a stateless component that scores
chunks based on keyword overlap with the query.

BM25 is one half of the hybrid retrieval strategy (BM25 + semantic).
It excels at exact term matching and is fast, deterministic, and interpretable.

Example:
    >>> scorer = BM25Scorer()
    >>> data = {
    ...     'query': 'kubernetes pod',
    ...     'chunks': ['kubernetes pod definition', 'docker container', 'kubernetes deployment']
    ... }
    >>> results = scorer.process(data)
    >>> # results = [
    >>> #     ScoredChunk(chunk_id='0', content='kubernetes pod definition', score=1.5, source='bm25', rank=1),
    >>> #     ScoredChunk(chunk_id='2', content='kubernetes deployment', score=1.2, source='bm25', rank=2),
    >>> #     ScoredChunk(chunk_id='1', content='docker container', score=0.3, source='bm25', rank=3)
    >>> # ]
"""

from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi

from ..base import Component
from ..types import ScoredChunk
from ..utils.logger import get_logger


class BM25Scorer(Component):
    """BM25Okapi lexical retrieval scorer.

    Implements stateless BM25 scoring for chunks. Builds the BM25 index
    fresh in each process() call (not in __init__) to maintain statelessness.

    The BM25 algorithm scores chunks based on term frequency and inverse
    document frequency, providing fast and interpretable lexical retrieval.

    Attributes:
        None (stateless component)

    Example:
        >>> scorer = BM25Scorer()
        >>> data = {
        ...     'query': 'kubernetes',
        ...     'chunks': ['kubernetes pod', 'docker container', 'kubernetes deployment']
        ... }
        >>> results = scorer.process(data)
        >>> len(results)
        3
        >>> results[0].source
        'bm25'
        >>> results[0].rank
        1
    """

    def __init__(self):
        """Initialize BM25Scorer with logger."""
        self._log = get_logger()
        self._log.debug("[BM25Scorer] Initialized")

    def process(self, data: dict[str, Any]) -> list[ScoredChunk]:
        """Score chunks using BM25 algorithm.

        Builds a fresh BM25 index from the chunks and scores them against
        the query. Returns scored chunks sorted by relevance (highest first).

        Args:
            data: Dictionary with required fields:
                - 'query' (str): The search query
                - 'chunks' (list): List of chunk strings or dicts with 'content' field

        Returns:
            List of ScoredChunk objects sorted by score (descending).
            Each chunk has:
            - chunk_id: String index of chunk in original list
            - content: The chunk text
            - score: BM25 relevance score
            - source: "bm25" (lexical signal)
            - rank: Position in sorted results (1-based)

        Raises:
            KeyError: If 'query' or 'chunks' field is missing
            ValueError: If chunks list is empty
            TypeError: If query is not a string

        Example:
            >>> scorer = BM25Scorer()
            >>> data = {
            ...     'query': 'kubernetes',
            ...     'chunks': ['kubernetes pod', 'docker container', 'kubernetes deployment']
            ... }
            >>> results = scorer.process(data)
            >>> results[0].score > results[1].score
            True
        """
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

        self._log.debug(
            f"[BM25Scorer] Indexing {len(chunks)} chunks for query: {query}"
        )

        # Extract chunk content (handle both strings and dicts)
        chunk_contents = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                if "content" not in chunk:
                    raise KeyError("Chunk dict must have 'content' field")
                chunk_contents.append(chunk["content"])
            elif isinstance(chunk, str):
                chunk_contents.append(chunk)
            else:
                raise TypeError(
                    f"Chunk must be string or dict, got {type(chunk).__name__}"
                )

        # Tokenize chunks (lowercase, split on whitespace)
        tokenized_chunks = [content.lower().split() for content in chunk_contents]

        # Build BM25 index (fresh in each process() call for statelessness)
        bm25 = BM25Okapi(tokenized_chunks)

        # Score query against chunks
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        self._log.trace(f"[BM25Scorer] Computed BM25 scores for {len(scores)} chunks")

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Create ScoredChunk objects
        scored_chunks = []
        for rank, idx in enumerate(sorted_indices, start=1):
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=str(idx),
                    content=chunk_contents[idx],
                    score=float(scores[idx]),
                    source="bm25",
                    rank=rank,
                )
            )

        self._log.debug(
            f"[BM25Scorer] Completed scoring, returned {len(scored_chunks)} ranked chunks"
        )

        return scored_chunks
