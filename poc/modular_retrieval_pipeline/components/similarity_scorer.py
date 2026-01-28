"""Semantic retrieval scorer using cosine similarity between embeddings.

This module implements SimilarityScorer for dense vector retrieval using cosine
similarity. It computes the similarity between a query embedding and multiple
chunk embeddings, returning scored chunks sorted by relevance.

SimilarityScorer is the semantic (dense) half of hybrid retrieval (BM25 + semantic).
It excels at semantic understanding and is fast with pre-computed embeddings.

Example:
    >>> scorer = SimilarityScorer()
    >>> data = {
    ...     'query_embedding': (1.0, 0.0, 0.0),
    ...     'chunk_embeddings': [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0)]
    ... }
    >>> results = scorer.process(data)
    >>> # results = [
    >>> #     ScoredChunk(chunk_id='0', content='', score=1.0, source='semantic', rank=1),
    >>> #     ScoredChunk(chunk_id='2', content='', score=0.707, source='semantic', rank=2),
    >>> #     ScoredChunk(chunk_id='1', content='', score=0.0, source='semantic', rank=3)
    >>> # ]
"""

from typing import Any
import numpy as np

from ..base import Component
from ..types import ScoredChunk


class SimilarityScorer(Component):
    """Semantic retrieval scorer using cosine similarity.

    Implements stateless cosine similarity scoring for chunks. Computes the
    similarity between a query embedding and chunk embeddings using normalized
    dot product (cosine similarity for normalized vectors).

    The cosine similarity metric:
    - Ranges from -1 to 1 (typically 0 to 1 for normalized embeddings)
    - 1.0 = identical direction (perfect match)
    - 0.0 = orthogonal (no similarity)
    - -1.0 = opposite direction (rare for normalized embeddings)

    Attributes:
        None (stateless component)

    Example:
        >>> scorer = SimilarityScorer()
        >>> data = {
        ...     'query_embedding': (1.0, 0.0, 0.0),
        ...     'chunk_embeddings': [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        ... }
        >>> results = scorer.process(data)
        >>> len(results)
        2
        >>> results[0].source
        'semantic'
        >>> results[0].score
        1.0
    """

    def process(self, data: dict[str, Any]) -> list[ScoredChunk]:
        """Score chunks using cosine similarity with query embedding.

        Computes cosine similarity between the query embedding and each chunk
        embedding. Returns scored chunks sorted by similarity (highest first).

        The algorithm:
        1. Normalize query embedding to unit vector
        2. Normalize each chunk embedding to unit vector
        3. Compute dot product (cosine similarity for normalized vectors)
        4. Sort by similarity score (descending)
        5. Return ScoredChunk objects with rank

        Args:
            data: Dictionary with required fields:
                - 'query_embedding' (tuple or array): Query embedding vector
                - 'chunk_embeddings' (list): List of chunk embedding vectors
                  (each can be tuple, list, or array)

        Returns:
            List of ScoredChunk objects sorted by similarity (descending).
            Each chunk has:
            - chunk_id: String index of chunk in original list
            - content: Empty string (embeddings don't contain content)
            - score: Cosine similarity score (0.0 to 1.0 for normalized embeddings)
            - source: "semantic" (dense vector signal)
            - rank: Position in sorted results (1-based)

        Raises:
            KeyError: If 'query_embedding' or 'chunk_embeddings' field is missing
            ValueError: If chunk_embeddings list is empty
            TypeError: If embeddings are not array-like
            ValueError: If embeddings have mismatched dimensions

        Example:
            >>> scorer = SimilarityScorer()
            >>> data = {
            ...     'query_embedding': (1.0, 0.0, 0.0),
            ...     'chunk_embeddings': [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            ... }
            >>> results = scorer.process(data)
            >>> results[0].score
            1.0
            >>> results[1].score
            0.0
        """
        # Validate input
        if "query_embedding" not in data:
            raise KeyError("Input dict must have 'query_embedding' field")
        if "chunk_embeddings" not in data:
            raise KeyError("Input dict must have 'chunk_embeddings' field")

        query_embedding = data["query_embedding"]
        chunk_embeddings = data["chunk_embeddings"]

        # Check if chunk_embeddings is empty (handle both lists and arrays)
        try:
            is_empty = len(chunk_embeddings) == 0
        except TypeError:
            is_empty = False

        if is_empty:
            raise ValueError("Chunk embeddings list cannot be empty")

        # Convert to numpy arrays
        try:
            query_emb = np.array(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Query embedding must be array-like, got {type(query_embedding).__name__}: {e}"
            )

        try:
            chunk_embs = np.array(chunk_embeddings, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Chunk embeddings must be array-like, got {type(chunk_embeddings).__name__}: {e}"
            )

        # Validate dimensions
        if query_emb.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1-dimensional, got shape {query_emb.shape}"
            )

        if chunk_embs.ndim != 2:
            raise ValueError(
                f"Chunk embeddings must be 2-dimensional, got shape {chunk_embs.shape}"
            )

        if query_emb.shape[0] != chunk_embs.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query embedding has {query_emb.shape[0]} dimensions, "
                f"but chunk embeddings have {chunk_embs.shape[1]} dimensions"
            )

        # Normalize query embedding to unit vector
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            raise ValueError("Query embedding is zero vector (all zeros)")
        query_normalized = query_emb / query_norm

        # Normalize chunk embeddings to unit vectors
        chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        # Handle zero vectors in chunks (set to zero similarity)
        chunk_normalized = np.divide(
            chunk_embs,
            chunk_norms,
            where=chunk_norms != 0,
            out=np.zeros_like(chunk_embs),
        )

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(chunk_normalized, query_normalized)

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Create ScoredChunk objects
        scored_chunks = []
        for rank, idx in enumerate(sorted_indices, start=1):
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=str(idx),
                    content="",  # Embeddings don't contain content
                    score=float(similarities[idx]),
                    source="semantic",
                    rank=rank,
                )
            )

        return scored_chunks
