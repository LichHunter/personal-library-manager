"""Semantic retrieval scorer using cosine similarity between embeddings.

This module implements SimilarityScorer for dense vector retrieval using cosine
similarity. It computes the similarity between a query embedding and multiple
chunk embeddings, returning scored chunks sorted by relevance.

SimilarityScorer is the semantic (dense) half of hybrid retrieval (BM25 + semantic).
It excels at semantic understanding and is fast with pre-computed embeddings.

Example:
    >>> from plm.search.components.semantic import SimilarityScorer
    >>> scorer = SimilarityScorer()
    >>> data = {
    ...     'query_embedding': (1.0, 0.0, 0.0),
    ...     'chunk_embeddings': [
    ...         {'id': 'c1', 'content': 'text1', 'embedding': (1.0, 0.0, 0.0)},
    ...         {'id': 'c2', 'content': 'text2', 'embedding': (0.0, 1.0, 0.0)},
    ...         {'id': 'c3', 'content': 'text3', 'embedding': (0.5, 0.5, 0.0)}
    ...     ]
    ... }
    >>> results = scorer.process(data)
    >>> # results[0].score ≈ 1.0 (perfect match)
    >>> # results[0].source == 'semantic'
"""

from typing import Any
import logging
import numpy as np

from plm.search.types import ScoredChunk

logger = logging.getLogger(__name__)


class SimilarityScorer:
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
        ...     'chunk_embeddings': [
        ...         {'id': 'c1', 'content': 'text1', 'embedding': (1.0, 0.0, 0.0)},
        ...         {'id': 'c2', 'content': 'text2', 'embedding': (0.0, 1.0, 0.0)}
        ...     ]
        ... }
        >>> results = scorer.process(data)
        >>> len(results)
        2
        >>> results[0].source
        'semantic'
        >>> results[0].score
        1.0
    """

    def __init__(self):
        """Initialize SimilarityScorer component."""
        logger.debug("[SimilarityScorer] Initialized")

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
                - 'chunk_embeddings' (list): List of chunk dicts with 'embedding' field
                  or list of embedding vectors directly

        Returns:
            List of ScoredChunk objects sorted by similarity (descending).
            Each chunk has:
            - chunk_id: String ID from chunk dict or index
            - content: Content from chunk dict or empty string
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
            ...     'chunk_embeddings': [
            ...         {'id': 'c1', 'content': 'text1', 'embedding': (1.0, 0.0, 0.0)},
            ...         {'id': 'c2', 'content': 'text2', 'embedding': (0.0, 1.0, 0.0)}
            ...     ]
            ... }
            >>> results = scorer.process(data)
            >>> results[0].score
            1.0
            >>> results[1].score
            0.0
        """
        logger.debug("[SimilarityScorer] Scoring chunks with query embedding")

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

        # Extract embeddings and metadata from chunk dicts if needed
        chunk_ids = []
        chunk_contents = []
        embeddings_list = []

        for i, chunk in enumerate(chunk_embeddings):
            if isinstance(chunk, dict):
                # Extract embedding from dict
                if "embedding" not in chunk:
                    raise KeyError(f"Chunk at index {i} missing 'embedding' field")
                embeddings_list.append(chunk["embedding"])
                chunk_ids.append(chunk.get("id", str(i)))
                chunk_contents.append(chunk.get("content", ""))
            else:
                # Assume it's a raw embedding vector
                embeddings_list.append(chunk)
                chunk_ids.append(str(i))
                chunk_contents.append("")

        # Convert to numpy arrays
        try:
            query_emb = np.array(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Query embedding must be array-like, got {type(query_embedding).__name__}: {e}"
            )

        try:
            chunk_embs = np.array(embeddings_list, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Chunk embeddings must be array-like, got {type(embeddings_list).__name__}: {e}"
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
        logger.debug(
            f"[SimilarityScorer] Computed similarities for {len(similarities)} chunks"
        )

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Create ScoredChunk objects
        scored_chunks = []
        for rank, idx in enumerate(sorted_indices, start=1):
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=chunk_ids[idx],
                    content=chunk_contents[idx],
                    score=float(similarities[idx]),
                    source="semantic",
                    rank=rank,
                )
            )

        logger.debug(
            f"[SimilarityScorer] Completed scoring: {len(scored_chunks)} chunks ranked"
        )
        return scored_chunks
