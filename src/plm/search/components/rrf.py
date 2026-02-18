"""Reciprocal Rank Fusion (RRF) component for combining multiple retrieval signals.

This module implements RRFFuser for fusing results from multiple retrievers (BM25, semantic)
using the Reciprocal Rank Fusion algorithm. RRF is the standard method for hybrid retrieval,
combining sparse (lexical) and dense (semantic) signals.

The RRF formula: score(d) = Σ (weight_i / (k + rank_i(d)))

Where:
- weight_i: Weight for retriever i (default 1.0 for equal weighting)
- k: RRF parameter controlling blend uniformity (default 60)
- rank_i(d): Rank of document d in retriever i's results (1-based)

Example:
    >>> from plm.search.types import ScoredChunk, FusionConfig
    >>> from plm.search.components.rrf import RRFFuser
    >>> fuser = RRFFuser(FusionConfig(k=60, bm25_weight=1.0, semantic_weight=1.0))
    >>> bm25_results = [
    ...     ScoredChunk('a', 'content a', 10.0, 'bm25', 1),
    ...     ScoredChunk('b', 'content b', 8.0, 'bm25', 2)
    ... ]
    >>> semantic_results = [
    ...     ScoredChunk('b', 'content b', 0.9, 'semantic', 1),
    ...     ScoredChunk('a', 'content a', 0.7, 'semantic', 2)
    ... ]
    >>> fused = fuser.process([bm25_results, semantic_results])
    >>> # fused = [
    >>> #     ScoredChunk('b', 'content b', 0.032, 'rrf', 1),
    >>> #     ScoredChunk('a', 'content a', 0.031, 'rrf', 2)
    >>> # ]
"""

from plm.search.types import ScoredChunk, FusionConfig


class RRFFuser:
    """Reciprocal Rank Fusion component for combining multiple retrieval signals.

    Implements stateless RRF scoring for fusing results from multiple retrievers.
    Accepts a list of ScoredChunk lists (one per retriever) and returns a single
    fused list sorted by RRF score.

    The RRF algorithm:
    1. For each retriever, iterate through ranked results
    2. For each chunk, accumulate RRF score: weight / (k + rank)
    3. Sort chunks by accumulated RRF score (descending)
    4. Return fused results with source="rrf"

    Attributes:
        config: FusionConfig with k parameter and weights

    Example:
        >>> config = FusionConfig(k=60, bm25_weight=1.0, semantic_weight=1.0)
        >>> fuser = RRFFuser(config)
        >>> bm25_results = [ScoredChunk('a', 'text', 10.0, 'bm25', 1)]
        >>> semantic_results = [ScoredChunk('a', 'text', 0.9, 'semantic', 1)]
        >>> fused = fuser.process([bm25_results, semantic_results])
        >>> len(fused)
        1
        >>> fused[0].source
        'rrf'
    """

    def __init__(self, config: FusionConfig):
        """Initialize RRFFuser with fusion configuration.

        Args:
            config: FusionConfig with k parameter and weights
                - k: RRF parameter (default 60)
                - bm25_weight: Weight for BM25 signal (default 0.5)
                - semantic_weight: Weight for semantic signal (default 0.5)
        """
        self.config = config

    def process(self, data: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

        Combines results from multiple retrievers (BM25, semantic, etc.) using RRF.
        Each retriever contributes a weighted score based on the chunk's rank in
        that retriever's results.

        Args:
            data: List of ScoredChunk lists, one per retriever.
                Each list should be sorted by relevance (highest first).
                Example: [bm25_results, semantic_results]

        Returns:
            List of ScoredChunk objects sorted by RRF score (descending).
            Each chunk has:
            - chunk_id: Original chunk ID
            - content: Original chunk content
            - score: RRF score (sum of weighted reciprocal ranks)
            - source: "rrf" (fusion signal)
            - rank: Position in fused results (1-based)

        Raises:
            ValueError: If data is empty or contains empty result lists
            TypeError: If data is not a list of lists
            KeyError: If weights don't match number of result lists

        Example:
            >>> fuser = RRFFuser(FusionConfig(k=60, bm25_weight=1.0, semantic_weight=1.0))
            >>> bm25 = [ScoredChunk('a', 'text', 10.0, 'bm25', 1)]
            >>> semantic = [ScoredChunk('a', 'text', 0.9, 'semantic', 1)]
            >>> fused = fuser.process([bm25, semantic])
            >>> fused[0].score > 0
            True
        """
        # Validate input
        if not isinstance(data, list):
            raise TypeError(f"Input must be list of lists, got {type(data).__name__}")

        if not data:
            raise ValueError("Input list cannot be empty")

        # Check that all elements are lists
        for i, result_list in enumerate(data):
            if not isinstance(result_list, list):
                raise TypeError(
                    f"Element {i} must be list, got {type(result_list).__name__}"
                )

        # Extract weights from config
        # Support both explicit weights and default equal weights
        weights = self._extract_weights(len(data))

        # Build RRF scores
        rrf_scores: dict[str, float] = {}  # chunk_id -> accumulated RRF score
        chunk_lookup: dict[str, ScoredChunk] = {}  # chunk_id -> ScoredChunk

        for retriever_idx, result_list in enumerate(data):
            weight = weights[retriever_idx]

            for rank, scored_chunk in enumerate(result_list, start=1):
                chunk_id = scored_chunk.chunk_id

                # Store chunk for later lookup
                chunk_lookup[chunk_id] = scored_chunk

                # Accumulate RRF score
                rrf_score = weight / (self.config.k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

        # Sort by RRF score (descending)
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        # Create fused ScoredChunk objects
        fused_chunks = []
        for rank, chunk_id in enumerate(sorted_chunk_ids, start=1):
            original_chunk = chunk_lookup[chunk_id]
            fused_chunks.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    content=original_chunk.content,
                    score=rrf_scores[chunk_id],
                    source="rrf",
                    rank=rank,
                )
            )

        return fused_chunks

    def _extract_weights(self, num_retrievers: int) -> list[float]:
        """Extract weights from config for the given number of retrievers.

        Supports flexible weight extraction:
        - If num_retrievers == 2: Use bm25_weight and semantic_weight
        - If num_retrievers == 1: Use equal weight (1.0)
        - Otherwise: Use equal weights (1.0 for each)

        Args:
            num_retrievers: Number of retriever result lists

        Returns:
            List of weights, one per retriever

        Raises:
            ValueError: If weights are invalid (negative or zero)
        """
        if num_retrievers == 2:
            # Standard case: BM25 + semantic
            weights = [self.config.bm25_weight, self.config.semantic_weight]
        else:
            # Single retriever or more than 2: use equal weights
            weights = [1.0] * num_retrievers

        # Validate weights
        for i, weight in enumerate(weights):
            if weight <= 0:
                raise ValueError(f"Weight {i} must be positive, got {weight}")

        return weights
