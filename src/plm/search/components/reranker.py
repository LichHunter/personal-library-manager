"""Cross-encoder reranker component for improving retrieval precision.

Uses a cross-encoder model to re-score query-document pairs. Cross-encoders
see both query and document together (unlike bi-encoders), enabling deeper
relevance understanding at the cost of higher latency.

The reranker is lazy-loaded on first use and cached for subsequent calls.

Example:
    >>> reranker = CrossEncoderReranker()
    >>> results = reranker.rerank("what is kubernetes?", candidates, top_k=10)
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


# Default model: fast, 80MB, good quality for documentation search
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Candidate multiplier: how many candidates to retrieve before reranking
DEFAULT_CANDIDATES_MULTIPLIER = 5


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Lazy-loads the model on first use. Thread-safe for single-threaded
    FastAPI/uvicorn usage (standard deployment).

    Attributes:
        model_name: HuggingFace model identifier
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        """Initialize reranker (model loaded lazily on first use).

        Args:
            model_name: Cross-encoder model from HuggingFace hub.
                Recommended: "cross-encoder/ms-marco-MiniLM-L-6-v2" (80MB, fast)
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """Load cross-encoder model on first use."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        logger.info(f"[Reranker] Loading model: {self.model_name}")
        start = time.time()
        self._model = CrossEncoder(self.model_name)
        elapsed = time.time() - start
        logger.info(f"[Reranker] Model loaded in {elapsed:.1f}s")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank candidates by cross-encoder relevance score.

        Scores each (query, candidate.content) pair with the cross-encoder
        and returns the top_k highest-scoring candidates.

        Args:
            query: Query text
            candidates: List of retrieval result dicts (must have 'content' key)
            top_k: Number of results to return after reranking

        Returns:
            List of top_k candidate dicts, sorted by cross-encoder score descending.
            Each dict gets an additional 'rerank_score' field.
        """
        if not candidates:
            return []

        self._load_model()
        assert self._model is not None

        # Build query-document pairs
        pairs = [[query, c.get("content", "")] for c in candidates]

        start = time.perf_counter()
        scores = self._model.predict(pairs)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"[Reranker] Scored {len(pairs)} pairs in {latency_ms:.0f}ms"
        )

        # Sort by score descending, take top_k
        scored_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in scored_indices:
            candidate = dict(candidates[idx])  # shallow copy
            candidate["rerank_score"] = float(scores[idx])
            results.append(candidate)

        return results
