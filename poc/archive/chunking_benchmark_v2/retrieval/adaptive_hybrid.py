"""Adaptive Hybrid retrieval strategy.

Dynamically adjusts BM25/semantic weights based on query technical score.
Technical queries (with code, configs, etc.) get higher BM25 weight.
Natural language queries get higher semantic weight.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument
from .gem_utils import (
    detect_technical_score,
    reciprocal_rank_fusion,
    measure_latency,
)


class AdaptiveHybridRetrieval(RetrievalStrategy, EmbedderMixin):
    """Adaptive hybrid retrieval with dynamic BM25/semantic weighting."""

    def __init__(self, name: str = "adaptive_hybrid", **kwargs):
        super().__init__(name, **kwargs)
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks for retrieval.

        Args:
            chunks: List of chunks to index
            documents: Optional list of source documents
            structured_docs: Optional structured documents (unused)
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks

        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        self.embeddings = self.encode_texts(texts)

        # Create BM25 index
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks with adaptive weighting.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of top-k chunks
        """
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        # 1. Detect technical score
        tech_score = detect_technical_score(query)

        # 2. Calculate adaptive weights
        if tech_score > 0.3:
            bm25_weight, sem_weight = 0.7, 0.3
        else:
            bm25_weight, sem_weight = 0.4, 0.6

        # 3. Run BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1][:20]
        bm25_results = [self.chunks[i] for i in bm25_ranks]

        # 4. Run semantic search
        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1][:20]
        sem_results = [self.chunks[i] for i in sem_ranks]

        # 5. RRF fusion with adaptive weights
        fused = reciprocal_rank_fusion(
            [bm25_results, sem_results], weights=[bm25_weight, sem_weight]
        )

        return fused[:k]

    def get_index_stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
        }
