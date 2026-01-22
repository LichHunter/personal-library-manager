"""Hybrid retrieval with cross-encoder reranking."""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, RerankerMixin, StructuredDocument


class HybridRerankRetrieval(RetrievalStrategy, EmbedderMixin, RerankerMixin):
    """Hybrid BM25 + semantic retrieval with cross-encoder reranking.
    
    Two-stage retrieval:
    1. Get candidates using hybrid RRF fusion (BM25 + semantic)
    2. Rerank candidates using a cross-encoder model
    
    Args:
        rrf_k: RRF constant (default 60).
        initial_k: Number of candidates to retrieve before reranking.
        candidate_multiplier: Multiplier for initial candidates relative to k.
    """

    def __init__(
        self,
        name: str = "hybrid_rerank",
        rrf_k: int = 60,
        initial_k: Optional[int] = None,
        candidate_multiplier: int = 4,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.rrf_k = rrf_k
        self.initial_k = initial_k
        self.candidate_multiplier = candidate_multiplier
        
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks with both embeddings and BM25."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks
        
        # Semantic index
        self.embeddings = self.encode_texts([c.content for c in chunks])
        
        # BM25 index
        tokenized = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve using hybrid + reranking."""
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        # Determine number of candidates
        n_initial = self.initial_k or (k * self.candidate_multiplier)
        n_initial = min(n_initial, len(self.chunks))
        n_candidates = min(n_initial * 2, len(self.chunks))

        # Semantic scores and ranks
        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1]

        # BM25 scores and ranks
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        # RRF fusion
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(sem_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)

        # Get top candidates for reranking
        candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:n_initial]
        candidate_chunks = [self.chunks[i] for i in candidates]

        # Rerank with cross-encoder
        if self.reranker is not None and candidate_chunks:
            return self.rerank(query, candidate_chunks, k)
        else:
            return candidate_chunks[:k]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "rrf_k": self.rrf_k,
            "has_reranker": self.reranker is not None,
        }
