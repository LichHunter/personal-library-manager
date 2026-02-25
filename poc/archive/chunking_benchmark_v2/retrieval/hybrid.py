"""Hybrid retrieval strategy - BM25 + semantic with RRF fusion."""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument


class HybridRetrieval(RetrievalStrategy, EmbedderMixin):
    """Hybrid BM25 + semantic retrieval with Reciprocal Rank Fusion.
    
    Combines lexical (BM25) and semantic (embedding) search using RRF
    to get the best of both approaches.
    
    Args:
        rrf_k: RRF constant (default 60). Higher values give more weight
               to lower-ranked results.
        candidate_multiplier: How many candidates to consider from each
                             method before fusion (multiplier of k).
    """

    def __init__(
        self,
        name: str = "hybrid",
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.rrf_k = rrf_k
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
        """Retrieve using RRF fusion of BM25 and semantic scores."""
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        n_candidates = min(k * self.candidate_multiplier, len(self.chunks))

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

        # Sort by RRF score
        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "rrf_k": self.rrf_k,
        }
