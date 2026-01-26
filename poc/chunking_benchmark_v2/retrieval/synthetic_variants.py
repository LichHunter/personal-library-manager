"""Synthetic Query Variants retrieval strategy.

Generates diverse query variants using LLM and fuses results.
Addresses vocabulary mismatch by exploring alternative phrasings.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument
from .gem_utils import (
    reciprocal_rank_fusion,
    measure_latency,
)

from enrichment.provider import call_llm


VARIANT_PROMPT = """Generate 3 diverse search queries for: {query}
Vary terminology, specificity, and framing.
Output exactly 3 queries, one per line:"""


class SyntheticVariantsRetrieval(RetrievalStrategy, EmbedderMixin):
    """Synthetic query variants with LLM-based query expansion."""

    def __init__(self, name: str = "synthetic_variants", **kwargs):
        super().__init__(name, **kwargs)
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.cache: dict[str, list[str]] = {}

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks for retrieval."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks

        texts = [chunk.content for chunk in chunks]
        self.embeddings = self.encode_texts(texts)

        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def _base_retrieve(self, query: str, k: int = 15) -> list[Chunk]:
        """Base hybrid retrieval: BM25 + semantic with RRF fusion."""
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1][:k]
        sem_results = [self.chunks[i] for i in sem_ranks]

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [self.chunks[i] for i in bm25_ranks]

        return reciprocal_rank_fusion([sem_results, bm25_results])

    def _generate_variants(self, query: str) -> list[str]:
        """Generate query variants using LLM.

        Args:
            query: Original query

        Returns:
            List of 3 variant queries (or empty list if LLM fails)
        """
        if query in self.cache:
            return self.cache[query]

        try:
            response = call_llm(
                VARIANT_PROMPT.format(query=query), model="claude-haiku", timeout=5
            )

            variants = [v.strip() for v in response.split("\n") if v.strip()][:3]
            self.cache[query] = variants
            return variants

        except Exception as e:
            print(f"Warning: LLM variant generation failed: {e}")
            self.cache[query] = []
            return []

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks using synthetic query variants.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of top-k chunks
        """
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        variants = self._generate_variants(query)

        all_results = []
        for variant_query in [query] + variants:
            results = self._base_retrieve(variant_query, k=15)
            all_results.append(results)

        if all_results:
            fused = reciprocal_rank_fusion(all_results)
            return fused[:k]
        else:
            return []

    def get_index_stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "cache_size": len(self.cache),
        }
