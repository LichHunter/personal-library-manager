"""Negation-Aware retrieval strategy.

Detects negation in queries (e.g., "Why doesn't X work?", "What should I NOT do?")
and adjusts retrieval to prioritize warning/caution content over positive instructions.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument
from .gem_utils import (
    detect_negation,
    reciprocal_rank_fusion,
    measure_latency,
)


class NegationAwareRetrieval(RetrievalStrategy, EmbedderMixin):
    """Negation-aware retrieval with query expansion and result filtering."""

    def __init__(self, name: str = "negation_aware", **kwargs):
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
        """Index chunks for retrieval."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks

        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        self.embeddings = self.encode_texts(texts)

        # Create BM25 index
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def _base_retrieve(self, query: str, k: int = 20) -> list[Chunk]:
        """Base hybrid retrieval: BM25 + semantic with RRF fusion."""
        # Semantic search
        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1][:k]
        sem_results = [self.chunks[i] for i in sem_ranks]

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [self.chunks[i] for i in bm25_ranks]

        # RRF fusion (equal weights by default)
        return reciprocal_rank_fusion([sem_results, bm25_results])

    def _expand_negation_query(self, query: str, neg_info: dict) -> str:
        """Expand query with negation-aware terms.

        Args:
            query: Original query
            neg_info: Negation detection result from detect_negation()

        Returns:
            Expanded query with warning/caution terms
        """
        expansion_terms = []

        # Add warning terms based on negation type
        if "prohibition" in neg_info["types"]:
            expansion_terms.extend(["warning", "caution", "avoid", "never"])
        if "failure" in neg_info["types"]:
            expansion_terms.extend(["error", "troubleshooting", "fix", "problem"])
        if "limitation" in neg_info["types"]:
            expansion_terms.extend(["limitation", "constraint", "maximum", "minimum"])
        if "consequence" in neg_info["types"]:
            expansion_terms.extend(["consequence", "result", "happens", "effect"])

        # Deduplicate and join
        unique_terms = list(set(expansion_terms))
        return f"{query} {' '.join(unique_terms)}"

    def _filter_negation_results(
        self, results: list[Chunk], neg_info: dict
    ) -> list[Chunk]:
        """Boost warning chunks, penalize positive-only chunks.

        Args:
            results: Retrieved chunks
            neg_info: Negation detection result

        Returns:
            Re-ranked chunks with negation-aware scoring
        """
        warning_keywords = {
            "warning",
            "caution",
            "avoid",
            "never",
            "don't",
            "not",
            "cannot",
            "should not",
        }
        positive_keywords = {
            "how to",
            "recommended",
            "implement",
            "setup",
            "enable",
            "configure",
        }

        scored_results = []
        for chunk in results:
            content_lower = chunk.content.lower()

            # Count warning keywords
            warning_count = sum(1 for kw in warning_keywords if kw in content_lower)

            # Count positive keywords
            positive_count = sum(1 for kw in positive_keywords if kw in content_lower)

            # Calculate boost score
            # Boost chunks with warnings, penalize chunks with only positive content
            boost_score = warning_count * 2.0 - positive_count * 0.5

            scored_results.append((chunk, boost_score))

        # Sort by boost score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored_results]

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks with negation-aware filtering.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of top-k chunks
        """
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        # 1. Detect negation
        neg_info = detect_negation(query)

        # 2. Expand query if negation detected
        if neg_info["has_negation"]:
            expanded_query = self._expand_negation_query(query, neg_info)
        else:
            expanded_query = query

        # 3. Base retrieval
        results = self._base_retrieve(expanded_query, k=20)

        # 4. Post-filter for negation
        if neg_info["has_negation"]:
            results = self._filter_negation_results(results, neg_info)

        return results[:k]

    def get_index_stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
        }
