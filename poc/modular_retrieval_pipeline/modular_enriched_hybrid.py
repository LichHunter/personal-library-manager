"""Modular Enriched Hybrid retrieval orchestrator (without LLM rewriting).

This module provides a stateful orchestrator that replicates the exact behavior of
EnrichedHybridLLMRetrieval from chunking_benchmark_v2 using modular components,
but WITHOUT LLM-based query rewriting.

The orchestrator produces results using:
- RRF order (semantic FIRST, BM25 SECOND)
- Adaptive weights (bm25=3.0, sem=0.3, rrf_k=10 when expanded; else 1.0, 1.0, 60)
- Multiplier (10x normal, 20x when expanded)
- dict.get(idx, 0) accumulation pattern

Architecture:
    ModularEnrichedHybrid (Stateful Orchestrator)
    ├── State: chunks, embeddings, bm25, _enriched_contents
    ├── Components: enrichment_pipeline, query_expander
    └── Methods: set_embedder(), index(), retrieve()

Example:
    >>> from modular_enriched_hybrid import ModularEnrichedHybrid
    >>> retriever = ModularEnrichedHybrid()
    >>> retriever.set_embedder(embedder)
    >>> retriever.index(chunks)
    >>> results = retriever.retrieve("What is HPA?", k=5)
"""

import time
from typing import Optional, Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .components.keyword_extractor import KeywordExtractor
from .components.entity_extractor import EntityExtractor
from .components.content_enricher import ContentEnricher
from .components.query_expander import QueryExpander, DOMAIN_EXPANSIONS
from .base import Pipeline
from .types import Query
from .cache.redis_client import RedisCacheClient
from .cache.cached_keyword_extractor import CachedKeywordExtractor
from .cache.cached_entity_extractor import CachedEntityExtractor
from .cache.caching_embedder import CachingEmbedder


class Embedder(Protocol):
    """Protocol for embedder interface."""

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        ...


class Chunk(Protocol):
    """Protocol for chunk interface (matches strategies.Chunk)."""

    id: str
    doc_id: str
    content: str


class ModularEnrichedHybrid:
    """Stateful orchestrator for enriched hybrid retrieval using modular components.

    This class replicates the behavior of EnrichedHybridLLMRetrieval from
    chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py using modular components,
    but skips LLM-based query rewriting for faster latency.

    The orchestrator:
    - Uses KeywordExtractor, EntityExtractor, ContentEnricher for enrichment pipeline
    - Uses QueryExpander (with DOMAIN_EXPANSIONS) for domain-specific expansion
    - Implements EXACT RRF fusion: semantic results FIRST, BM25 SECOND
    - Uses EXACT adaptive weights based on expansion_triggered
    - Skips LLM query rewriting (uses original query directly)

    State:
        chunks: List of indexed chunks
        embeddings: Numpy array of chunk embeddings
        bm25: BM25Okapi index
        _enriched_contents: List of enriched content strings

    Attributes:
        rrf_k: RRF parameter k (default 60, 10 when expanded)
        candidate_multiplier: Multiplier for k (default 10, 20 when expanded)
        verbose: Print progress messages (default True)
        debug: Enable debug logging (default False)
    """

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
        verbose: bool = True,
        debug: bool = False,
        cache: Optional[RedisCacheClient] = None,
    ):
        """Initialize ModularEnrichedHybrid orchestrator.

        Args:
            rrf_k: RRF parameter k (default 60)
            candidate_multiplier: Multiplier for candidate retrieval (default 10)
            verbose: Print progress messages (default True)
            debug: Enable debug logging (default False)
            cache: Optional RedisCacheClient for caching components (default None)
        """
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.verbose = verbose
        self.debug = debug
        self._cache = cache

        # State (set by index())
        self.chunks: Optional[list[Any]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self._enriched_contents: list[str] = []

        # Embedder (set by set_embedder())
        self.embedder: Optional[Any] = None

        # Modular components
        if cache:
            self._keyword_extractor = CachedKeywordExtractor(
                KeywordExtractor(max_keywords=10), cache
            )
            self._entity_extractor = CachedEntityExtractor(EntityExtractor(), cache)
        else:
            self._keyword_extractor = KeywordExtractor(max_keywords=10)
            self._entity_extractor = EntityExtractor()

        self._content_enricher = ContentEnricher()
        self._query_expander = QueryExpander()

        # Build enrichment pipeline: KeywordExtractor -> EntityExtractor -> ContentEnricher
        self._enrichment_pipeline = (
            Pipeline()
            .add(self._keyword_extractor)
            .add(self._entity_extractor)
            .add(self._content_enricher)
        )

        # Stats
        self.enrichment_time_s = 0.0
        self._enrichment_count = 0

    def _trace_log(self, msg: str):
        """Log trace message if debug is enabled."""
        if self.debug:
            print(f"[modular-enriched-hybrid] {msg}", flush=True)

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for encoding texts.

        Args:
            embedder: Embedder instance with encode() method
        """
        self.embedder = embedder

    def set_cached_embedder(self, embedder: Any, cache: RedisCacheClient) -> None:
        """Set embedder with caching wrapper.

        Args:
            embedder: Embedder instance with encode() method
            cache: RedisCacheClient for caching embeddings
        """
        if cache:
            self.embedder = CachingEmbedder(
                embedder, cache, model_name="BAAI/bge-base-en-v1.5"
            )
        else:
            self.embedder = embedder

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the embedder.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        return self.embedder.encode(texts, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query.

        Args:
            query: Query text to encode

        Returns:
            Numpy array embedding
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        return self.embedder.encode([query], show_progress_bar=False)[0]

    def _enrich_content(self, content: str) -> str:
        """Enrich content using modular pipeline.

        Runs content through KeywordExtractor -> EntityExtractor -> ContentEnricher
        to produce an enriched content string.

        Args:
            content: Original content text

        Returns:
            Enriched content string
        """
        self._enrichment_count += 1
        if self.verbose and self._enrichment_count % 50 == 0:
            # Show cache stats if available
            if self._cache:
                stats = self.get_cache_stats()
                if stats:
                    total = stats["total_hits"] + stats["total_misses"]
                    hit_rate = (stats["total_hits"] / total * 100) if total > 0 else 0
                    print(
                        f"    [enriching {self._enrichment_count}] cache hit rate: {hit_rate:.1f}% ({stats['total_hits']}/{total})",
                        flush=True,
                    )
            else:
                print(
                    f"    [enriching {self._enrichment_count}] modular pipeline...",
                    flush=True,
                )

        # Run through enrichment pipeline
        data = {"content": content}
        enriched = self._enrichment_pipeline.run(data)

        return enriched

    def index(
        self,
        chunks: list[Any],
        documents: Optional[list[Any]] = None,
        structured_docs: Optional[list[Any]] = None,
    ) -> None:
        """Index chunks for retrieval.

        Replicates lines 146-179 from enriched_hybrid_llm.py:
        1. Enrich each chunk using modular pipeline
        2. Encode enriched contents to embeddings
        3. Build BM25 index on enriched contents

        Args:
            chunks: List of Chunk objects to index
            documents: Optional list of Document objects (unused, for API compatibility)
            structured_docs: Optional list of StructuredDocument objects (unused)
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        enrichment_start = time.time()

        self.chunks = chunks
        enriched_contents = []

        for chunk in chunks:
            enriched = self._enrich_content(chunk.content)
            enriched_contents.append(enriched)

        self.enrichment_time_s = time.time() - enrichment_start

        if self.verbose:
            print(
                f"    [enriched] chunks={len(chunks)} "
                f"time={self.enrichment_time_s:.1f}s",
                flush=True,
            )

        # Encode enriched contents to embeddings
        self.embeddings = self.encode_texts(enriched_contents)

        # Build BM25 index on enriched contents
        tokenized = [content.lower().split() for content in enriched_contents]
        self.bm25 = BM25Okapi(tokenized)
        self._enriched_contents = enriched_contents

    def _expand_query_direct(self, query: str) -> tuple[str, bool]:
        """Expand query with domain-specific terms directly.

        This replicates the expand_query function from enriched_hybrid_llm.py
        (lines 45-78) to ensure EXACT same expansion behavior.

        Args:
            query: Query to expand

        Returns:
            Tuple of (expanded_query, expansion_triggered)
        """
        query_lower = query.lower()
        expansions_applied = []

        for term, expansion in DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                expansions_applied.append((term, expansion))

        if not expansions_applied:
            return query, False

        # Combine all expansions, avoiding duplicates
        expansion_terms = set()
        for _, expansion in expansions_applied:
            expansion_terms.update(expansion.split())

        # Remove terms already in the query
        query_terms = set(query_lower.split())
        new_terms = expansion_terms - query_terms

        expanded = f"{query} {' '.join(sorted(new_terms))}"
        return expanded, True

    def retrieve(self, query: str, k: int = 5) -> list[Any]:
        """Retrieve top-k chunks for a query.

        Replicates lines 181-339 from enriched_hybrid_llm.py:
        1. Expand query with domain terms (skip LLM rewriting)
        2. Select adaptive weights based on expansion_triggered
        3. Get semantic and BM25 scores
        4. Fuse with RRF (semantic FIRST, BM25 SECOND)
        5. Return top-k chunks

        CRITICAL: This method replicates EXACT behavior:
        - RRF order: semantic results FIRST (lines 270-278), BM25 SECOND (lines 280-288)
        - dict.get(idx, 0) for RRF accumulation (not setdefault)
        - Adaptive weights: bm25=3.0, sem=0.3, rrf_k=10 when expanded
        - Multiplier: 10x normal, 20x when expanded

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of top-k Chunk objects in RRF score order
        """
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        self._trace_log(f"=== RETRIEVE START ===")
        self._trace_log(f"ORIGINAL_QUERY: {query}")

        # Skip LLM rewriting - use original query directly
        expanded_query, expansion_triggered = self._expand_query_direct(query)
        if expansion_triggered:
            self._trace_log(f"EXPANDED_QUERY: {expanded_query}")

        # Adaptive weights based on expansion_triggered (lines 210-217)
        # EXACT values from reference:
        bm25_weight = 3.0 if expansion_triggered else 1.0
        sem_weight = 0.3 if expansion_triggered else 1.0
        multiplier = (
            self.candidate_multiplier * 2
            if expansion_triggered
            else self.candidate_multiplier
        )
        rrf_k = 10 if expansion_triggered else self.rrf_k

        if expansion_triggered:
            self._trace_log(
                f"WEIGHTED_RRF: bm25_weight={bm25_weight} sem_weight={sem_weight} rrf_k={rrf_k}"
            )

        # Calculate n_candidates (line 233)
        n_candidates = min(k * multiplier, len(self.chunks))
        self._trace_log(
            f"n_candidates={n_candidates} (k={k} * multiplier={multiplier})"
        )

        # Get semantic scores (lines 238-251)
        q_emb = self.encode_query(expanded_query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1]

        self._trace_log(f"TOP-10 SEMANTIC SCORES:")
        for rank in range(min(10, len(sem_ranks))):
            idx = sem_ranks[rank]
            chunk = self.chunks[idx]
            content_preview = (
                chunk.content[:100].replace("\n", " ") if chunk.content else ""
            )
            self._trace_log(
                f"  SEM[{rank}] idx={idx} chunk_id={chunk.id} score={sem_scores[idx]:.4f} | {content_preview}..."
            )

        # Get BM25 scores (lines 253-265)
        bm25_scores = self.bm25.get_scores(expanded_query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        self._trace_log(f"TOP-10 BM25 SCORES:")
        for rank in range(min(10, len(bm25_ranks))):
            idx = bm25_ranks[rank]
            chunk = self.chunks[idx]
            content_preview = (
                chunk.content[:100].replace("\n", " ") if chunk.content else ""
            )
            self._trace_log(
                f"  BM25[{rank}] idx={idx} chunk_id={chunk.id} score={bm25_scores[idx]:.4f} | {content_preview}..."
            )

        # RRF fusion (lines 267-288)
        # CRITICAL: Use dict[int, float] and dict.get(idx, 0) pattern
        rrf_scores: dict[int, float] = {}
        rrf_components: dict[int, tuple[float, float]] = {}

        # CRITICAL: Process semantic results FIRST (lines 270-278)
        for rank, idx in enumerate(sem_ranks[:n_candidates]):
            sem_component = sem_weight / (rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_component
            if idx not in rrf_components:
                rrf_components[idx] = (0.0, 0.0)
            rrf_components[idx] = (
                rrf_components[idx][0] + sem_component,
                rrf_components[idx][1],
            )

        # CRITICAL: Process BM25 results SECOND (lines 280-288)
        for rank, idx in enumerate(bm25_ranks[:n_candidates]):
            bm25_component = bm25_weight / (rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_component
            if idx not in rrf_components:
                rrf_components[idx] = (0.0, 0.0)
            rrf_components[idx] = (
                rrf_components[idx][0],
                rrf_components[idx][1] + bm25_component,
            )

        # Log RRF calculation (lines 290-303)
        self._trace_log(f"RRF SCORE CALCULATION (top 10):")
        top_rrf_idx = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:10]
        for rank, idx in enumerate(top_rrf_idx):
            sem_comp, bm25_comp = rrf_components[idx]
            total = rrf_scores[idx]
            sem_rank = np.where(sem_ranks == idx)[0][0] if idx in sem_ranks else -1
            bm25_rank = np.where(bm25_ranks == idx)[0][0] if idx in bm25_ranks else -1
            self._trace_log(
                f"  RRF[{rank}] idx={idx} total={total:.4f} "
                f"(sem_rank={sem_rank} comp={sem_comp:.4f} + bm25_rank={bm25_rank} comp={bm25_comp:.4f})"
            )

        # Get top-k indices sorted by RRF score descending (line 305-307)
        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
            :k
        ]

        # Get result chunks (line 309)
        results = [self.chunks[i] for i in top_idx]

        # Log final results (lines 311-336)
        self._trace_log(f"FINAL TOP-{k} RESULTS:")
        for rank, idx in enumerate(top_idx):
            chunk = self.chunks[idx]
            enriched = self._enriched_contents[idx] if self._enriched_contents else ""
            enriched_preview = enriched[:200].replace("\n", " ") if enriched else ""

            sem_rank = np.where(sem_ranks == idx)[0][0] if idx in sem_ranks else -1
            bm25_rank = np.where(bm25_ranks == idx)[0][0] if idx in bm25_ranks else -1
            sem_comp, bm25_comp = rrf_components[idx]

            dominant = (
                "semantic"
                if sem_comp > bm25_comp
                else "bm25"
                if bm25_comp > sem_comp
                else "balanced"
            )

            self._trace_log(
                f"  RESULT[{rank + 1}] chunk_id={chunk.id} doc_id={chunk.doc_id} "
                f"rrf_score={rrf_scores[idx]:.4f} sem_rank={sem_rank} bm25_rank={bm25_rank} "
                f"dominant={dominant} | enriched: {enriched_preview}..."
            )

        self._trace_log(f"=== RETRIEVE END ===")
        return results

    def get_index_stats(self) -> dict:
        """Get statistics about the index.

        Returns:
            Dict with index statistics
        """
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "rrf_k": self.rrf_k,
            "enrichment_time_s": self.enrichment_time_s,
        }

    def get_cache_stats(self) -> Optional[dict]:
        """Get cache statistics for all cached components.

        Returns:
            Dict with cache stats if cache enabled, None otherwise
        """
        if not self._cache or not self._cache.is_connected():
            return None

        keyword_hits = (
            self._keyword_extractor.hits
            if hasattr(self._keyword_extractor, "hits")
            else 0
        )
        keyword_misses = (
            self._keyword_extractor.misses
            if hasattr(self._keyword_extractor, "misses")
            else 0
        )
        entity_hits = (
            self._entity_extractor.hits
            if hasattr(self._entity_extractor, "hits")
            else 0
        )
        entity_misses = (
            self._entity_extractor.misses
            if hasattr(self._entity_extractor, "misses")
            else 0
        )
        embedding_hits = self.embedder.hits if hasattr(self.embedder, "hits") else 0
        embedding_misses = (
            self.embedder.misses if hasattr(self.embedder, "misses") else 0
        )

        total_hits = keyword_hits + entity_hits + embedding_hits
        total_misses = keyword_misses + entity_misses + embedding_misses

        return {
            "enabled": True,
            "keyword_hits": keyword_hits,
            "keyword_misses": keyword_misses,
            "entity_hits": entity_hits,
            "entity_misses": entity_misses,
            "embedding_hits": embedding_hits,
            "embedding_misses": embedding_misses,
            "total_hits": total_hits,
            "total_misses": total_misses,
        }
