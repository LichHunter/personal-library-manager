"""Enriched Hybrid retrieval - BM25 + semantic with chunk enrichment."""

import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from enrichment import EnrichmentCache, EnrichmentResult
from enrichment.fast import FastEnricher


# Domain expansion dictionary for query expansion
# Addresses VOCABULARY_MISMATCH and ACRONYM_GAP root causes
DOMAIN_EXPANSIONS = {
    # RPO/RTO acronyms - 100% miss rate on disaster recovery queries
    "rpo": "recovery point objective RPO data loss backup",
    "recovery point objective": "RPO data loss backup disaster recovery",
    "rto": "recovery time objective RTO downtime recovery restore",
    "recovery time objective": "RTO downtime recovery restore disaster",
    # JWT terminology - "iat" is JWT-specific
    "jwt": "JSON web token JWT authentication iat exp issued claims",
    "token": "JWT authentication token iat exp issued claims expiration",
    # Database stack vocabulary
    "database stack": "PostgreSQL Redis Kafka database storage data layer",
    "database": "PostgreSQL Redis Kafka storage data layer",
    # Monitoring stack vocabulary
    "monitoring stack": "Prometheus Grafana Jaeger observability metrics tracing",
    "monitoring": "Prometheus Grafana Jaeger observability metrics tracing",
    "observability": "Prometheus Grafana Jaeger monitoring metrics tracing",
    # HPA/autoscaling terms
    "hpa": "horizontal pod autoscaler HPA scaling replicas CPU utilization",
    "autoscaling": "horizontal pod autoscaler HPA scaling replicas CPU utilization",
    "autoscaler": "horizontal pod autoscaler HPA scaling replicas CPU",
}


def expand_query(query: str, debug: bool = False) -> str:
    """Expand query with domain-specific terms.

    Checks for expansion terms (case-insensitive) and appends
    expansion terms to the query to improve retrieval.

    Args:
        query: Original query string
        debug: If True, log expansion details

    Returns:
        Expanded query string with domain terms appended
    """
    query_lower = query.lower()
    expansions_applied = []

    for term, expansion in DOMAIN_EXPANSIONS.items():
        if term in query_lower:
            expansions_applied.append((term, expansion))

    if not expansions_applied:
        return query

    # Combine all expansions, avoiding duplicates
    expansion_terms = set()
    for _, expansion in expansions_applied:
        expansion_terms.update(expansion.split())

    # Remove terms already in the query
    query_terms = set(query_lower.split())
    new_terms = expansion_terms - query_terms

    expanded = f"{query} {' '.join(sorted(new_terms))}"
    return expanded


class EnrichedHybridRetrieval(RetrievalStrategy, EmbedderMixin):
    def __init__(
        self,
        name: str = "enriched_hybrid",
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
        use_cache: bool = True,
        cache_dir: str = "enrichment_cache",
        verbose: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.use_cache = use_cache
        self.verbose = verbose
        self.debug = debug

        self.cache = EnrichmentCache(cache_dir) if use_cache else None
        self.enricher = FastEnricher(debug=debug)

        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

        self.enrichment_time_s = 0.0
        self._enrichment_count = 0
        self._cache_hits = 0
        self._enriched_contents: list[str] = []

    def _trace_log(self, msg: str):
        if self.debug:
            from logger import get_logger

            get_logger().trace(f"[enriched-hybrid] {msg}")

    def _enrich_content(self, content: str, context: Optional[dict] = None) -> str:
        if self.cache:
            cached = self.cache.get(content, "fast", "fast")
            if cached:
                self._cache_hits += 1
                self._trace_log(f"CACHE HIT: {self._cache_hits} hits so far")
                return cached.enhanced_content

        self._enrichment_count += 1
        if self.verbose and self._enrichment_count % 50 == 0:
            print(
                f"    [enriching {self._enrichment_count}] fast (YAKE+spaCy)...",
                flush=True,
            )

        result = self.enricher.enrich(content, context)

        if self.cache:
            self.cache.put(content, "fast", "fast", result)

        self._trace_log(
            f"CACHE MISS: enriched chunk (total enrichments: {self._enrichment_count})"
        )
        return result.enhanced_content

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
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
            stats = self.enricher.get_stats()
            print(
                f"    [enriched] chunks={len(chunks)} new={self._enrichment_count} "
                f"cache_hits={self._cache_hits} time={self.enrichment_time_s:.1f}s "
                f"avg={stats['avg_time_ms']:.1f}ms/chunk",
                flush=True,
            )

        self.embeddings = self.encode_texts(enriched_contents)

        tokenized = [content.lower().split() for content in enriched_contents]
        self.bm25 = BM25Okapi(tokenized)
        self._enriched_contents = enriched_contents

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.chunks is None or self.embeddings is None or self.bm25 is None:
            return []

        self._trace_log(f"=== RETRIEVE START ===")
        self._trace_log(f"ORIGINAL_QUERY: {query}")

        expanded_query = expand_query(query, debug=self.debug)
        if expanded_query != query:
            self._trace_log(f"EXPANDED_QUERY (BM25 only): {expanded_query}")

        if (
            hasattr(self, "embedder")
            and self.embedder
            and hasattr(self.embedder, "prefix")
        ):
            prefix = getattr(self.embedder, "prefix", "")
            if prefix:
                self._trace_log(f"ENRICHMENT_PREFIX: {prefix}")

        n_candidates = min(k * self.candidate_multiplier, len(self.chunks))
        self._trace_log(
            f"n_candidates={n_candidates} (k={k} * multiplier={self.candidate_multiplier})"
        )

        q_emb = self.encode_query(query)
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

        # Calculate RRF scores with detailed logging
        rrf_scores: dict[int, float] = {}
        rrf_components: dict[
            int, tuple[float, float]
        ] = {}  # (sem_component, bm25_component)

        for rank, idx in enumerate(sem_ranks[:n_candidates]):
            sem_component = 1 / (self.rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_component
            if idx not in rrf_components:
                rrf_components[idx] = (0.0, 0.0)
            rrf_components[idx] = (
                rrf_components[idx][0] + sem_component,
                rrf_components[idx][1],
            )

        for rank, idx in enumerate(bm25_ranks[:n_candidates]):
            bm25_component = 1 / (self.rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_component
            if idx not in rrf_components:
                rrf_components[idx] = (0.0, 0.0)
            rrf_components[idx] = (
                rrf_components[idx][0],
                rrf_components[idx][1] + bm25_component,
            )

        # Log RRF calculation for top-10 candidates
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

        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
            :k
        ]

        results = [self.chunks[i] for i in top_idx]

        # Log final top-k results with dominant signal
        self._trace_log(f"FINAL TOP-{k} RESULTS:")
        for rank, idx in enumerate(top_idx):
            chunk = self.chunks[idx]
            enriched = self._enriched_contents[idx] if self._enriched_contents else ""
            enriched_preview = enriched[:200].replace("\n", " ") if enriched else ""

            # Get ranks for this chunk
            sem_rank = np.where(sem_ranks == idx)[0][0] if idx in sem_ranks else -1
            bm25_rank = np.where(bm25_ranks == idx)[0][0] if idx in bm25_ranks else -1
            sem_comp, bm25_comp = rrf_components[idx]

            # Determine which signal dominated
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
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
            "rrf_k": self.rrf_k,
            "enrichment_time_s": self.enrichment_time_s,
            "enricher_stats": self.enricher.get_stats(),
        }
