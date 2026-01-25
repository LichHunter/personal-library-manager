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

        self._trace_log(f"QUERY: {query}")

        n_candidates = min(k * self.candidate_multiplier, len(self.chunks))
        self._trace_log(
            f"n_candidates={n_candidates} (k={k} * multiplier={self.candidate_multiplier})"
        )

        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1]

        # Log top semantic scores
        self._trace_log(f"TOP SEMANTIC SCORES (top 5):")
        for rank in range(min(5, len(sem_ranks))):
            idx = sem_ranks[rank]
            self._trace_log(f"  sem_rank[{rank}] idx={idx} score={sem_scores[idx]:.4f}")

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        # Log top BM25 scores
        self._trace_log(f"TOP BM25 SCORES (top 5):")
        for rank in range(min(5, len(bm25_ranks))):
            idx = bm25_ranks[rank]
            self._trace_log(
                f"  bm25_rank[{rank}] idx={idx} score={bm25_scores[idx]:.4f}"
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

        # Log RRF calculation for top candidates
        self._trace_log(f"RRF SCORE CALCULATION (top 5):")
        top_rrf_idx = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:5]
        for rank, idx in enumerate(top_rrf_idx):
            sem_comp, bm25_comp = rrf_components[idx]
            total = rrf_scores[idx]
            sem_rank = np.where(sem_ranks == idx)[0][0] if idx in sem_ranks else -1
            bm25_rank = np.where(bm25_ranks == idx)[0][0] if idx in bm25_ranks else -1
            self._trace_log(
                f"  rrf[{rank}] idx={idx} total={total:.4f} "
                f"(sem_rank={sem_rank} comp={sem_comp:.4f} + bm25_rank={bm25_rank} comp={bm25_comp:.4f})"
            )

        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
            :k
        ]

        results = [self.chunks[i] for i in top_idx]

        for rank, idx in enumerate(top_idx):
            chunk = self.chunks[idx]
            enriched = self._enriched_contents[idx] if self._enriched_contents else ""
            enriched_preview = enriched[:300].replace("\n", " ") if enriched else ""

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
                f"RESULT[{rank + 1}] doc={chunk.doc_id} rrf_score={rrf_scores[idx]:.4f} "
                f"sem_rank={sem_rank} bm25_rank={bm25_rank} dominant={dominant} | "
                f"enriched: {enriched_preview}..."
            )

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
