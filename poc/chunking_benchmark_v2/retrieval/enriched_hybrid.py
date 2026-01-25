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

        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1]

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(sem_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)

        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
            :k
        ]

        results = [self.chunks[i] for i in top_idx]

        for rank, idx in enumerate(top_idx):
            chunk = self.chunks[idx]
            enriched = self._enriched_contents[idx] if self._enriched_contents else ""
            enriched_preview = enriched[:300].replace("\n", " ") if enriched else ""
            self._trace_log(
                f"RESULT[{rank + 1}] doc={chunk.doc_id} score={rrf_scores[idx]:.4f} | "
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
