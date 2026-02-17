"""HybridRetriever orchestrator combining BM25 and semantic search with RRF.

This module provides the main retrieval orchestrator that ties together all
search components: SQLite storage, BM25 lexical search, semantic embeddings,
content enrichment, and query expansion.

The retriever implements:
- RRF order: semantic FIRST, BM25 SECOND (critical for behavior parity)
- Adaptive weights: k=60, bm25_weight=1.0, sem_weight=1.0 (default)
                   k=10, bm25_weight=3.0, sem_weight=0.3 (when expanded)
- Candidate multiplier: 10x normal, 20x when expanded
- BM25 persistence: saves/loads index to disk

Example:
    >>> from plm.search.retriever import HybridRetriever
    >>> retriever = HybridRetriever('/path/to/db.sqlite', '/path/to/bm25_index')
    >>> retriever.ingest_document('doc1', 'test.md', [
    ...     {'content': 'Kubernetes pod definition', 'keywords': ['kubernetes'], 'entities': {}}
    ... ])
    >>> results = retriever.retrieve('kubernetes pod', k=5)
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from plm.search.components.bm25 import BM25Index
from plm.search.components.embedder import EmbeddingEncoder
from plm.search.components.enricher import ContentEnricher
from plm.search.components.expander import QueryExpander
from plm.search.components.query_rewriter import QueryRewriter
from plm.search.storage.sqlite import SQLiteStorage
from plm.search.types import Query, RewrittenQuery


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval orchestrator with BM25 + semantic search and RRF fusion.

    This class orchestrates the complete retrieval pipeline:
    - Document ingestion with content enrichment and embedding generation
    - Hybrid search combining BM25 (lexical) and semantic (embedding) signals
    - RRF (Reciprocal Rank Fusion) for combining rankings
    - BM25 index persistence to disk

    Critical Behavior (must preserve for POC parity):
    - RRF processes semantic results FIRST, then BM25 SECOND
    - Uses dict.get(idx, 0) accumulation pattern for RRF scores
    - Adaptive weights when query expansion is triggered:
        Default: k=60, bm25_weight=1.0, sem_weight=1.0
        Expanded: k=10, bm25_weight=3.0, sem_weight=0.3
    - Candidate multiplier: 10x normal, 20x when expanded

    Attributes:
        db_path: Path to SQLite database file
        bm25_index_path: Path to BM25 index directory
        storage: SQLiteStorage instance for document/chunk persistence
        bm25_index: BM25Index instance for lexical search
        embedder: EmbeddingEncoder instance for generating embeddings
        enricher: ContentEnricher instance for content enrichment
        expander: QueryExpander instance for query expansion
    """

    # Default RRF parameters
    DEFAULT_RRF_K = 60
    DEFAULT_BM25_WEIGHT = 1.0
    DEFAULT_SEM_WEIGHT = 1.0

    # Expanded RRF parameters (when query expansion triggered)
    EXPANDED_RRF_K = 10
    EXPANDED_BM25_WEIGHT = 3.0
    EXPANDED_SEM_WEIGHT = 0.3

    # Candidate multipliers
    DEFAULT_CANDIDATE_MULTIPLIER = 10
    EXPANDED_CANDIDATE_MULTIPLIER = 20

    def __init__(self, db_path: str, bm25_index_path: str, rewrite_timeout: float = 5.0) -> None:
        """Initialize HybridRetriever.

        Creates SQLite storage and loads BM25 index if it exists on disk.

        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
            bm25_index_path: Path to BM25 index directory (loaded if exists)
            rewrite_timeout: Timeout in seconds for query rewriting (default 5.0)
        """
        self.db_path = db_path
        self.bm25_index_path = bm25_index_path
        self.rewrite_timeout = rewrite_timeout

        # Initialize storage and create tables
        self.storage = SQLiteStorage(db_path)
        self.storage.create_tables()

        # Initialize components
        self.embedder = EmbeddingEncoder()
        self.enricher = ContentEnricher()
        self.expander = QueryExpander()

        # Lazy-initialized query rewriter (only when use_rewrite=True)
        self._query_rewriter: QueryRewriter | None = None

        # Load or create BM25 index
        self.bm25_index: BM25Index | None = None
        if os.path.exists(bm25_index_path):
            try:
                self.bm25_index = BM25Index.load(bm25_index_path)
                logger.debug(f"[HybridRetriever] Loaded BM25 index from {bm25_index_path}")
            except Exception as e:
                logger.warning(f"[HybridRetriever] Failed to load BM25 index: {e}")
                self.bm25_index = None

        logger.debug(f"[HybridRetriever] Initialized with db={db_path}, bm25={bm25_index_path}, rewrite_timeout={rewrite_timeout}")

    def ingest_document(
        self,
        doc_id: str,
        source_file: str,
        chunks: list[dict],
        *,
        rebuild_index: bool = True,
    ) -> None:
        """Ingest a document with pre-extracted chunks.

        Processes each chunk:
        1. Enriches content using keywords and entities
        2. Generates embedding for enriched content
        3. Stores in SQLite

        After all chunks are processed (if rebuild_index=True):
        4. Rebuilds BM25 index on ALL enriched content
        5. Saves BM25 index to disk

        Args:
            doc_id: Unique document identifier
            source_file: Source file path
            chunks: List of chunk dicts with fields:
                - 'content' (str): Raw chunk text
                - 'keywords' (list[str]): Keywords from YAKE
                - 'entities' (dict[str, list[str]]): Entities from GLiNER
                - 'heading' (str, optional): Section heading
                - 'start_char' (int, optional): Start character offset
                - 'end_char' (int, optional): End character offset
            rebuild_index: Whether to rebuild BM25 index after ingestion.
                Set to False when batch-ingesting many documents, then call
                rebuild_index() manually after all documents are ingested.

        Example:
            >>> retriever.ingest_document('doc1', 'test.md', [
            ...     {'content': 'Kubernetes pod', 'keywords': ['kubernetes'], 'entities': {}}
            ... ])
        """
        logger.debug(f"[HybridRetriever] Ingesting document {doc_id} with {len(chunks)} chunks")

        # Add document to storage
        self.storage.add_document(doc_id, source_file)

        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}"

            content = chunk.get("content", "")
            keywords = chunk.get("keywords", [])
            entities = chunk.get("entities", {})
            heading = chunk.get("heading")
            start_char = chunk.get("start_char")
            end_char = chunk.get("end_char")

            # Enrich content using ContentEnricher
            enriched_content = self.enricher.process({
                "content": content,
                "keywords": keywords,
                "entities": entities,
            })

            # Generate embedding for enriched content
            embedding_result = self.embedder.process(enriched_content)
            embedding = np.array(embedding_result["embedding"], dtype=np.float32)

            # Store chunk with embedding
            self.storage.add_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                enriched_content=enriched_content,
                heading=heading,
                start_char=start_char,
                end_char=end_char,
            )

        # Rebuild BM25 index on all enriched content
        if rebuild_index:
            self._rebuild_bm25_index()

        logger.debug(f"[HybridRetriever] Completed ingestion of document {doc_id}")

    def is_indexed(self) -> bool:
        """Check if documents have already been ingested and indexed."""
        return self.storage.get_chunk_count() > 0 and self.bm25_index is not None

    def batch_ingest(
        self,
        documents: list[dict],
        batch_size: int = 256,
        show_progress: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Batch-ingest multiple documents with efficient batch encoding.

        Much faster than calling ingest_document() per document because
        embeddings are computed in a single batch rather than one-at-a-time.

        Args:
            documents: List of document dicts, each with:
                - 'doc_id' (str): Unique document identifier
                - 'source_file' (str): Source file path
                - 'chunks' (list[dict]): Chunks with 'content', 'keywords', 'entities'
            batch_size: Batch size for embedding encoding
            show_progress: Show progress bar during encoding
            on_progress: Optional callable(step, total, message) for progress reporting
        """
        all_enriched = []
        all_meta = []

        for doc in documents:
            doc_id = doc["doc_id"]
            source_file = doc["source_file"]
            self.storage.add_document(doc_id, source_file)

            for i, chunk in enumerate(doc["chunks"]):
                content = chunk.get("content", "")
                keywords = chunk.get("keywords", [])
                entities = chunk.get("entities", {})

                enriched = self.enricher.process({
                    "content": content, "keywords": keywords, "entities": entities,
                })
                all_enriched.append(enriched)
                all_meta.append((
                    doc_id, i, content, enriched,
                    chunk.get("heading"), chunk.get("start_char"), chunk.get("end_char"),
                ))

        if not all_enriched:
            return

        if on_progress:
            on_progress(1, 3, f"Enriched {len(all_enriched)} chunks")

        embedding_tuples = self.embedder.encode_batch(
            all_enriched, batch_size=batch_size, show_progress=show_progress,
        )

        if on_progress:
            on_progress(2, 3, f"Encoded {len(embedding_tuples)} embeddings")

        for idx, (doc_id, chunk_idx, content, enriched, heading, start_char, end_char) in enumerate(all_meta):
            chunk_id = f"{doc_id}_{chunk_idx}_{uuid.uuid4().hex[:8]}"
            embedding = np.array(embedding_tuples[idx], dtype=np.float32)
            self.storage.add_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                enriched_content=enriched,
                heading=heading,
                start_char=start_char,
                end_char=end_char,
            )

        self._rebuild_bm25_index()

        if on_progress:
            on_progress(3, 3, f"Indexed {len(all_enriched)} chunks")

        logger.info(f"[HybridRetriever] Batch ingested {len(documents)} documents, {len(all_enriched)} chunks")

    def rebuild_index(self) -> None:
        """Rebuild BM25 index from all stored chunks.

        Call this after batch-ingesting documents with rebuild_index=False.
        """
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Rebuild and persist BM25 index from all stored chunks."""
        # Get all chunks from storage
        all_chunks = self.storage.get_all_chunks()
        if not all_chunks:
            logger.debug("[HybridRetriever] No chunks to index")
            return

        # Extract enriched content for BM25 indexing
        enriched_contents = [
            chunk.get("enriched_content") or chunk.get("content", "")
            for chunk in all_chunks
        ]

        # Build BM25 index
        self.bm25_index = BM25Index()
        self.bm25_index.index(enriched_contents)

        # Ensure parent directory exists
        Path(self.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to disk
        self.bm25_index.save(self.bm25_index_path)
        logger.debug(f"[HybridRetriever] Saved BM25 index with {len(enriched_contents)} documents")

    def retrieve(self, query: str, k: int = 5, use_rewrite: bool = False) -> list[dict]:
        """Retrieve top-k chunks for a query using hybrid search with RRF.

        Implements the retrieval pipeline:
        1. Optionally rewrite query with QueryRewriter (if use_rewrite=True)
        2. Expand query with domain-specific terms
        3. Select adaptive weights based on expansion
        4. Get semantic scores (embedding similarity)
        5. Get BM25 scores (lexical search)
        6. Fuse with RRF (semantic FIRST, BM25 SECOND)
        7. Return top-k results

        CRITICAL: This method preserves EXACT POC behavior:
        - RRF order: semantic results FIRST, BM25 SECOND
        - dict.get(idx, 0) for RRF accumulation
        - Adaptive weights when expansion triggered

        Args:
            query: Query text
            k: Number of results to return (default: 5)
            use_rewrite: Whether to rewrite query before expansion (default: False)

        Returns:
            List of result dicts with fields:
                - 'chunk_id': Chunk identifier
                - 'doc_id': Document identifier
                - 'content': Original chunk content
                - 'enriched_content': Enriched content used for retrieval
                - 'score': RRF fusion score
                - 'heading': Section heading (if available)
                - 'start_char': Start character offset (if available)
                - 'end_char': End character offset (if available)

        Example:
            >>> results = retriever.retrieve('kubernetes autoscaling', k=5)
            >>> for r in results:
            ...     print(f"{r['chunk_id']}: {r['score']:.4f}")
        """
        # Get all chunks from storage
        all_chunks = self.storage.get_all_chunks()
        if not all_chunks or self.bm25_index is None:
            logger.debug("[HybridRetriever] No chunks or BM25 index available")
            return []

        # Create Query and optionally rewrite it
        original_query = Query(text=query)
        
        if use_rewrite:
            # Lazy-initialize QueryRewriter on first use
            if self._query_rewriter is None:
                self._query_rewriter = QueryRewriter(timeout=self.rewrite_timeout)
            
            # Rewrite the query
            rewritten_query = self._query_rewriter.process(original_query)
            logger.debug(f"[HybridRetriever] Query rewritten: {query} → {rewritten_query.rewritten}")
        else:
            # Pass through original query (backward compatible)
            rewritten_query = RewrittenQuery(
                original=original_query,
                rewritten=query,
                model="passthrough",
            )

        # Expand query
        expanded = self.expander.process(rewritten_query)
        expanded_query = expanded.expanded
        expansion_triggered = len(expanded.expansions) > 0

        # Select adaptive parameters
        if expansion_triggered:
            rrf_k = self.EXPANDED_RRF_K
            bm25_weight = self.EXPANDED_BM25_WEIGHT
            sem_weight = self.EXPANDED_SEM_WEIGHT
            multiplier = self.EXPANDED_CANDIDATE_MULTIPLIER
            logger.debug(
                f"[HybridRetriever] Expansion triggered: rrf_k={rrf_k}, "
                f"bm25_weight={bm25_weight}, sem_weight={sem_weight}"
            )
        else:
            rrf_k = self.DEFAULT_RRF_K
            bm25_weight = self.DEFAULT_BM25_WEIGHT
            sem_weight = self.DEFAULT_SEM_WEIGHT
            multiplier = self.DEFAULT_CANDIDATE_MULTIPLIER

        # Calculate number of candidates
        n_candidates = min(k * multiplier, len(all_chunks))

        # Build chunk index for lookup
        chunk_index = {i: chunk for i, chunk in enumerate(all_chunks)}

        # Get semantic scores
        query_embedding = self.embedder.process(expanded_query)["embedding"]
        query_emb = np.array(query_embedding, dtype=np.float32)

        # Build embeddings matrix from all chunks
        embeddings = np.array([
            chunk["embedding"] for chunk in all_chunks
        ], dtype=np.float32)

        # Compute cosine similarity (embeddings are already normalized by EmbeddingEncoder)
        sem_scores = np.dot(embeddings, query_emb)
        sem_ranks = np.argsort(sem_scores)[::-1]

        # Get BM25 results
        bm25_results = self.bm25_index.search(expanded_query, k=n_candidates)
        bm25_rank_map = {r["index"]: rank for rank, r in enumerate(bm25_results)}

        # RRF fusion
        # CRITICAL: Use dict[int, float] and dict.get(idx, 0) pattern
        rrf_scores: dict[int, float] = {}

        # CRITICAL: Process semantic results FIRST
        for rank, idx in enumerate(sem_ranks[:n_candidates]):
            sem_component = sem_weight / (rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_component

        # CRITICAL: Process BM25 results SECOND
        for rank, result in enumerate(bm25_results):
            idx = result["index"]
            bm25_component = bm25_weight / (rrf_k + rank)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_component

        # Get top-k indices sorted by RRF score descending
        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

        # Build result list
        results = []
        for idx in top_indices:
            chunk = chunk_index[idx]
            results.append({
                "chunk_id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "content": chunk["content"],
                "enriched_content": chunk.get("enriched_content", ""),
                "score": rrf_scores[idx],
                "heading": chunk.get("heading"),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            })

        logger.debug(f"[HybridRetriever] Retrieved {len(results)} results for query: {query[:50]}...")
        return results
