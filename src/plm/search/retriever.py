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

import json
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

            # Store chunk with embedding and structured metadata
            self.storage.add_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                enriched_content=enriched_content,
                heading=heading,
                start_char=start_char,
                end_char=end_char,
                keywords_json=json.dumps(keywords) if keywords else None,
                entities_json=json.dumps(entities) if entities else None,
            )

        # Rebuild BM25 index on all enriched content
        if rebuild_index:
            self._rebuild_bm25_index()

        logger.debug(f"[HybridRetriever] Completed ingestion of document {doc_id}")

    def ingest_document_hierarchical(
        self,
        doc_id: str,
        source_file: str,
        headings: list[dict],
        *,
        rebuild_index: bool = True,
    ) -> None:
        """Ingest a document with hierarchical heading structure.

        Creates heading records and links chunks to headings.
        Aggregates keywords/entities/embeddings to heading and document levels.

        Args:
            doc_id: Unique document identifier
            source_file: Source file path
            headings: List of heading sections, each with:
                - 'heading' (str): Heading text
                - 'level' (int): Heading level (0=root, 2=##, etc.)
                - 'chunks' (list[dict]): Chunks with content, keywords, entities
            rebuild_index: Whether to rebuild BM25 index after ingestion
        """
        self.storage.add_document(doc_id, source_file)

        doc_keywords_lists: list[list] = []
        doc_entities_lists: list[list] = []
        doc_embeddings: list[np.ndarray] = []
        chunk_counter = 0

        for heading_idx, section in enumerate(headings):
            heading_text = section.get("heading", "(root)")
            heading_level = section.get("level", 0)
            section_chunks = section.get("chunks", [])

            heading_id = f"{doc_id}_h{heading_idx}"
            heading_keywords_lists: list[list] = []
            heading_entities_lists: list[list] = []
            heading_embeddings: list[np.ndarray] = []
            heading_start = None
            heading_end = None

            for chunk_idx, chunk in enumerate(section_chunks):
                chunk_id = f"{doc_id}_{chunk_counter}_{uuid.uuid4().hex[:8]}"
                content = chunk.get("text") or chunk.get("content", "")
                keywords = chunk.get("keywords", [])
                entities_list = chunk.get("entities", [])
                start_char = chunk.get("start_char")
                end_char = chunk.get("end_char")

                if heading_start is None or (start_char is not None and start_char < heading_start):
                    heading_start = start_char
                if heading_end is None or (end_char is not None and end_char > heading_end):
                    heading_end = end_char

                entities_dict = self._entities_list_to_dict(entities_list)

                enriched_content = self.enricher.process({
                    "content": content,
                    "keywords": keywords,
                    "entities": entities_dict,
                })

                embedding_result = self.embedder.process(enriched_content)
                embedding = np.array(embedding_result["embedding"], dtype=np.float32)

                self.storage.add_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=content,
                    embedding=embedding,
                    enriched_content=enriched_content,
                    heading=heading_text,
                    heading_id=heading_id,
                    chunk_index=chunk_idx,
                    start_char=start_char,
                    end_char=end_char,
                    keywords_json=json.dumps(keywords) if keywords else None,
                    entities_json=json.dumps(entities_list) if entities_list else None,
                )

                heading_keywords_lists.append(keywords)
                heading_entities_lists.append(entities_list)
                heading_embeddings.append(embedding)
                chunk_counter += 1

            if section_chunks:
                heading_keywords = self._dedupe_keywords(heading_keywords_lists)
                heading_entities = self._dedupe_entities(heading_entities_lists)
                heading_embedding = self._aggregate_embeddings(heading_embeddings)

                self.storage.add_heading(
                    heading_id=heading_id,
                    doc_id=doc_id,
                    heading_text=heading_text,
                    heading_level=heading_level,
                    start_char=heading_start,
                    end_char=heading_end,
                    embedding=heading_embedding,
                    keywords_json=json.dumps(heading_keywords) if heading_keywords else None,
                    entities_json=json.dumps(heading_entities) if heading_entities else None,
                )

                doc_keywords_lists.append(heading_keywords)
                doc_entities_lists.extend(heading_entities_lists)
                if heading_embedding is not None:
                    doc_embeddings.append(heading_embedding)

        if headings:
            doc_keywords = self._dedupe_keywords(doc_keywords_lists)
            doc_entities = self._dedupe_entities(doc_entities_lists)
            doc_embedding = self._aggregate_embeddings(doc_embeddings)

            self.storage.update_document_aggregates(
                doc_id=doc_id,
                embedding=doc_embedding,
                keywords_json=json.dumps(doc_keywords) if doc_keywords else None,
                entities_json=json.dumps(doc_entities) if doc_entities else None,
            )

        if rebuild_index:
            self._rebuild_bm25_index()

        logger.debug(f"[HybridRetriever] Ingested {doc_id} hierarchically: {chunk_counter} chunks, {len(headings)} headings")

    @staticmethod
    def _entities_list_to_dict(entities_list: list[dict]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for entity in entities_list:
            label = entity.get("label", "unknown")
            text = entity.get("text", "")
            if label not in seen:
                seen[label] = set()
                result[label] = []
            if text not in seen[label]:
                result[label].append(text)
                seen[label].add(text)
        return result

    @staticmethod
    def _dedupe_keywords(keywords_lists: list[list]) -> list[str]:
        seen_lower: set[str] = set()
        result: list[str] = []
        for keywords in keywords_lists:
            for kw in keywords:
                if isinstance(kw, str) and kw.lower() not in seen_lower:
                    result.append(kw)
                    seen_lower.add(kw.lower())
        return sorted(result)

    @staticmethod
    def _dedupe_entities(entities_lists: list[list]) -> dict[str, list[str]]:
        from collections import defaultdict
        by_label: dict[str, set[str]] = defaultdict(set)
        seen_lower: dict[str, set[str]] = defaultdict(set)
        for entities in entities_lists:
            for ent in entities:
                if isinstance(ent, dict):
                    label = ent.get("label", "unknown")
                    text = ent.get("text", "")
                    if text.lower() not in seen_lower[label]:
                        by_label[label].add(text)
                        seen_lower[label].add(text.lower())
        return {label: sorted(texts) for label, texts in by_label.items()}

    @staticmethod
    def _aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
        valid = [e for e in embeddings if e is not None and len(e) > 0]
        if not valid:
            return None
        return np.mean(valid, axis=0).astype(np.float32)

    def is_indexed(self) -> bool:
        """Check if documents have already been ingested and indexed."""
        return self.storage.get_chunk_count() > 0 and self.bm25_index is not None

    def batch_ingest(
        self,
        documents: list[dict],
        batch_size: int = 256,
        show_progress: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
        *,
        rebuild_index: bool = True,
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
            rebuild_index: Whether to rebuild BM25 index after ingestion.
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
                    keywords, entities,
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

        for idx, (doc_id, chunk_idx, content, enriched, heading, start_char, end_char, keywords, entities) in enumerate(all_meta):
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
                keywords_json=json.dumps(keywords) if keywords else None,
                entities_json=json.dumps(entities) if entities else None,
            )

        if rebuild_index:
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

    def retrieve_headings(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k headings using semantic search on heading embeddings.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of heading dicts with heading_id, doc_id, heading_text, score, etc.
        """
        all_headings = self.storage.get_all_headings()
        if not all_headings:
            logger.debug("[HybridRetriever] No headings available")
            return []

        headings_with_emb = [h for h in all_headings if h.get("embedding") is not None]
        if not headings_with_emb:
            logger.debug("[HybridRetriever] No heading embeddings available")
            return []

        query_embedding = self.embedder.process(query)["embedding"]
        query_emb = np.array(query_embedding, dtype=np.float32)

        embeddings = np.array([h["embedding"] for h in headings_with_emb], dtype=np.float32)
        scores = np.dot(embeddings, query_emb)

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            h = headings_with_emb[idx]
            results.append({
                "heading_id": h["id"],
                "doc_id": h["doc_id"],
                "heading_text": h["heading_text"],
                "heading_level": h.get("heading_level"),
                "score": float(scores[idx]),
                "keywords_json": h.get("keywords_json"),
                "entities_json": h.get("entities_json"),
                "start_char": h.get("start_char"),
                "end_char": h.get("end_char"),
            })

        logger.debug(f"[HybridRetriever] Retrieved {len(results)} headings for query: {query[:50]}...")
        return results

    def retrieve_documents(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k documents using semantic search on document embeddings.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of document dicts with doc_id, source_file, score, etc.
        """
        all_docs = self.storage.get_all_documents()
        if not all_docs:
            logger.debug("[HybridRetriever] No documents available")
            return []

        docs_with_emb = [d for d in all_docs if d.get("embedding") is not None]
        if not docs_with_emb:
            logger.debug("[HybridRetriever] No document embeddings available")
            return []

        query_embedding = self.embedder.process(query)["embedding"]
        query_emb = np.array(query_embedding, dtype=np.float32)

        embeddings = np.array([d["embedding"] for d in docs_with_emb], dtype=np.float32)
        scores = np.dot(embeddings, query_emb)

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            d = docs_with_emb[idx]
            results.append({
                "doc_id": d["id"],
                "source_file": d["source_file"],
                "score": float(scores[idx]),
                "keywords_json": d.get("keywords_json"),
                "entities_json": d.get("entities_json"),
            })

        logger.debug(f"[HybridRetriever] Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
