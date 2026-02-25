"""Flat embedding retrieval strategy - baseline approach."""

import logging
import time
import uuid
from typing import Optional

from ..core.protocols import Embedder, VectorStore, LLM, RetrievalStrategy
from ..core.types import (
    Document,
    Chunk,
    IndexStats,
    SearchResponse,
    SearchHit,
    SearchStats,
)

logger = logging.getLogger(__name__)


class FlatStrategy(RetrievalStrategy):
    """
    Flat embedding strategy - the baseline approach.
    
    Index:  Document → Chunks → Embed all → Store
    Search: Query → Embed → Top-K similarity → Return
    
    Simple, fast, and effective for many use cases. Serves as the
    baseline for comparing more sophisticated strategies.
    """
    
    COLLECTION_NAME = "flat_chunks"
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        self._embedder: Optional[Embedder] = None
        self._store: Optional[VectorStore] = None
        self._llm: Optional[LLM] = None  # Not used, but kept for interface
        
        self._is_indexed = False
        self._index_stats: Optional[IndexStats] = None
    
    @property
    def name(self) -> str:
        return "flat"
    
    @property
    def requires_llm(self) -> bool:
        return False
    
    def configure(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: Optional[LLM] = None,
    ) -> None:
        """Configure the strategy with required components."""
        self._embedder = embedder
        self._store = store
        self._llm = llm  # Ignored for flat strategy
        
        # Create collection
        self._store.create_collection(
            self.COLLECTION_NAME,
            dimension=self._embedder.dimension,
        )
    
    def index(self, documents: list[Document]) -> IndexStats:
        """
        Index documents by chunking and embedding all content.
        
        Process:
        1. Chunk each document into overlapping segments
        2. Embed all chunks in batches
        3. Store vectors with metadata
        """
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        start_time = time.perf_counter()
        
        # Chunk all documents
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Document {doc.id}: {len(chunks)} chunks")
        
        logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents"
        )
        
        if not all_chunks:
            logger.warning("No chunks to index")
            return IndexStats(
                strategy=self.name,
                backend=self._store.backend_name,
                embedding_model=self._embedder.model_name,
                llm_model=None,
                duration_sec=0.0,
                num_documents=len(documents),
                num_chunks=0,
                num_vectors=0,
                llm_calls=0,
                embed_calls=0,
            )
        
        # Embed all chunks
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embedder.embed_batch(texts)
        embed_calls = 1  # Single batch call
        
        # Prepare for insertion
        ids = [chunk.id for chunk in all_chunks]
        metadata = [
            {
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "content": chunk.content,
                "level": chunk.level,
                **chunk.metadata,
            }
            for chunk in all_chunks
        ]
        
        # Insert into vector store
        self._store.insert(
            collection=self.COLLECTION_NAME,
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
        )
        
        duration = time.perf_counter() - start_time
        
        self._index_stats = IndexStats(
            strategy=self.name,
            backend=self._store.backend_name,
            embedding_model=self._embedder.model_name,
            llm_model=None,
            duration_sec=duration,
            num_documents=len(documents),
            num_chunks=len(all_chunks),
            num_vectors=len(all_chunks),
            llm_calls=0,
            embed_calls=embed_calls,
        )
        
        self._is_indexed = True
        logger.info(
            f"Indexed {len(all_chunks)} vectors in {duration:.2f}s"
        )
        
        return self._index_stats
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        """
        Search for relevant chunks using embedding similarity.
        
        Process:
        1. Embed the query
        2. Find top-k most similar chunks
        3. Return results with scores
        """
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        if not self._is_indexed:
            logger.warning("No documents indexed yet")
            return SearchResponse(
                hits=[],
                stats=SearchStats(
                    duration_ms=0.0,
                    embed_calls=0,
                    llm_calls=0,
                    vectors_searched=0,
                ),
            )
        
        start_time = time.perf_counter()
        
        # Embed query
        query_embedding = self._embedder.embed(query)
        
        # Search vector store
        hits = self._store.search(
            collection=self.COLLECTION_NAME,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        vectors_searched = self._store.count(self.COLLECTION_NAME)
        
        return SearchResponse(
            hits=hits,
            stats=SearchStats(
                duration_ms=duration_ms,
                embed_calls=1,
                llm_calls=0,
                vectors_searched=vectors_searched,
            ),
        )
    
    def clear(self) -> None:
        """Clear the index and free resources."""
        if self._store is not None:
            self._store.delete_collection(self.COLLECTION_NAME)
            # Recreate empty collection
            if self._embedder is not None:
                self._store.create_collection(
                    self.COLLECTION_NAME,
                    dimension=self._embedder.dimension,
                )
        
        self._is_indexed = False
        self._index_stats = None
    
    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Chunk a document into overlapping segments.
        
        Strategy: Chunk by sections first, then split large sections.
        This preserves section boundaries when possible.
        """
        chunks: list[Chunk] = []
        
        for section in doc.sections:
            section_chunks = self._chunk_text(
                text=section.content,
                document_id=doc.id,
                section_id=section.id,
            )
            chunks.extend(section_chunks)
        
        # If no sections, chunk the full content
        if not chunks and doc.content:
            chunks = self._chunk_text(
                text=doc.content,
                document_id=doc.id,
                section_id=None,
            )
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        section_id: Optional[str],
    ) -> list[Chunk]:
        """
        Split text into overlapping chunks.
        
        Uses a simple character-based approach with overlap.
        Tries to break at sentence boundaries when possible.
        """
        if not text.strip():
            return []
        
        chunks: list[Chunk] = []
        text = text.strip()
        
        # Simple chunking with overlap
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundaries
                for delim in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_delim = text[start:end].rfind(delim)
                    if last_delim > self._chunk_size // 2:
                        end = start + last_delim + len(delim)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"{document_id}_{section_id or 'full'}_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        section_id=section_id,
                        content=chunk_text,
                        level=0,
                        metadata={
                            "chunk_idx": chunk_idx,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                chunk_idx += 1
            
            # Move forward with overlap
            start = end - self._chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
