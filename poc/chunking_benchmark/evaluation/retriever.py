"""Retriever for finding relevant chunks."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .embedder import Embedder


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    rank: int


@dataclass 
class IndexedChunk:
    """A chunk with its embedding and metadata."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: np.ndarray
    heading: Optional[str] = None
    heading_path: list[str] = None


class Retriever:
    """Simple cosine similarity retriever.
    
    Uses pre-computed embeddings for fast retrieval.
    """
    
    def __init__(self, embedder: Optional[Embedder] = None):
        """Initialize retriever.
        
        Args:
            embedder: Embedder to use. If None, creates a new one.
        """
        self.embedder = embedder or Embedder()
        self.chunks: list[IndexedChunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def index(self, chunks: list, show_progress: bool = True) -> None:
        """Index a list of chunks.
        
        Args:
            chunks: List of Chunk objects to index.
            show_progress: Whether to show progress bar.
        """
        if not chunks:
            self.chunks = []
            self.embeddings = None
            return
        
        # Embed all chunks
        embeddings = self.embedder.embed_chunks(chunks, show_progress=show_progress)
        
        # Store indexed chunks
        self.chunks = [
            IndexedChunk(
                chunk_id=c.id,
                doc_id=c.doc_id,
                content=c.content,
                embedding=embeddings[i],
                heading=c.heading,
                heading_path=c.heading_path or []
            )
            for i, c in enumerate(chunks)
        ]
        self.embeddings = embeddings
    
    def search(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Search for relevant chunks.
        
        Args:
            query: Query string.
            k: Number of results to return.
            
        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        if not self.chunks or self.embeddings is None:
            return []
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Compute cosine similarities (embeddings are already normalized)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        k = min(k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=float(similarities[idx]),
                rank=rank + 1
            ))
        
        return results
    
    def search_batch(self, queries: list[str], k: int = 10) -> list[list[RetrievalResult]]:
        """Search for multiple queries.
        
        Args:
            queries: List of query strings.
            k: Number of results per query.
            
        Returns:
            List of result lists, one per query.
        """
        return [self.search(q, k=k) for q in queries]
    
    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[IndexedChunk]:
        """Get a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_doc_id(self, doc_id: str) -> list[IndexedChunk]:
        """Get all chunks from a document."""
        return [c for c in self.chunks if c.doc_id == doc_id]
