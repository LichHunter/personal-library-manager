"""Embedding service for chunks and queries."""

import numpy as np
from typing import Optional
from dataclasses import dataclass

# Lazy import to avoid loading model at module import time
_model = None


def _get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: np.ndarray
    texts: list[str]
    model_name: str = "all-MiniLM-L6-v2"


class Embedder:
    """Embed text using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 (384 dimensions) for fast, local embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder.
        
        Args:
            model_name: Name of sentence-transformers model to use.
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts.
        
        Args:
            texts: List of strings to embed.
            show_progress: Whether to show progress bar.
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.
        
        Args:
            query: Query string to embed.
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed([query])[0]
    
    def embed_chunks(self, chunks: list, show_progress: bool = True) -> np.ndarray:
        """Embed a list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects (must have .content attribute).
            show_progress: Whether to show progress bar.
            
        Returns:
            numpy array of shape (len(chunks), embedding_dim)
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed(texts, show_progress=show_progress)
