"""SentenceTransformers embedding implementation."""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from ..core.protocols import Embedder

logger = logging.getLogger(__name__)


class SBERTEmbedder(Embedder):
    """Embedder using SentenceTransformers library."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to use (None = auto-detect)
        """
        self._model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()
