"""Semantic retrieval strategy - pure embedding similarity."""

from typing import Optional

import numpy as np

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument


class SemanticRetrieval(RetrievalStrategy, EmbedderMixin):
    """Pure semantic embedding retrieval.
    
    Retrieves chunks by computing cosine similarity between
    query embedding and chunk embeddings.
    """

    def __init__(self, name: str = "semantic", **kwargs):
        super().__init__(name, **kwargs)
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks by computing embeddings."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks
        self.embeddings = self.encode_texts([c.content for c in chunks])

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks by cosine similarity."""
        if self.chunks is None or self.embeddings is None:
            return []

        q_emb = self.encode_query(query)
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in top_idx]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
        }
