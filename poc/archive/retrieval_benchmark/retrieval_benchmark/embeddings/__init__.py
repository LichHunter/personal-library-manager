"""Embedding model implementations."""

from ..config.schema import EmbeddingConfig
from ..core.protocols import Embedder
from .sbert import SBERTEmbedder


def create_embedder(config: EmbeddingConfig) -> Embedder:
    """Factory function to create an embedder from config."""
    if config.provider == "sentence_transformers":
        return SBERTEmbedder(model_name=config.model)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")


__all__ = ["create_embedder", "SBERTEmbedder"]
