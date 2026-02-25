"""Retrieval strategies for the benchmark framework."""

from typing import Optional

from ..config.schema import StrategyConfig
from ..core.protocols import Embedder, VectorStore, LLM, RetrievalStrategy

from .flat import FlatStrategy
from .raptor import RaptorStrategy
from .lod import LODEmbedStrategy, LODLLMStrategy


def create_strategy(
    config: StrategyConfig,
    embedder: Embedder,
    store: VectorStore,
    llm: Optional[LLM] = None,
) -> RetrievalStrategy:
    """Factory function to create a retrieval strategy from configuration."""
    if config.name == "flat":
        strategy = FlatStrategy(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    elif config.name == "raptor":
        strategy = RaptorStrategy(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_layers=config.max_layers,
            cluster_threshold=config.cluster_threshold,
        )
    elif config.name == "lod_embed":
        strategy = LODEmbedStrategy(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            doc_top_k=config.doc_top_k,
            section_top_k=config.section_top_k,
        )
    elif config.name == "lod_llm":
        strategy = LODLLMStrategy(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            doc_top_k=config.doc_top_k,
            section_top_k=config.section_top_k,
        )
    else:
        raise ValueError(
            f"Unknown strategy: {config.name}. "
            f"Available: flat, raptor, lod_embed, lod_llm"
        )
    
    strategy.configure(embedder, store, llm)
    return strategy


__all__ = [
    "create_strategy",
    "FlatStrategy",
    "RaptorStrategy",
    "LODEmbedStrategy",
    "LODLLMStrategy",
]
