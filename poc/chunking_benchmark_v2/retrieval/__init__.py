"""Retrieval strategy implementations.

Registry pattern for easy configuration-driven instantiation.
"""

from .base import (
    RetrievalStrategy,
    EmbedderMixin,
    RerankerMixin,
    Section,
    StructuredDocument,
    parse_document_sections,
)
from .semantic import SemanticRetrieval
from .hybrid import HybridRetrieval
from .hybrid_rerank import HybridRerankRetrieval
from .lod import LODRetrieval
from .lod_llm import LODLLMRetrieval
from .raptor import RAPTORRetrieval


# Registry mapping strategy names to classes
RETRIEVAL_STRATEGIES: dict[str, type[RetrievalStrategy]] = {
    "semantic": SemanticRetrieval,
    "hybrid": HybridRetrieval,
    "hybrid_rerank": HybridRerankRetrieval,
    "lod": LODRetrieval,
    "lod_llm": LODLLMRetrieval,
    "raptor": RAPTORRetrieval,
}


def create_retrieval_strategy(
    strategy_type: str,
    name: str | None = None,
    **kwargs
) -> RetrievalStrategy:
    """Create a retrieval strategy by type name.
    
    Args:
        strategy_type: One of "semantic", "hybrid", "hybrid_rerank", 
                      "lod", "lod_llm", "raptor"
        name: Optional custom name for the strategy instance
        **kwargs: Strategy-specific parameters
        
    Returns:
        Configured RetrievalStrategy instance
        
    Raises:
        ValueError: If strategy_type is unknown
    """
    if strategy_type not in RETRIEVAL_STRATEGIES:
        valid = ", ".join(RETRIEVAL_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy type: {strategy_type}. Valid: {valid}")
    
    strategy_class = RETRIEVAL_STRATEGIES[strategy_type]
    instance_name = name or strategy_type
    return strategy_class(name=instance_name, **kwargs)


def list_retrieval_strategies() -> list[str]:
    """List available retrieval strategy types."""
    return list(RETRIEVAL_STRATEGIES.keys())


__all__ = [
    # Base classes
    "RetrievalStrategy",
    "EmbedderMixin", 
    "RerankerMixin",
    "Section",
    "StructuredDocument",
    "parse_document_sections",
    # Strategy classes
    "SemanticRetrieval",
    "HybridRetrieval",
    "HybridRerankRetrieval",
    "LODRetrieval",
    "LODLLMRetrieval",
    "RAPTORRetrieval",
    # Registry functions
    "RETRIEVAL_STRATEGIES",
    "create_retrieval_strategy",
    "list_retrieval_strategies",
]
