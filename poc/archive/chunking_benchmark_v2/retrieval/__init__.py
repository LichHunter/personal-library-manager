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
from .hyde import HyDERetrieval
from .multi_query import MultiQueryRetrieval
from .reverse_hyde import ReverseHyDERetrieval
from .enriched_lod import EnrichedLODRetrieval
from .enriched_hybrid import EnrichedHybridRetrieval
from .enriched_hybrid_bmx import EnrichedHybridBMXRetrieval
from .bmx_semantic import BMXSemanticRetrieval
from .bmx_pure import BMXPureRetrieval
from .enriched_hybrid_llm import EnrichedHybridLLMRetrieval
from .bmx_wqa import BMXWQARetrieval

RETRIEVAL_STRATEGIES: dict[str, type[RetrievalStrategy]] = {
    "semantic": SemanticRetrieval,
    "hybrid": HybridRetrieval,
    "hybrid_rerank": HybridRerankRetrieval,
    "lod": LODRetrieval,
    "lod_llm": LODLLMRetrieval,
    "raptor": RAPTORRetrieval,
    "hyde": HyDERetrieval,
    "multi_query": MultiQueryRetrieval,
    "reverse_hyde": ReverseHyDERetrieval,
    "enriched_lod": EnrichedLODRetrieval,
    "enriched_lod_keywords": EnrichedLODRetrieval,
    "enriched_lod_contextual": EnrichedLODRetrieval,
    "enriched_lod_questions": EnrichedLODRetrieval,
    "enriched_lod_summary": EnrichedLODRetrieval,
    "enriched_lod_entities": EnrichedLODRetrieval,
    "enriched_hybrid": EnrichedHybridRetrieval,
    "enriched_hybrid_bmx": EnrichedHybridBMXRetrieval,
    "bmx_semantic": BMXSemanticRetrieval,
    "bmx_pure": BMXPureRetrieval,
    "enriched_hybrid_llm": EnrichedHybridLLMRetrieval,
    "bmx_wqa": BMXWQARetrieval,
}

COLBERT_AVAILABLE = False
try:
    from .colbert import ColBERTRetrieval as _ColBERTRetrieval
    from .colbert import ColBERTReranker as _ColBERTReranker

    RETRIEVAL_STRATEGIES["colbert"] = _ColBERTRetrieval
    RETRIEVAL_STRATEGIES["colbert_rerank"] = _ColBERTReranker
    COLBERT_AVAILABLE = True
except ImportError:
    pass


def create_retrieval_strategy(
    strategy_type: str, name: str | None = None, **kwargs
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
    "RetrievalStrategy",
    "EmbedderMixin",
    "RerankerMixin",
    "Section",
    "StructuredDocument",
    "parse_document_sections",
    "SemanticRetrieval",
    "HybridRetrieval",
    "HybridRerankRetrieval",
    "LODRetrieval",
    "LODLLMRetrieval",
    "RAPTORRetrieval",
    "HyDERetrieval",
    "MultiQueryRetrieval",
    "ReverseHyDERetrieval",
    "EnrichedLODRetrieval",
    "EnrichedHybridRetrieval",
    "EnrichedHybridBMXRetrieval",
    "BMXSemanticRetrieval",
    "BMXPureRetrieval",
    "EnrichedHybridLLMRetrieval",
    "BMXWQARetrieval",
    "RETRIEVAL_STRATEGIES",
    "create_retrieval_strategy",
    "list_retrieval_strategies",
    "COLBERT_AVAILABLE",
]
