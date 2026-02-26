"""Slow extraction system — hybrid NER pipeline.

Multi-stage LLM pipeline: Extract → Ground → Filter → Validate → Postprocess.
Achieves F1=0.932 on SO NER benchmark with retrieval-augmented few-shot prompting.
"""

from .hybrid_ner import (
    extract_candidate_verify,
    build_term_index,
    get_strategy_config,
    StrategyConfig,
    STRATEGY_PRESETS,
)
from .retrieval_ner import build_retrieval_index
from .scoring import normalize_term, verify_span

__all__ = [
    "extract_candidate_verify",
    "build_term_index",
    "build_retrieval_index",
    "get_strategy_config",
    "StrategyConfig",
    "STRATEGY_PRESETS",
    "normalize_term",
    "verify_span",
]
