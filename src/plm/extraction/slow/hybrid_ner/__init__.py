"""Hybrid multi-stage NER extraction pipeline.

Combines retrieval few-shot, exhaustive extraction, haiku extraction,
and auto-vocabulary seeding for high recall, then applies progressive
filtering (voting, noise filter, context validation) for high precision.

Target: 95% precision, 95% recall, <5% hallucination — with ZERO manual vocabulary.

Architecture:
  Stage 1: High-recall candidate generation (3 LLM extractors + auto-seeds)
  Stage 2: Grounding (span verification) + dedup
  Stage 3: Confidence scoring + tiered filtering (voting, noise, negatives)
  Stage 4: Context validation for ambiguous common words
  Stage 5: Final assembly
"""

from .config import StrategyConfig, STRATEGY_PRESETS, get_strategy_config
from .parsing import _parse_terms_json
from .validation import build_term_index
from .pipeline import extract_hybrid, extract_hybrid_v5, clear_low_confidence_stats

__all__ = [
    "StrategyConfig",
    "STRATEGY_PRESETS",
    "get_strategy_config",
    "_parse_terms_json",
    "build_term_index",
    "extract_hybrid",
    "extract_hybrid_v5",
    "clear_low_confidence_stats",
]
