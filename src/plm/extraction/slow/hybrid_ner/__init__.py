"""Candidate-verify NER extraction pipeline.

Multi-stage pipeline: Extract → Ground → Filter → Validate → Postprocess.
Achieves 91-93% P/R on SO NER benchmark.
"""

from .config import StrategyConfig, STRATEGY_PRESETS, get_strategy_config
from .parsing import _parse_terms_json
from .validation import build_term_index
from .pipeline import extract_candidate_verify

__all__ = [
    "StrategyConfig",
    "STRATEGY_PRESETS",
    "get_strategy_config",
    "_parse_terms_json",
    "build_term_index",
    "extract_candidate_verify",
]
