"""Main extraction pipeline orchestrator."""
from dataclasses import dataclass, field
from typing import Literal

from .fast import extract_all_heuristic, compute_confidence
from .slow import (
    verify_span,
    normalize_term,
    filter_noise,
    validate_terms,
    expand_spans,
    suppress_subspans,
    final_dedup,
)


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""

    use_fast_only: bool = False
    use_slow_only: bool = False
    confidence_threshold: float = 0.7
    validate_medium_confidence: bool = True


@dataclass
class ExtractionResult:
    """Result of extraction pipeline."""

    terms: list[str]
    fast_candidates: list[str] = field(default_factory=list)
    slow_candidates: list[str] = field(default_factory=list)
    filtered_count: int = 0
    validated_count: int = 0


def extract(
    text: str,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """
    Extract technical entities from text.

    Args:
        text: Document text to extract from
        config: Pipeline configuration

    Returns:
        ExtractionResult with extracted terms and stats
    """
    config = config or ExtractionConfig()
    result = ExtractionResult(terms=[])

    # Stage 1: Fast extraction (heuristics)
    if not config.use_slow_only:
        fast_candidates = extract_all_heuristic(text)
        result.fast_candidates = fast_candidates

    # Stage 2: Slow extraction (LLM-based) - TODO: implement full V6
    if not config.use_fast_only:
        # Placeholder for full V6 pipeline
        slow_candidates = []  # Will call taxonomy, candidate_verify, etc.
        result.slow_candidates = slow_candidates

    # Merge candidates
    all_candidates = list(set(result.fast_candidates + result.slow_candidates))

    # Stage 3: Ground and filter
    grounded = [(c, verify_span(c, text)) for c in all_candidates]
    valid = [c for c, (valid, _) in grounded if valid]

    # Stage 4: Noise filter
    filtered = filter_noise(valid)
    result.filtered_count = len(valid) - len(filtered)

    # Stage 5: Postprocess
    expanded = expand_spans(filtered, text)
    suppressed = suppress_subspans(expanded)
    final = final_dedup(suppressed)
    result.terms = final

    return result


# Convenience functions
def fast_extract(text: str) -> list[str]:
    """Fast-only extraction (zero LLM cost)."""
    return extract(text, ExtractionConfig(use_fast_only=True)).terms


def slow_extract(text: str) -> list[str]:
    """Slow-only extraction (full V6 pipeline)."""
    return extract(text, ExtractionConfig(use_slow_only=True)).terms
