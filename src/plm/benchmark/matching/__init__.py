"""Matching utilities for benchmark framework."""

from .fragment_matcher import (
    MIN_RECIPROCAL_WORDS,
    ReciprocalMatch,
    extract_words,
    find_reciprocal_matches,
    normalize_anchor,
)
from .quote_matcher import (
    MIN_QUOTE_LENGTH,
    GENERIC_BLACKLIST,
    QuoteMatch,
    extract_text_from_html,
    find_quote_matches,
    is_generic_text,
    normalize_text,
)

__all__ = [
    "QuoteMatch",
    "ReciprocalMatch",
    "MIN_QUOTE_LENGTH",
    "MIN_RECIPROCAL_WORDS",
    "GENERIC_BLACKLIST",
    "normalize_text",
    "extract_text_from_html",
    "find_quote_matches",
    "is_generic_text",
    "normalize_anchor",
    "extract_words",
    "find_reciprocal_matches",
]
