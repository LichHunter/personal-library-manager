"""Deterministic verification module for benchmark cases.

Verifies generated cases against corpus without LLM.
Checks: chunk_exists, quote_exists, quote_length, tier_match, query_length.
"""

from .verifier import (
    VerificationFailure,
    VerificationResult,
    verify,
)

__all__ = [
    "VerificationFailure",
    "VerificationResult",
    "verify",
]
