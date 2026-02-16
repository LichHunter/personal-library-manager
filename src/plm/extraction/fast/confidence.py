"""Confidence scoring for routing decisions."""
from dataclasses import dataclass
from typing import Literal

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class ExtractionResult:
    """Result of extraction with confidence scoring."""

    term: str
    sources: list[str]
    confidence: float
    level: ConfidenceLevel


def compute_confidence(
    term: str,
    sources: list[str],
    entity_ratio: float | None = None,
) -> tuple[float, ConfidenceLevel]:
    """Compute confidence score and level for an extracted term.

    Args:
        term: The extracted term
        sources: List of extraction sources (e.g., ['camel_case', 'backtick'])
        entity_ratio: Optional entity ratio for additional scoring

    Returns:
        Tuple of (confidence_score, confidence_level)
    """
    source_score = min(len(sources) / 3, 1.0)
    ratio_score = entity_ratio if entity_ratio else 0.5
    confidence = (source_score * 0.6) + (ratio_score * 0.4)

    if confidence >= 0.8:
        return confidence, "HIGH"
    elif confidence >= 0.5:
        return confidence, "MEDIUM"
    else:
        return confidence, "LOW"
