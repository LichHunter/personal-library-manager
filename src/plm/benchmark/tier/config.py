"""
Tier assignment configuration with thresholds.

These thresholds determine how evidence signals map to trust tiers.
All values can be overridden via environment variables or config file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldConfig:
    """GOLD tier requires provable signals: fragment match, quote, or containment."""
    quote_min_length: int = 30  # Minimum chars for exact quote match
    reciprocal_min_words: int = 20  # Minimum words for reciprocal containment


@dataclass(frozen=True)
class SilverConfig:
    """SILVER tier: URL match with high trust signals."""
    upvotes_with_accepted: int = 10  # Minimum upvotes when answer is accepted
    upvotes_alone: int = 25  # Minimum upvotes without accepted flag
    corroboration_count: int = 2  # Minimum corroborating answers linking same URL


@dataclass(frozen=True)
class BronzeConfig:
    """BRONZE tier: URL match with community validation."""
    min_upvotes: int = 5  # Minimum answer score


@dataclass(frozen=True)
class ConfidenceConfig:
    """Base confidence scores and modifiers for each signal type."""
    # Base scores
    fragment_anchor_match: float = 1.0
    quote_match: float = 0.9
    reciprocal_containment: float = 0.85
    url_high_trust: float = 0.75
    url_corroborated: float = 0.70
    url_community_validated: float = 0.60
    
    # Modifiers
    quote_long_bonus: float = 0.1  # Added if quote length > 50
    quote_long_threshold: int = 50  # Chars threshold for long quote bonus
    reciprocal_extra_words_bonus: float = 0.05  # Per 10 extra words
    reciprocal_extra_words_step: int = 10  # Words per bonus step
    high_trust_both_bonus: float = 0.1  # If both upvotes >= 10 AND accepted
    corroboration_extra_bonus: float = 0.05  # Per extra corroborating answer
    community_per_upvote_bonus: float = 0.02  # Per upvote above 5
    community_max_confidence: float = 0.75  # Cap for community validated


@dataclass(frozen=True)
class TierConfig:
    """Complete tier assignment configuration."""
    gold: GoldConfig
    silver: SilverConfig
    bronze: BronzeConfig
    confidence: ConfidenceConfig


# Default configuration instance
DEFAULT_CONFIG = TierConfig(
    gold=GoldConfig(),
    silver=SilverConfig(),
    bronze=BronzeConfig(),
    confidence=ConfidenceConfig(),
)


def get_config() -> TierConfig:
    """
    Get tier configuration.
    
    Currently returns default config. Can be extended to load from
    environment variables or config file.
    """
    return DEFAULT_CONFIG
