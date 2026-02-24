from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from .config import TierConfig, get_config

log = logging.getLogger(__name__)

Tier = Literal["gold", "silver", "bronze", "exclude"]


@dataclass
class QuoteMatch:
    matched_text: str
    match_length: int
    source_type: Literal["code", "blockquote", "prose"]
    chunk_id: str
    chunk_offset: int
    answer_offset: int


@dataclass
class ReciprocalMatch:
    matched_words: list[str]
    word_count: int
    chunk_id: str
    direction: Literal["chunk_in_answer", "answer_in_chunk"]


@dataclass
class TierAssignmentInput:
    so_answer_id: int
    url_match: bool
    fragment_anchor: str | None
    fragment_matches_heading: bool
    quote_matches: list[QuoteMatch]
    reciprocal_matches: list[ReciprocalMatch]
    upvotes: int
    is_accepted: bool
    multiple_answers_same_url: int


@dataclass
class TierAssignment:
    tier: Tier
    tier_reason: str
    confidence_score: float
    signals_detected: list[str]
    primary_signal: str
    evidence: dict = field(default_factory=dict)


def _calculate_confidence(
    tier: Tier,
    primary_signal: str,
    input_data: TierAssignmentInput,
    config: TierConfig,
) -> float:
    conf = config.confidence
    
    if primary_signal == "fragment_anchor_match":
        return conf.fragment_anchor_match
    
    if primary_signal.startswith("quote_match_"):
        base = conf.quote_match
        max_length = max(
            (m.match_length for m in input_data.quote_matches),
            default=0,
        )
        if max_length > conf.quote_long_threshold:
            base += conf.quote_long_bonus
        return min(base, 1.0)
    
    if primary_signal.startswith("reciprocal_containment_"):
        base = conf.reciprocal_containment
        max_words = max(
            (m.word_count for m in input_data.reciprocal_matches),
            default=0,
        )
        extra_words = max_words - config.gold.reciprocal_min_words
        if extra_words > 0:
            steps = extra_words // conf.reciprocal_extra_words_step
            base += steps * conf.reciprocal_extra_words_bonus
        return min(base, 1.0)
    
    if primary_signal == "url_high_trust":
        base = conf.url_high_trust
        if input_data.upvotes >= config.silver.upvotes_with_accepted and input_data.is_accepted:
            base += conf.high_trust_both_bonus
        return min(base, 1.0)
    
    if primary_signal == "url_corroborated":
        base = conf.url_corroborated
        extra_corr = input_data.multiple_answers_same_url - config.silver.corroboration_count
        if extra_corr > 0:
            base += extra_corr * conf.corroboration_extra_bonus
        return min(base, 1.0)
    
    if primary_signal == "url_community_validated":
        base = conf.url_community_validated
        extra_upvotes = input_data.upvotes - config.bronze.min_upvotes
        if extra_upvotes > 0:
            base += extra_upvotes * conf.community_per_upvote_bonus
        return min(base, conf.community_max_confidence)
    
    return 0.0


def _collect_signals(input_data: TierAssignmentInput, config: TierConfig) -> list[str]:
    signals: list[str] = []
    
    if input_data.fragment_matches_heading:
        signals.append("fragment_anchor_match")
    
    for m in input_data.quote_matches:
        if m.match_length >= config.gold.quote_min_length:
            signals.append(f"quote_match_{m.match_length}_chars")
    
    for m in input_data.reciprocal_matches:
        if m.word_count >= config.gold.reciprocal_min_words:
            signals.append(f"reciprocal_containment_{m.word_count}_words")
    
    if input_data.url_match:
        if (input_data.upvotes >= config.silver.upvotes_with_accepted and input_data.is_accepted) or \
           input_data.upvotes >= config.silver.upvotes_alone:
            signals.append("url_high_trust")
        
        if input_data.multiple_answers_same_url >= config.silver.corroboration_count:
            signals.append("url_corroborated")
        
        if input_data.upvotes >= config.bronze.min_upvotes:
            signals.append("url_community_validated")
    
    return signals


def _build_evidence(input_data: TierAssignmentInput) -> dict:
    evidence: dict = {
        "so_answer_id": input_data.so_answer_id,
        "url_match": input_data.url_match,
        "fragment_anchor": input_data.fragment_anchor,
        "fragment_matches_heading": input_data.fragment_matches_heading,
        "upvotes": input_data.upvotes,
        "is_accepted": input_data.is_accepted,
        "multiple_answers_same_url": input_data.multiple_answers_same_url,
    }
    
    if input_data.quote_matches:
        evidence["quote_matches"] = [
            {
                "matched_text": m.matched_text,
                "match_length": m.match_length,
                "source_type": m.source_type,
                "chunk_id": m.chunk_id,
            }
            for m in input_data.quote_matches
        ]
    
    if input_data.reciprocal_matches:
        evidence["reciprocal_matches"] = [
            {
                "word_count": m.word_count,
                "chunk_id": m.chunk_id,
                "direction": m.direction,
            }
            for m in input_data.reciprocal_matches
        ]
    
    return evidence


def assign_tier(
    input_data: TierAssignmentInput,
    config: TierConfig | None = None,
) -> TierAssignment:
    if config is None:
        config = get_config()
    
    signals = _collect_signals(input_data, config)
    evidence = _build_evidence(input_data)
    
    if input_data.fragment_matches_heading:
        tier: Tier = "gold"
        reason = "fragment_anchor_match"
        log.debug(
            "GOLD tier assigned: fragment_anchor_match for answer %d",
            input_data.so_answer_id,
        )
    
    elif any(m.match_length >= config.gold.quote_min_length for m in input_data.quote_matches):
        tier = "gold"
        max_match = max(input_data.quote_matches, key=lambda m: m.match_length)
        reason = f"quote_match_{max_match.match_length}_chars"
        log.debug(
            "GOLD tier assigned: %s for answer %d",
            reason,
            input_data.so_answer_id,
        )
    
    elif any(m.word_count >= config.gold.reciprocal_min_words for m in input_data.reciprocal_matches):
        tier = "gold"
        max_match = max(input_data.reciprocal_matches, key=lambda m: m.word_count)
        reason = f"reciprocal_containment_{max_match.word_count}_words"
        log.debug(
            "GOLD tier assigned: %s for answer %d",
            reason,
            input_data.so_answer_id,
        )
    
    elif input_data.url_match and (
        (input_data.upvotes >= config.silver.upvotes_with_accepted and input_data.is_accepted) or
        input_data.upvotes >= config.silver.upvotes_alone
    ):
        tier = "silver"
        reason = "url_high_trust"
        log.debug(
            "SILVER tier assigned: url_high_trust for answer %d (upvotes=%d, accepted=%s)",
            input_data.so_answer_id,
            input_data.upvotes,
            input_data.is_accepted,
        )
    
    elif input_data.url_match and input_data.multiple_answers_same_url >= config.silver.corroboration_count:
        tier = "silver"
        reason = "url_corroborated"
        log.debug(
            "SILVER tier assigned: url_corroborated for answer %d (count=%d)",
            input_data.so_answer_id,
            input_data.multiple_answers_same_url,
        )
    
    elif input_data.url_match and input_data.upvotes >= config.bronze.min_upvotes:
        tier = "bronze"
        reason = "url_community_validated"
        log.debug(
            "BRONZE tier assigned: url_community_validated for answer %d (upvotes=%d)",
            input_data.so_answer_id,
            input_data.upvotes,
        )
    
    else:
        tier = "exclude"
        reason = "insufficient_signal"
        log.debug(
            "EXCLUDE: insufficient_signal for answer %d (url_match=%s, upvotes=%d)",
            input_data.so_answer_id,
            input_data.url_match,
            input_data.upvotes,
        )
    
    confidence = _calculate_confidence(tier, reason, input_data, config)
    
    return TierAssignment(
        tier=tier,
        tier_reason=reason,
        confidence_score=confidence,
        signals_detected=signals,
        primary_signal=reason,
        evidence=evidence,
    )
