from __future__ import annotations

import pytest

from plm.benchmark.tier import (
    QuoteMatch,
    ReciprocalMatch,
    TierAssignment,
    TierAssignmentInput,
    TierConfig,
    assign_tier,
    get_config,
)
from plm.benchmark.tier.config import (
    BronzeConfig,
    ConfidenceConfig,
    GoldConfig,
    SilverConfig,
)


def make_input(
    so_answer_id: int = 1,
    url_match: bool = True,
    fragment_anchor: str | None = None,
    fragment_matches_heading: bool = False,
    quote_matches: list[QuoteMatch] | None = None,
    reciprocal_matches: list[ReciprocalMatch] | None = None,
    upvotes: int = 5,
    is_accepted: bool = False,
    multiple_answers_same_url: int = 0,
) -> TierAssignmentInput:
    return TierAssignmentInput(
        so_answer_id=so_answer_id,
        url_match=url_match,
        fragment_anchor=fragment_anchor,
        fragment_matches_heading=fragment_matches_heading,
        quote_matches=quote_matches or [],
        reciprocal_matches=reciprocal_matches or [],
        upvotes=upvotes,
        is_accepted=is_accepted,
        multiple_answers_same_url=multiple_answers_same_url,
    )


class TestGoldTier:
    def test_fragment_anchor_match_returns_gold(self) -> None:
        input_data = make_input(
            fragment_anchor="pod-lifecycle",
            fragment_matches_heading=True,
        )
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.tier_reason == "fragment_anchor_match"
        assert result.primary_signal == "fragment_anchor_match"
        assert result.confidence_score == 1.0
        assert "fragment_anchor_match" in result.signals_detected
    
    def test_quote_match_30_chars_returns_gold(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods --all-namespaces",
            match_length=34,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote])
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.tier_reason == "quote_match_34_chars"
        assert result.confidence_score == 0.9
    
    def test_quote_match_over_50_chars_gets_bonus(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods --all-namespaces --output=wide --show-labels",
            match_length=61,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote])
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.confidence_score == 1.0
    
    def test_quote_match_under_30_chars_not_gold(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods",
            match_length=16,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote], upvotes=5)
        result = assign_tier(input_data)
        
        assert result.tier == "bronze"
    
    def test_reciprocal_containment_20_words_returns_gold(self) -> None:
        reciprocal = ReciprocalMatch(
            matched_words=["a"] * 22,
            word_count=22,
            chunk_id="chunk_1",
            direction="chunk_in_answer",
        )
        input_data = make_input(reciprocal_matches=[reciprocal])
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.tier_reason == "reciprocal_containment_22_words"
        assert result.confidence_score >= 0.85
    
    def test_reciprocal_containment_extra_words_increases_confidence(self) -> None:
        reciprocal = ReciprocalMatch(
            matched_words=["a"] * 40,
            word_count=40,
            chunk_id="chunk_1",
            direction="chunk_in_answer",
        )
        input_data = make_input(reciprocal_matches=[reciprocal])
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.confidence_score > 0.85
    
    def test_reciprocal_under_20_words_not_gold(self) -> None:
        reciprocal = ReciprocalMatch(
            matched_words=["a"] * 15,
            word_count=15,
            chunk_id="chunk_1",
            direction="chunk_in_answer",
        )
        input_data = make_input(reciprocal_matches=[reciprocal], upvotes=5)
        result = assign_tier(input_data)
        
        assert result.tier == "bronze"


class TestSilverTier:
    def test_high_upvotes_with_accepted_returns_silver(self) -> None:
        input_data = make_input(upvotes=10, is_accepted=True)
        result = assign_tier(input_data)
        
        assert result.tier == "silver"
        assert result.tier_reason == "url_high_trust"
        assert result.confidence_score >= 0.75
    
    def test_high_upvotes_alone_returns_silver(self) -> None:
        input_data = make_input(upvotes=25, is_accepted=False)
        result = assign_tier(input_data)
        
        assert result.tier == "silver"
        assert result.tier_reason == "url_high_trust"
    
    def test_both_upvotes_and_accepted_gets_bonus(self) -> None:
        input_data = make_input(upvotes=15, is_accepted=True)
        result = assign_tier(input_data)
        
        assert result.tier == "silver"
        assert result.confidence_score == 0.85
    
    def test_corroborated_url_returns_silver(self) -> None:
        input_data = make_input(upvotes=6, multiple_answers_same_url=3)
        result = assign_tier(input_data)
        
        assert result.tier == "silver"
        assert result.tier_reason == "url_corroborated"
        assert result.confidence_score >= 0.70
    
    def test_upvotes_9_not_accepted_falls_to_bronze(self) -> None:
        input_data = make_input(upvotes=9, is_accepted=False)
        result = assign_tier(input_data)
        
        assert result.tier == "bronze"


class TestBronzeTier:
    def test_url_match_with_5_upvotes_returns_bronze(self) -> None:
        input_data = make_input(upvotes=5)
        result = assign_tier(input_data)
        
        assert result.tier == "bronze"
        assert result.tier_reason == "url_community_validated"
        assert result.confidence_score == 0.60
    
    def test_more_upvotes_increases_confidence(self) -> None:
        input_data = make_input(upvotes=10)
        result = assign_tier(input_data)
        
        assert result.tier == "bronze"
        assert result.confidence_score > 0.60
    
    def test_bronze_confidence_capped_at_075(self) -> None:
        input_data = make_input(upvotes=100)
        result = assign_tier(input_data)
        
        assert result.confidence_score <= 0.75


class TestExcludeTier:
    def test_no_url_match_returns_exclude(self) -> None:
        input_data = make_input(url_match=False, upvotes=100)
        result = assign_tier(input_data)
        
        assert result.tier == "exclude"
        assert result.tier_reason == "insufficient_signal"
    
    def test_url_match_low_upvotes_returns_exclude(self) -> None:
        input_data = make_input(upvotes=4)
        result = assign_tier(input_data)
        
        assert result.tier == "exclude"
        assert result.tier_reason == "insufficient_signal"
    
    def test_exclude_has_zero_confidence(self) -> None:
        input_data = make_input(url_match=False)
        result = assign_tier(input_data)
        
        assert result.confidence_score == 0.0


class TestTierPriority:
    def test_gold_over_silver_signals(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods --all-namespaces",
            match_length=34,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(
            quote_matches=[quote],
            upvotes=30,
            is_accepted=True,
        )
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert "url_high_trust" in result.signals_detected
    
    def test_fragment_over_quote(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods --all-namespaces",
            match_length=34,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(
            fragment_anchor="pod-lifecycle",
            fragment_matches_heading=True,
            quote_matches=[quote],
        )
        result = assign_tier(input_data)
        
        assert result.tier == "gold"
        assert result.tier_reason == "fragment_anchor_match"


class TestEvidenceTrail:
    def test_evidence_includes_input_data(self) -> None:
        input_data = make_input(
            so_answer_id=12345,
            upvotes=15,
            is_accepted=True,
        )
        result = assign_tier(input_data)
        
        assert result.evidence["so_answer_id"] == 12345
        assert result.evidence["upvotes"] == 15
        assert result.evidence["is_accepted"] is True
    
    def test_evidence_includes_quote_matches(self) -> None:
        quote = QuoteMatch(
            matched_text="kubectl get pods",
            match_length=16,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote])
        result = assign_tier(input_data)
        
        assert "quote_matches" in result.evidence
        assert len(result.evidence["quote_matches"]) == 1
        assert result.evidence["quote_matches"][0]["chunk_id"] == "chunk_1"
    
    def test_evidence_is_json_serializable(self) -> None:
        import json
        
        quote = QuoteMatch(
            matched_text="test",
            match_length=4,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote])
        result = assign_tier(input_data)
        
        json_str = json.dumps(result.evidence)
        assert json_str is not None


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        input_data = make_input(upvotes=15, is_accepted=True)
        
        result1 = assign_tier(input_data)
        result2 = assign_tier(input_data)
        
        assert result1.tier == result2.tier
        assert result1.tier_reason == result2.tier_reason
        assert result1.confidence_score == result2.confidence_score
        assert result1.signals_detected == result2.signals_detected


class TestCustomConfig:
    def test_custom_gold_threshold(self) -> None:
        config = TierConfig(
            gold=GoldConfig(quote_min_length=50, reciprocal_min_words=30),
            silver=SilverConfig(),
            bronze=BronzeConfig(),
            confidence=ConfidenceConfig(),
        )
        
        quote = QuoteMatch(
            matched_text="kubectl get pods --all-namespaces",
            match_length=34,
            source_type="code",
            chunk_id="chunk_1",
            chunk_offset=0,
            answer_offset=0,
        )
        input_data = make_input(quote_matches=[quote], upvotes=5)
        result = assign_tier(input_data, config=config)
        
        assert result.tier == "bronze"
    
    def test_custom_bronze_threshold(self) -> None:
        config = TierConfig(
            gold=GoldConfig(),
            silver=SilverConfig(),
            bronze=BronzeConfig(min_upvotes=10),
            confidence=ConfidenceConfig(),
        )
        
        input_data = make_input(upvotes=5)
        result = assign_tier(input_data, config=config)
        
        assert result.tier == "exclude"
