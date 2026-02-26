"""Strategy configuration and presets for hybrid NER pipeline."""

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    name: str = "baseline"

    high_confidence_min_sources: int = 3
    validate_min_sources: int = 2

    use_term_retrieval_for_review: bool = False
    review_default_decision: str = "REJECT"

    contrastive_positive_snippets: int = 2
    contrastive_negative_snippets: int = 2
    contrastive_show_reasoning: bool = False
    safety_net_ratio: float = 0.8

    reject_bare_version_numbers: bool = False
    reject_bare_numbers_with_dot: bool = False
    smart_version_filter: bool = False

    boost_common_word_seeds: bool = False
    common_word_seed_list: list[str] = field(default_factory=list)

    ratio_gated_review: bool = False
    ratio_auto_approve_threshold: float = 0.70
    ratio_auto_reject_threshold: float = 0.20

    seed_bypass_to_high_confidence: bool = False
    seed_bypass_require_context: bool = False
    seed_bypass_min_sources_for_auto: int = 2
    suppress_path_embedded: bool = False

    validate_high_confidence_too: bool = False

    use_haiku_extraction: bool = False
    use_heuristic_extraction: bool = False
    log_low_confidence: bool = False
    high_entity_ratio_threshold: float = 0.8
    medium_entity_ratio_threshold: float = 0.5
    skip_validation_entity_ratio: float = 0.7

    use_contextual_seeds: bool = False
    use_low_precision_filter: bool = False
    allcaps_require_corroboration: bool = False
    use_sonnet_taxonomy: bool = False

    route_single_vote_to_validation: bool = False
    single_vote_min_entity_ratio: float = 0.0

    structural_require_llm_vote: bool = False
    disable_seed_bypass: bool = False

    use_candidate_verify: bool = False


STRATEGY_PRESETS: dict[str, StrategyConfig] = {
    "baseline": StrategyConfig(name="baseline"),

    "strategy_candidate_verify": StrategyConfig(
        name="strategy_candidate_verify",
        use_candidate_verify=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.95,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.90,
        safety_net_ratio=0.95,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=True,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
        structural_require_llm_vote=True,
        disable_seed_bypass=False,
    ),
}


def get_strategy_config(name: str) -> StrategyConfig:
    if name not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_PRESETS.keys())}")
    return STRATEGY_PRESETS[name]
