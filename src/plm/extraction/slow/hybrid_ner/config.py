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

    # v5: Haiku+heuristic extraction mode
    use_haiku_extraction: bool = False
    use_heuristic_extraction: bool = False
    log_low_confidence: bool = False
    high_entity_ratio_threshold: float = 0.8
    medium_entity_ratio_threshold: float = 0.5
    skip_validation_entity_ratio: float = 0.7

    # v5.1: Optimizations
    use_contextual_seeds: bool = False
    use_low_precision_filter: bool = False
    allcaps_require_corroboration: bool = False
    use_sonnet_taxonomy: bool = False

    # v5.2: entity_ratio as signal not gate
    route_single_vote_to_validation: bool = False
    # v5.3: minimum entity_ratio for single-vote routing to validation
    single_vote_min_entity_ratio: float = 0.0

    # v5.4: tighten validation routing
    structural_require_llm_vote: bool = False
    disable_seed_bypass: bool = False

    # v6: candidate_verify extraction mode
    use_candidate_verify: bool = False


STRATEGY_PRESETS: dict[str, StrategyConfig] = {
    "baseline": StrategyConfig(name="baseline"),

    "strategy_a": StrategyConfig(
        name="strategy_a",
        high_confidence_min_sources=2,
        validate_min_sources=1,
    ),

    "strategy_b": StrategyConfig(
        name="strategy_b",
        use_term_retrieval_for_review=True,
    ),

    "strategy_c": StrategyConfig(
        name="strategy_c",
        contrastive_positive_snippets=3,
        contrastive_negative_snippets=3,
        contrastive_show_reasoning=True,
    ),

    "strategy_d": StrategyConfig(
        name="strategy_d",
        reject_bare_version_numbers=True,
        reject_bare_numbers_with_dot=True,
    ),

    "strategy_e": StrategyConfig(
        name="strategy_e",
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
    ),

    "strategy_v4": StrategyConfig(
        name="strategy_v4",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.20,
    ),

    "strategy_v4_1": StrategyConfig(
        name="strategy_v4_1",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
    ),

    "strategy_v4_2": StrategyConfig(
        name="strategy_v4_2",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        seed_bypass_to_high_confidence=True,
        suppress_path_embedded=True,
    ),

    "strategy_v4_3": StrategyConfig(
        name="strategy_v4_3",
        smart_version_filter=True,
        boost_common_word_seeds=True,
        common_word_seed_list=[
            "image", "form", "page", "phone", "keyboard", "screen",
            "button", "table", "column", "row", "list", "array",
            "string", "boolean", "float", "integer", "exception",
            "server", "client", "browser", "console", "container",
            "padding", "key", "keys", "cursor", "log", "request",
            "calculator", "global", "session", "camera", "pad",
            "ruby", "symlinks",
        ],
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
        suppress_path_embedded=True,
    ),

    "strategy_v5": StrategyConfig(
        name="strategy_v5",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=2,
    ),

    "strategy_v5_1": StrategyConfig(
        name="strategy_v5_1",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=True,
        allcaps_require_corroboration=True,
        use_sonnet_taxonomy=True,
    ),

    "strategy_v5_2": StrategyConfig(
        name="strategy_v5_2",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
    ),

    "strategy_v5_3": StrategyConfig(
        name="strategy_v5_3",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
    ),

    "strategy_v5_4": StrategyConfig(
        name="strategy_v5_4",
        use_haiku_extraction=True,
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

    "strategy_v6": StrategyConfig(
        name="strategy_v6",
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

    "strategy_v5_3b": StrategyConfig(
        name="strategy_v5_3b",
        use_haiku_extraction=True,
        use_heuristic_extraction=True,
        smart_version_filter=True,
        ratio_gated_review=True,
        ratio_auto_approve_threshold=0.70,
        ratio_auto_reject_threshold=0.30,
        suppress_path_embedded=True,
        log_low_confidence=True,
        high_entity_ratio_threshold=0.8,
        medium_entity_ratio_threshold=0.5,
        skip_validation_entity_ratio=0.7,
        seed_bypass_to_high_confidence=True,
        seed_bypass_require_context=True,
        seed_bypass_min_sources_for_auto=1,
        use_contextual_seeds=True,
        use_low_precision_filter=False,
        allcaps_require_corroboration=False,
        use_sonnet_taxonomy=True,
        route_single_vote_to_validation=True,
        single_vote_min_entity_ratio=0.01,
    ),
}


def get_strategy_config(name: str) -> StrategyConfig:
    if name not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_PRESETS.keys())}")
    return STRATEGY_PRESETS[name]
