"""Slow extraction pipeline modules.

V6 strategy implementation from POC-1c achieving F1=0.932 @ 10 docs.
Multi-stage pipeline: extraction -> grounding -> noise filter -> validation -> postprocess.

Modules:
    grounding: Stage 2 - Span verification and deduplication
    noise_filter: Stage 3 - Stop words, generic phrases, negatives
    candidate_verify: Stage 1 part - Heuristic extraction + LLM verify
    taxonomy: Stage 1 part - Taxonomy-based LLM extraction
    validation: Stage 4 - Context validation for ambiguous terms
    postprocess: Stage 5 - Span expansion, subspan suppression, final dedup
"""

# Stage 2: Grounding
from .grounding import (
    verify_span,
    normalize_term,
    ground_candidates,
    deduplicate,
)

# Stage 3: Noise filtering
from .noise_filter import (
    filter_noise,
    load_negatives,
    is_stop_word,
    is_generic_phrase,
    PURE_STOP_WORDS,
    ACTION_GERUNDS,
    DESCRIPTIVE_ADJECTIVES,
    CATEGORY_SUFFIXES,
)

# Stage 1 part: Candidate verify
from .candidate_verify import (
    extract_candidates_heuristic,
    classify_candidates_llm,
)

# Stage 1 part: Taxonomy extraction
from .taxonomy import (
    extract_by_taxonomy,
    ENTITY_TYPES,
)

# Stage 4: Validation
from .validation import (
    validate_terms,
    build_validation_prompt,
)

# Stage 5: Post-processing
from .postprocess import (
    expand_spans,
    suppress_subspans,
    final_dedup,
    is_embedded_in_path,
    filter_path_embedded,
    filter_urls,
)


__all__ = [
    # Grounding
    "verify_span",
    "normalize_term",
    "ground_candidates",
    "deduplicate",
    # Noise filter
    "filter_noise",
    "load_negatives",
    "is_stop_word",
    "is_generic_phrase",
    "PURE_STOP_WORDS",
    "ACTION_GERUNDS",
    "DESCRIPTIVE_ADJECTIVES",
    "CATEGORY_SUFFIXES",
    # Candidate verify
    "extract_candidates_heuristic",
    "classify_candidates_llm",
    # Taxonomy
    "extract_by_taxonomy",
    "ENTITY_TYPES",
    # Validation
    "validate_terms",
    "build_validation_prompt",
    # Postprocess
    "expand_spans",
    "suppress_subspans",
    "final_dedup",
    "is_embedded_in_path",
    "filter_path_embedded",
    "filter_urls",
]
