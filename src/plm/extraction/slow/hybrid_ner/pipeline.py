"""Main pipeline orchestrators: extract_hybrid (v1-v4) and extract_hybrid_v5 (v5-v6)."""

import json
import re
import time
from pathlib import Path

from ..scoring import normalize_term
from .config import StrategyConfig
from .extractors import (
    _extract_exhaustive_sonnet,
    _extract_haiku_simple,
    _extract_retrieval_fixed,
    _extract_seeds,
    _extract_heuristic,
    _extract_haiku_fewshot,
    _extract_haiku_taxonomy,
)
from .grounding import _ground_and_dedup
from .noise_filter import _auto_keep_structural, _auto_reject_noise, _run_sonnet_review
from .validation import (
    _run_context_validation,
    _run_term_retrieval_validation,
    _run_term_retrieval_review,
    _has_technical_context,
)
from .postprocess import _expand_spans, _is_embedded_in_path, _suppress_subspans


# ============================================================================
# STAGE 5: MAIN PIPELINE (v1-v4)
# ============================================================================

def extract_hybrid(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    auto_vocab: dict,
    term_index: dict[str, dict] | None = None,
    strategy: StrategyConfig | None = None,
) -> list[str]:
    cfg = strategy or StrategyConfig()

    bypass_set = set(t.lower() for t in auto_vocab.get("bypass", []))
    seeds_list = list(auto_vocab.get("seeds", []))
    negatives_set = set(t.lower() for t in auto_vocab.get("negatives", []))

    if cfg.boost_common_word_seeds and cfg.common_word_seed_list:
        seeds_list = seeds_list + [s for s in cfg.common_word_seed_list if s not in seeds_list]
        bypass_set.update(s.lower() for s in cfg.common_word_seed_list)

    doc_text = doc["text"]

    retrieval_terms, _ = _extract_retrieval_fixed(doc, train_docs, index, model)
    exhaustive_terms, _ = _extract_exhaustive_sonnet(doc)
    haiku_terms, _ = _extract_haiku_simple(doc)
    seed_terms = _extract_seeds(doc, seeds_list)

    candidates_by_source = {
        "retrieval": retrieval_terms,
        "exhaustive": exhaustive_terms,
        "haiku": haiku_terms,
        "seeds": seed_terms,
    }

    grounded = _ground_and_dedup(candidates_by_source, doc_text)

    after_noise: dict[str, dict] = {}
    for key, cand in grounded.items():
        term = cand["term"]
        if _auto_reject_noise(term, negatives_set, bypass_set, strategy=cfg, doc_text=doc_text):
            continue
        after_noise[key] = cand

    high_confidence: list[str] = []
    needs_validation: list[str] = []
    needs_review: list[str] = []

    protected_seed_set: set[str] = set()
    if cfg.seed_bypass_to_high_confidence and cfg.common_word_seed_list:
        protected_seed_set = {s.lower() for s in cfg.common_word_seed_list}

    for key, cand in after_noise.items():
        term = cand["term"]
        source_count = cand["source_count"]

        if _auto_keep_structural(term):
            high_confidence.append(term)
            continue

        if cfg.seed_bypass_to_high_confidence and term.lower() in protected_seed_set and "seeds" in cand["sources"]:
            if not cfg.seed_bypass_require_context:
                high_confidence.append(term)
                continue
            if source_count >= cfg.seed_bypass_min_sources_for_auto:
                high_confidence.append(term)
                continue
            if _has_technical_context(term, doc_text):
                high_confidence.append(term)
                continue

        if source_count >= cfg.high_confidence_min_sources:
            high_confidence.append(term)
            continue

        if source_count >= cfg.validate_min_sources:
            needs_validation.append(term)
            continue

        needs_review.append(term)

    if needs_review:
        if cfg.ratio_gated_review and term_index:
            protected_terms = set(bypass_set)
            if cfg.boost_common_word_seeds and cfg.common_word_seed_list:
                protected_terms.update(s.lower() for s in cfg.common_word_seed_list)

            auto_approved: list[str] = []
            uncertain: list[str] = []
            for term in needs_review:
                info = term_index.get(term.lower())
                is_protected = term.lower() in protected_terms

                if is_protected:
                    uncertain.append(term)
                elif info:
                    ratio = info.get("entity_ratio", 0.5)
                    if ratio > cfg.ratio_auto_approve_threshold:
                        auto_approved.append(term)
                    elif ratio < cfg.ratio_auto_reject_threshold:
                        pass
                    else:
                        uncertain.append(term)
                else:
                    uncertain.append(term)

            review_decisions = _run_sonnet_review(uncertain, doc_text) if uncertain else {}
            for term in uncertain:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)
            needs_validation.extend(auto_approved)

        elif cfg.use_term_retrieval_for_review and term_index:
            review_decisions = _run_term_retrieval_review(
                needs_review, doc_text, term_index, strategy=cfg,
            )
            for term in needs_review:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)
        else:
            review_decisions = _run_sonnet_review(needs_review, doc_text)
            for term in needs_review:
                decision = review_decisions.get(term, cfg.review_default_decision)
                if decision == "APPROVE":
                    needs_validation.append(term)

    if cfg.validate_high_confidence_too and term_index:
        all_to_validate = high_confidence + needs_validation
        validated = _run_term_retrieval_validation(
            all_to_validate, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        )
    elif term_index:
        validated_subset = _run_term_retrieval_validation(
            needs_validation, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        )
        validated = high_confidence + validated_subset
    else:
        validated_subset = _run_context_validation(
            needs_validation, doc_text, bypass_set,
        )
        validated = high_confidence + validated_subset

    expanded = _expand_spans(validated, doc_text)

    if cfg.suppress_path_embedded:
        expanded = [t for t in expanded if not _is_embedded_in_path(t, doc_text)]

    suppressed = _suppress_subspans(expanded, protected_seeds=protected_seed_set)

    seen: set[str] = set()
    final: list[str] = []
    for term in suppressed:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            final.append(term)

    return final


# ============================================================================
# LOW CONFIDENCE STATISTICS LOGGING
# ============================================================================

_LOW_CONF_STATS_PATH = Path(__file__).parent.parent / "artifacts" / "results" / "low_confidence_stats.jsonl"


def _log_low_confidence_stats(
    term: str,
    doc_id: str,
    vote_count: int,
    entity_ratio: float,
    sources: list[str],
    in_gt: bool,
) -> None:
    _LOW_CONF_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "term": term,
        "doc_id": doc_id,
        "vote_count": vote_count,
        "entity_ratio": round(entity_ratio, 4),
        "sources": sources,
        "in_gt": in_gt,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(_LOW_CONF_STATS_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def clear_low_confidence_stats() -> None:
    if _LOW_CONF_STATS_PATH.exists():
        _LOW_CONF_STATS_PATH.unlink()


# ============================================================================
# STAGE 5b: V5 PIPELINE (3 Haiku + Heuristic + Sonnet validation)
# ============================================================================

def extract_hybrid_v5(
    doc: dict,
    train_docs: list[dict],
    index,  # faiss.Index
    model,  # SentenceTransformer
    auto_vocab: dict,
    term_index: dict[str, dict] | None = None,
    strategy: StrategyConfig | None = None,
) -> list[str]:
    cfg = strategy or StrategyConfig()

    bypass_set = set(t.lower() for t in auto_vocab.get("bypass", []))
    seeds_list = list(auto_vocab.get("seeds", []))
    negatives_set = set(t.lower() for t in auto_vocab.get("negatives", []))
    contextual_seeds_set = set(
        t.lower() for t in auto_vocab.get("contextual_seeds", [])
    ) if cfg.use_contextual_seeds else set()
    low_precision_set = set(
        t.lower() for t in auto_vocab.get("low_precision", [])
    ) if cfg.use_low_precision_filter else set()

    doc_text = doc["text"]
    doc_id = doc.get("doc_id", "unknown")
    gt_terms_lower = {t.lower() for t in doc.get("gt_terms", [])}

    # --- Stage 1: Extraction ---
    if cfg.use_candidate_verify:
        from ..benchmark_prompt_variants import run_prompt_variant
        cv_terms, _ = run_prompt_variant(
            "candidate_verify_v1", doc, train_docs, index, model,
        )
        taxonomy_model = "sonnet" if cfg.use_sonnet_taxonomy else "haiku"
        taxonomy_terms, _ = _extract_haiku_taxonomy(doc, llm_model=taxonomy_model)
        heuristic_terms = _extract_heuristic(doc) if cfg.use_heuristic_extraction else []
        seed_terms = _extract_seeds(doc, seeds_list)
        contextual_seed_terms = (
            _extract_seeds(doc, list(auto_vocab.get("contextual_seeds", [])))
            if cfg.use_contextual_seeds else []
        )

        taxonomy_source_name = "sonnet_taxonomy" if cfg.use_sonnet_taxonomy else "haiku_taxonomy"
        candidates_by_source = {
            "candidate_verify": cv_terms,
            taxonomy_source_name: taxonomy_terms,
            "heuristic": heuristic_terms,
            "seeds": seed_terms,
            "contextual_seeds": contextual_seed_terms,
        }
    else:
        haiku_fewshot_terms, _ = _extract_haiku_fewshot(doc, train_docs, index, model)
        taxonomy_model = "sonnet" if cfg.use_sonnet_taxonomy else "haiku"
        taxonomy_terms, _ = _extract_haiku_taxonomy(doc, llm_model=taxonomy_model)
        haiku_simple_terms, _ = _extract_haiku_simple(doc)
        heuristic_terms = _extract_heuristic(doc) if cfg.use_heuristic_extraction else []
        seed_terms = _extract_seeds(doc, seeds_list)
        contextual_seed_terms = (
            _extract_seeds(doc, list(auto_vocab.get("contextual_seeds", [])))
            if cfg.use_contextual_seeds else []
        )

        taxonomy_source_name = "sonnet_taxonomy" if cfg.use_sonnet_taxonomy else "haiku_taxonomy"
        candidates_by_source = {
            "haiku_fewshot": haiku_fewshot_terms,
            taxonomy_source_name: taxonomy_terms,
            "haiku_simple": haiku_simple_terms,
            "heuristic": heuristic_terms,
            "seeds": seed_terms,
            "contextual_seeds": contextual_seed_terms,
        }

    # --- Stage 2: Grounding + dedup ---
    grounded = _ground_and_dedup(candidates_by_source, doc_text)

    # --- Stage 3: Noise filter ---
    after_noise: dict[str, dict] = {}
    for key, cand in grounded.items():
        term = cand["term"]
        if _auto_reject_noise(term, negatives_set, bypass_set, strategy=cfg, doc_text=doc_text):
            continue
        after_noise[key] = cand

    if cfg.use_candidate_verify:
        llm_sources = {"candidate_verify", taxonomy_source_name}
    else:
        llm_sources = {"haiku_fewshot", taxonomy_source_name, "haiku_simple"}

    # --- Stage 4: Confidence tier routing ---
    high_confidence: list[str] = []
    needs_validation: list[str] = []
    low_confidence: list[dict] = []

    seeds_set = {s.lower() for s in seeds_list}

    for key, cand in after_noise.items():
        term = cand["term"]
        source_count = cand["source_count"]
        sources = list(cand["sources"])
        sources_set = cand["sources"]

        tl = term.lower().strip()
        info = term_index.get(tl) if term_index else None
        entity_ratio = info["entity_ratio"] if info else 0.5

        has_llm_vote = bool(sources_set & llm_sources)
        is_heuristic_only = sources_set == {"heuristic"} or sources_set <= {"heuristic", "seeds", "contextual_seeds"}

        # ALL_CAPS corroboration: heuristic-only ALL_CAPS need ≥1 LLM vote
        # Skip for terms validated by training data (seeds/bypass/high entity_ratio)
        if (
            cfg.allcaps_require_corroboration
            and re.match(r"^[A-Z][A-Z0-9_]+$", term)
            and not has_llm_vote
            and tl not in bypass_set
            and tl not in seeds_set
            and entity_ratio < cfg.high_entity_ratio_threshold
        ):
            low_confidence.append({
                "term": term,
                "vote_count": source_count,
                "entity_ratio": entity_ratio,
                "sources": sources,
            })
            continue

        # LOW_PRECISION filter: borderline generic terms need 3+ sources
        if (
            cfg.use_low_precision_filter
            and tl in low_precision_set
            and source_count < cfg.high_confidence_min_sources
        ):
            low_confidence.append({
                "term": term,
                "vote_count": source_count,
                "entity_ratio": entity_ratio,
                "sources": sources,
            })
            continue

        # HIGH: structural pattern keeps (with training evidence guard)
        if _auto_keep_structural(term):
            if entity_ratio == 0 and tl not in seeds_set and tl not in bypass_set:
                needs_validation.append(term)
            elif cfg.structural_require_llm_vote and not has_llm_vote:
                needs_validation.append(term)
            else:
                high_confidence.append(term)
            continue

        # HIGH: seed bypass (data-driven common words from training)
        if cfg.seed_bypass_to_high_confidence and not cfg.disable_seed_bypass and tl in seeds_set and "seeds" in sources_set:
            if not cfg.seed_bypass_require_context:
                high_confidence.append(term)
                continue
            if source_count >= cfg.seed_bypass_min_sources_for_auto:
                high_confidence.append(term)
                continue
            if _has_technical_context(term, doc_text):
                high_confidence.append(term)
                continue

        # HIGH: 3+ sources agree
        if source_count >= cfg.high_confidence_min_sources:
            high_confidence.append(term)
            continue

        # HIGH: training data strongly supports (entity_ratio >= 0.8)
        if entity_ratio >= cfg.high_entity_ratio_threshold:
            high_confidence.append(term)
            continue

        # MEDIUM: contextual seed + any LLM vote
        if cfg.use_contextual_seeds and tl in contextual_seeds_set and "contextual_seeds" in sources_set:
            needs_validation.append(term)
            continue

        # MEDIUM: 2 sources agree
        if source_count >= cfg.validate_min_sources:
            needs_validation.append(term)
            continue

        # MEDIUM: training data moderately supports
        if entity_ratio >= cfg.medium_entity_ratio_threshold:
            needs_validation.append(term)
            continue

        # v5.2: Route single-vote LLM terms to validation instead of LOW
        if cfg.route_single_vote_to_validation and has_llm_vote and source_count >= 1:
            if entity_ratio >= cfg.single_vote_min_entity_ratio:
                needs_validation.append(term)
                continue

        # LOW: everything else
        low_confidence.append({
            "term": term,
            "vote_count": source_count,
            "entity_ratio": entity_ratio,
            "sources": sources,
        })

    # --- Log LOW confidence stats ---
    if cfg.log_low_confidence and low_confidence:
        for lc in low_confidence:
            in_gt = lc["term"].lower() in gt_terms_lower
            _log_low_confidence_stats(
                term=lc["term"],
                doc_id=doc_id,
                vote_count=lc["vote_count"],
                entity_ratio=lc["entity_ratio"],
                sources=lc["sources"],
                in_gt=in_gt,
            )

    # --- Stage 4b: Validate MEDIUM confidence ---
    if needs_validation and term_index:
        skip_validation = []
        needs_sonnet = []
        for term in needs_validation:
            tl = term.lower().strip()
            info = term_index.get(tl)
            er = info["entity_ratio"] if info else 0.5
            if er >= cfg.skip_validation_entity_ratio:
                skip_validation.append(term)
            else:
                needs_sonnet.append(term)

        validated_sonnet = _run_term_retrieval_validation(
            needs_sonnet, doc_text, term_index, strategy=cfg,
            bypass_set=bypass_set,
        ) if needs_sonnet else []

        validated = high_confidence + skip_validation + validated_sonnet
    elif needs_validation:
        validated = high_confidence + needs_validation
    else:
        validated = high_confidence

    # --- Stage 5: Post-processing ---
    expanded = _expand_spans(validated, doc_text)

    if cfg.suppress_path_embedded:
        expanded = [t for t in expanded if not _is_embedded_in_path(t, doc_text)]

    _url_re = re.compile(r"https?://\S+|^www\.\S+", re.I)
    expanded = [t for t in expanded if not _url_re.search(t)]

    protected_seed_set = seeds_set | contextual_seeds_set | bypass_set
    suppressed = _suppress_subspans(expanded, protected_seeds=protected_seed_set)

    seen: set[str] = set()
    final: list[str] = []
    for term in suppressed:
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            final.append(term)

    return final
