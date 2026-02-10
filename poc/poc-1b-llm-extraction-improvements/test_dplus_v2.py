#!/usr/bin/env python3
"""Architecture D+v2: Sonnet-Anchored Complementary Merge.

Pipeline:
  Phase 1: 3 parallel extractors (Sonnet exhaustive + Haiku exhaustive + Haiku simple)
  Phase 2: Span grounding (deterministic) — discard ungrounded terms
  Phase 3: Vote-based routing (3-vote auto-keep, 2-vote auto-keep, 1-vote → Phase 4)
  Phase 4: Sonnet batch approval for 1-vote terms only
  Phase 5: Assembly + deduplication

Design rationale:
  - Sonnet exhaustive achieves 94.3% recall (vs Haiku 74.8%) — it's the recall anchor
  - 3-vote terms have 95.8% precision — auto-keep
  - 2-vote terms have 84.4% precision — auto-keep (grounded)
  - 1-vote terms have 52.9% precision — Sonnet filters noise

Results saved with full per-term audit trail:
  - Which extractors found each term
  - Vote count per term
  - Span grounding result
  - Sonnet approval decision (for 1-vote terms)
  - Final status (KEPT / REJECTED_UNGROUNDED / REJECTED_SONNET)

Usage:
    python test_dplus_v2.py [--chunks N]
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"
RESULTS_PATH = ARTIFACTS_DIR / "dplus_v2_results.json"
AUDIT_PATH = ARTIFACTS_DIR / "dplus_v2_audit.json"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# PROMPTS
# ============================================================================

EXHAUSTIVE_PROMPT = """Extract ALL technical terms from the following documentation chunk. Be EXHAUSTIVE — capture every technical term, concept, resource, component, tool, protocol, abbreviation, and domain-specific vocabulary.

DOCUMENTATION:
{content}

Extract every term that someone studying this documentation would need to understand. This includes:
- Domain-specific resources, components, and concepts
- Tools, CLI commands, API objects, and protocols
- Technical vocabulary (even if the term also exists in other domains)
- Abbreviations, acronyms, and proper nouns
- Infrastructure, security, and networking terms used in technical context
- Architecture and process terms (e.g., "high availability", "garbage collection")
- Compound terms AND their key individual components when independently meaningful

Rules:
- Extract terms EXACTLY as they appear in the text
- Be EXHAUSTIVE — missing a term is worse than including a borderline one
- DO include terms used across multiple domains IF they carry technical meaning here
- DO NOT include structural/formatting words (title, section, overview, content, weight)

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

SIMPLE_PROMPT = """Extract technical terms from this documentation chunk.

DOCUMENTATION:
{content}

List all technical terms, concepts, and domain-specific vocabulary.

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

SONNET_APPROVAL_PROMPT = """You are reviewing candidate technical terms extracted from a documentation chunk. For each term, decide APPROVE or REJECT.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS (each found by only ONE extractor — lower confidence):
{terms_json}

DEFAULT: APPROVE. Only REJECT terms that clearly fail ALL criteria below.

APPROVE if ANY of these are true:
1. The term names a technical concept, resource, tool, protocol, or component discussed in the chunk
2. The term is technical vocabulary that a learner studying this documentation would benefit from understanding
3. The term refers to infrastructure, security, networking, or system concepts with specific meaning in this context
4. The term is an abbreviation, acronym, CLI flag, API path, or version identifier

REJECT only if ALL of these are true:
1. The term is purely structural/formatting (e.g., "title", "section", "overview", "body")
2. AND it names NO technical concept, resource, tool, or domain entity
3. AND a learner would gain NO technical understanding from looking up this term

When in doubt, APPROVE. False negatives are MUCH worse than false positives.

For EACH term provide:
- term: The term name
- decision: "APPROVE" or "REJECT"
- reasoning: Brief justification (1 sentence)

Output JSON:
{{
  "terms": [
    {{"term": "Pod", "decision": "APPROVE", "reasoning": "Core resource type"}},
    {{"term": "title", "decision": "REJECT", "reasoning": "Structural YAML key, not technical"}}
  ]
}}
"""

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class TermAudit:
    """Full audit trail for a single candidate term."""

    term: str
    normalized: str
    sources: list[str]  # which extractors found it
    vote_count: int
    is_grounded: bool
    grounding_type: str  # "exact", "normalized", "camelcase", "singular_plural", "none"
    routing: str  # "auto_keep_3vote", "auto_keep_2vote", "sonnet_review", "rejected_ungrounded"
    sonnet_decision: str  # "APPROVE", "REJECT", "N/A" (for auto-kept)
    sonnet_reasoning: str
    final_status: str  # "KEPT", "REJECTED_UNGROUNDED", "REJECTED_SONNET"
    matched_gt: str  # which GT term it matched (empty if false positive)


# ============================================================================
# PARSING
# ============================================================================


def parse_terms_response(response: str) -> list[str]:
    """Parse a JSON response containing a terms list."""
    if not response:
        return []
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return []
        data = json.loads(json_match.group())
        terms = data.get("terms", [])
        if isinstance(terms, list):
            return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def parse_approval_response(response: str, logger: BenchmarkLogger) -> dict[str, dict]:
    """Parse Sonnet approval response. Returns {term: {decision, reasoning}}."""
    if not response:
        return {}
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warn("No JSON found in Sonnet approval response")
            return {}
        data = json.loads(json_match.group())
        terms_data = data.get("terms", [])

        decisions: dict[str, dict] = {}
        for item in terms_data:
            if isinstance(item, dict):
                term = item.get("term", "").strip()
                decision = item.get("decision", "APPROVE").strip().upper()
                reasoning = item.get("reasoning", "").strip()
                if decision in ("KEEP",):
                    decision = "APPROVE"
                elif decision in ("REMOVE",):
                    decision = "REJECT"
                if term:
                    decisions[term] = {"decision": decision, "reasoning": reasoning}
        return decisions
    except (json.JSONDecodeError, ValueError) as e:
        logger.warn(f"Approval parse error: {e}")
        return {}


# ============================================================================
# SPAN GROUNDING
# ============================================================================


def verify_span(term: str, content: str) -> tuple[bool, str]:
    """Verify term exists in content. Returns (grounded, match_type)."""
    if not term or len(term) < 2:
        return False, "too_short"

    content_lower = content.lower()
    term_lower = term.lower().strip()

    # Exact match (case-insensitive)
    if term_lower in content_lower:
        return True, "exact"

    # Normalized (- and _ as spaces)
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    content_norm = content_lower.replace("-", " ").replace("_", " ")
    if term_norm in content_norm:
        return True, "normalized"

    # CamelCase split
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True, "camelcase"

    # Singular/plural
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in content_lower:
        return True, "singular_of_plural"
    if not term_lower.endswith("s") and (term_lower + "s") in content_lower:
        return True, "plural_of_singular"
    if term_lower.endswith("es") and len(term_lower) > 4 and term_lower[:-2] in content_lower:
        return True, "singular_of_plural_es"

    return False, "none"


# ============================================================================
# METRICS
# ============================================================================


def normalize_term(term: str) -> str:
    """Normalize for dedup and matching."""
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms_fn(extracted: str, ground_truth: str) -> bool:
    """Check if extracted term matches a ground truth term."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)
    if ext_norm == gt_norm:
        return True
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    if ext_norm.endswith("s") and ext_norm[:-1] == gt_norm:
        return True
    if gt_norm.endswith("s") and gt_norm[:-1] == ext_norm:
        return True
    return False


def calculate_metrics(extracted: list[str], gt_terms: list[str]) -> dict:
    """Calculate P/R/H/F1 metrics."""
    matched_gt: set[int] = set()
    tp = 0
    tp_terms: list[tuple[str, str]] = []  # (extracted, matched_gt)
    fp_terms: list[str] = []

    for ext in extracted:
        found = False
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_terms_fn(ext, gt):
                matched_gt.add(j)
                tp += 1
                tp_terms.append((ext, gt))
                found = True
                break
        if not found:
            fp_terms.append(ext)

    fn_terms = [gt_terms[j] for j in range(len(gt_terms)) if j not in matched_gt]

    fp = len(extracted) - tp
    fn = len(gt_terms) - tp
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted_count": len(extracted),
        "gt_count": len(gt_terms),
        "tp_terms": tp_terms,
        "fp_terms": fp_terms,
        "fn_terms": fn_terms,
    }


# ============================================================================
# D+v2 PIPELINE
# ============================================================================


def run_dplus_v2_chunk(
    chunk_content: str,
    chunk_id: str,
    logger: BenchmarkLogger,
) -> tuple[list[str], list[dict]]:
    """Run D+v2 pipeline on a single chunk.

    Returns:
        (final_terms, audit_trail) where audit_trail is a list of TermAudit dicts
    """

    # ── PHASE 1: Three-Extractor Extraction ──────────────────────────────
    logger.subsection("Phase 1: Extraction")

    # 1a: Sonnet exhaustive (recall anchor)
    logger.info("  [1a] Sonnet exhaustive...")
    t0 = time.time()
    sonnet_response = call_llm(
        EXHAUSTIVE_PROMPT.format(content=chunk_content),
        model="sonnet",
        max_tokens=3000,
        temperature=0.0,
    )
    sonnet_terms = parse_terms_response(sonnet_response)
    t_sonnet = time.time() - t0
    logger.info(f"  [1a] Sonnet: {len(sonnet_terms)} terms in {t_sonnet:.1f}s")

    # 1b: Haiku exhaustive
    logger.info("  [1b] Haiku exhaustive...")
    t0 = time.time()
    haiku_exh_response = call_llm(
        EXHAUSTIVE_PROMPT.format(content=chunk_content),
        model="haiku",
        max_tokens=3000,
        temperature=0.0,
    )
    haiku_exh_terms = parse_terms_response(haiku_exh_response)
    t_haiku_exh = time.time() - t0
    logger.info(f"  [1b] Haiku exh: {len(haiku_exh_terms)} terms in {t_haiku_exh:.1f}s")

    # 1c: Haiku simple
    logger.info("  [1c] Haiku simple...")
    t0 = time.time()
    haiku_sim_response = call_llm(
        SIMPLE_PROMPT.format(content=chunk_content),
        model="haiku",
        max_tokens=2000,
        temperature=0.0,
    )
    haiku_sim_terms = parse_terms_response(haiku_sim_response)
    t_haiku_sim = time.time() - t0
    logger.info(f"  [1c] Haiku sim: {len(haiku_sim_terms)} terms in {t_haiku_sim:.1f}s")

    # Build candidate pool with vote tracking
    candidates: dict[str, dict] = {}  # normalized -> {term, sources, vote_count}
    for source_name, source_terms in [
        ("sonnet_exhaustive", sonnet_terms),
        ("haiku_exhaustive", haiku_exh_terms),
        ("haiku_simple", haiku_sim_terms),
    ]:
        seen_this_source: set[str] = set()
        for t in source_terms:
            key = normalize_term(t)
            if key in seen_this_source:
                continue
            seen_this_source.add(key)
            if key not in candidates:
                candidates[key] = {"term": t, "sources": [], "vote_count": 0}
            candidates[key]["sources"].append(source_name)
            candidates[key]["vote_count"] += 1

    logger.info(
        f"  Union: {len(candidates)} unique candidates "
        f"(S={len(sonnet_terms)}, He={len(haiku_exh_terms)}, Hs={len(haiku_sim_terms)})"
    )

    # Vote distribution
    vote_dist = {1: 0, 2: 0, 3: 0}
    for c in candidates.values():
        vote_dist[c["vote_count"]] = vote_dist.get(c["vote_count"], 0) + 1
    logger.info(f"  Votes: 3={vote_dist.get(3, 0)}, 2={vote_dist.get(2, 0)}, 1={vote_dist.get(1, 0)}")

    # ── PHASE 2: Span Grounding ──────────────────────────────────────────
    logger.subsection("Phase 2: Span Grounding")

    grounded: dict[str, dict] = {}
    ungrounded: list[str] = []
    grounding_results: dict[str, tuple[bool, str]] = {}

    for key, cand in candidates.items():
        is_grounded, match_type = verify_span(cand["term"], chunk_content)
        grounding_results[key] = (is_grounded, match_type)
        if is_grounded:
            grounded[key] = cand
        else:
            ungrounded.append(cand["term"])

    logger.info(
        f"  Grounded: {len(grounded)} / {len(candidates)} "
        f"({len(ungrounded)} removed)"
    )
    if ungrounded:
        for t in ungrounded[:10]:
            logger.info(f"    UNGROUNDED: '{t}'")
        if len(ungrounded) > 10:
            logger.info(f"    ... and {len(ungrounded) - 10} more")

    # ── PHASE 3: Vote-Based Routing ──────────────────────────────────────
    logger.subsection("Phase 3: Vote-Based Routing")

    auto_kept: list[str] = []  # 3-vote and 2-vote grounded terms
    needs_review: list[str] = []  # 1-vote grounded terms → Phase 4

    for key, cand in grounded.items():
        if cand["vote_count"] >= 2:
            auto_kept.append(cand["term"])
        else:
            needs_review.append(cand["term"])

    logger.info(f"  Auto-kept (2+ votes, grounded): {len(auto_kept)}")
    logger.info(f"  Needs Sonnet review (1 vote): {len(needs_review)}")

    # ── PHASE 4: Sonnet Batch Approval ───────────────────────────────────
    sonnet_decisions: dict[str, dict] = {}  # term -> {decision, reasoning}

    if needs_review:
        logger.subsection("Phase 4: Sonnet Batch Approval")
        logger.info(f"  Reviewing {len(needs_review)} 1-vote terms...")

        terms_for_review = json.dumps(needs_review, indent=2)
        approval_prompt = SONNET_APPROVAL_PROMPT.format(
            content=chunk_content[:3000],
            terms_json=terms_for_review,
        )

        t0 = time.time()
        approval_response = call_llm(
            approval_prompt,
            model="sonnet",
            max_tokens=3000,
            temperature=0.0,
        )
        t_approval = time.time() - t0

        sonnet_decisions = parse_approval_response(approval_response, logger)
        logger.info(f"  Sonnet responded in {t_approval:.1f}s")

        # Log each decision
        approved_count = 0
        rejected_count = 0
        for t in needs_review:
            decision_info = sonnet_decisions.get(t, {})
            decision = decision_info.get("decision", "APPROVE")  # default approve
            reasoning = decision_info.get("reasoning", "No response from Sonnet")
            if decision == "APPROVE":
                approved_count += 1
                logger.info(f"    APPROVE: '{t}' — {reasoning}")
            else:
                rejected_count += 1
                logger.info(f"    REJECT:  '{t}' — {reasoning}")

        logger.info(
            f"  Sonnet decisions: {approved_count} APPROVE, {rejected_count} REJECT"
        )
    else:
        logger.info("  Phase 4: Skipped (no 1-vote terms to review)")

    # ── PHASE 5: Assembly ────────────────────────────────────────────────
    logger.subsection("Phase 5: Assembly")

    final_terms: list[str] = list(auto_kept)

    # Add approved 1-vote terms
    for t in needs_review:
        decision_info = sonnet_decisions.get(t, {})
        decision = decision_info.get("decision", "APPROVE")
        if decision == "APPROVE":
            final_terms.append(t)

    # Deduplicate by normalized form (keep first occurrence)
    seen_normalized: set[str] = set()
    deduped_terms: list[str] = []
    for t in final_terms:
        key = normalize_term(t)
        if key not in seen_normalized:
            seen_normalized.add(key)
            deduped_terms.append(t)

    logger.info(f"  Final terms: {len(deduped_terms)} (before dedup: {len(final_terms)})")

    # ── BUILD AUDIT TRAIL ────────────────────────────────────────────────
    audit_trail: list[dict] = []

    for key, cand in candidates.items():
        is_grounded, match_type = grounding_results[key]

        if not is_grounded:
            routing = "rejected_ungrounded"
            sonnet_dec = "N/A"
            sonnet_reason = ""
            final_status = "REJECTED_UNGROUNDED"
        elif cand["vote_count"] >= 2:
            routing = f"auto_keep_{cand['vote_count']}vote"
            sonnet_dec = "N/A"
            sonnet_reason = ""
            final_status = "KEPT"
        else:
            routing = "sonnet_review"
            dec_info = sonnet_decisions.get(cand["term"], {})
            sonnet_dec = dec_info.get("decision", "APPROVE")
            sonnet_reason = dec_info.get("reasoning", "Default approve (no response)")
            final_status = "KEPT" if sonnet_dec == "APPROVE" else "REJECTED_SONNET"

        audit_trail.append(
            {
                "term": cand["term"],
                "normalized": key,
                "sources": cand["sources"],
                "vote_count": cand["vote_count"],
                "is_grounded": is_grounded,
                "grounding_type": match_type,
                "routing": routing,
                "sonnet_decision": sonnet_dec,
                "sonnet_reasoning": sonnet_reason,
                "final_status": final_status,
                "matched_gt": "",  # filled in later during metrics
            }
        )

    return deduped_terms, audit_trail


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment(num_chunks: int = 15):
    """Run D+v2 experiment on all chunks."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = BenchmarkLogger(
        log_dir=LOG_DIR,
        log_file=f"dplus_v2_{timestamp}.log",
        console=True,
        min_level="INFO",
    )

    logger.section("D+v2: Sonnet-Anchored Complementary Merge")
    logger.info(f"GT: {GT_V2_PATH}")
    logger.info(f"Chunks: {num_chunks}")
    logger.info(f"Timestamp: {timestamp}")

    # Load v2 GT
    with open(GT_V2_PATH) as f:
        gt_v2 = json.load(f)

    chunks = gt_v2["chunks"][:num_chunks]
    logger.info(f"Loaded {len(chunks)} chunks ({gt_v2['total_terms']} total terms)")

    per_chunk_results: list[dict] = []
    all_audits: list[dict] = []
    total_time = 0

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        gt_terms = [t["term"] for t in chunk["terms"]]
        gt_tiers = {t["term"]: t["tier"] for t in chunk["terms"]}

        logger.section(f"Chunk {i + 1}/{len(chunks)}: {chunk_id}")
        logger.info(f"Content length: {len(content)}")
        logger.info(f"GT terms: {len(gt_terms)}")

        start = time.time()
        final_terms, audit_trail = run_dplus_v2_chunk(content, chunk_id, logger)
        elapsed = time.time() - start
        total_time += elapsed

        # Calculate metrics
        metrics = calculate_metrics(final_terms, gt_terms)

        # Enrich audit trail with GT match info
        tp_map = {ext: gt for ext, gt in metrics["tp_terms"]}
        for audit in audit_trail:
            if audit["final_status"] == "KEPT":
                matched = tp_map.get(audit["term"], "")
                audit["matched_gt"] = matched
            audit["chunk_id"] = chunk_id

        all_audits.extend(audit_trail)

        # Log results
        logger.section(f"Results: {chunk_id}")
        logger.info(
            f"P={metrics['precision']:.1%}  R={metrics['recall']:.1%}  "
            f"H={metrics['hallucination']:.1%}  F1={metrics['f1']:.3f}"
        )
        logger.info(
            f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
            f"Extracted={metrics['extracted_count']}  GT={metrics['gt_count']}"
        )
        logger.info(f"Time: {elapsed:.1f}s")

        # Log false positives
        if metrics["fp_terms"]:
            logger.subsection("False Positives (extracted but not in GT)")
            for fp in metrics["fp_terms"]:
                # Find audit entry for this FP
                fp_audit = next(
                    (a for a in audit_trail if a["term"] == fp and a["final_status"] == "KEPT"),
                    None,
                )
                sources = fp_audit["sources"] if fp_audit else ["?"]
                votes = fp_audit["vote_count"] if fp_audit else 0
                logger.info(f"  FP: '{fp}' (votes={votes}, src={sources})")

        # Log false negatives
        if metrics["fn_terms"]:
            logger.subsection("False Negatives (in GT but not extracted)")
            for fn_term in metrics["fn_terms"]:
                tier = gt_tiers.get(fn_term, "?")
                # Check if it was extracted but rejected
                fn_audit = next(
                    (
                        a
                        for a in audit_trail
                        if normalize_term(a["term"]) == normalize_term(fn_term)
                        and a["final_status"] != "KEPT"
                    ),
                    None,
                )
                if fn_audit:
                    logger.info(
                        f"  FN: '{fn_term}' (T{tier}) — WAS EXTRACTED but {fn_audit['final_status']}: "
                        f"votes={fn_audit['vote_count']}, routing={fn_audit['routing']}"
                    )
                else:
                    logger.info(f"  FN: '{fn_term}' (T{tier}) — NEVER EXTRACTED by any model")

        per_chunk_results.append(
            {
                "chunk_id": chunk_id,
                "metrics": {k: v for k, v in metrics.items() if k not in ("tp_terms", "fp_terms", "fn_terms")},
                "extracted_terms": final_terms,
                "ground_truth_terms": gt_terms,
                "fp_terms": metrics["fp_terms"],
                "fn_terms": metrics["fn_terms"],
                "tp_matches": [(ext, gt) for ext, gt in metrics["tp_terms"]],
                "elapsed": elapsed,
            }
        )

    # ── AGGREGATE METRICS ────────────────────────────────────────────────
    logger.section("AGGREGATE RESULTS")

    agg = {
        "precision": sum(c["metrics"]["precision"] for c in per_chunk_results) / len(per_chunk_results),
        "recall": sum(c["metrics"]["recall"] for c in per_chunk_results) / len(per_chunk_results),
        "hallucination": sum(c["metrics"]["hallucination"] for c in per_chunk_results) / len(per_chunk_results),
        "f1": sum(c["metrics"]["f1"] for c in per_chunk_results) / len(per_chunk_results),
        "total_time": total_time,
        "avg_time": total_time / len(per_chunk_results),
    }

    logger.info(f"Precision:     {agg['precision']:.1%}")
    logger.info(f"Recall:        {agg['recall']:.1%}")
    logger.info(f"Hallucination: {agg['hallucination']:.1%}")
    logger.info(f"F1:            {agg['f1']:.3f}")
    logger.info(f"Total time:    {agg['total_time']:.1f}s")
    logger.info(f"Avg time/chunk: {agg['avg_time']:.1f}s")

    # Target assessment
    logger.subsection("Target Assessment (95% P, 95% R, <5% H)")
    p_gap = max(0, 0.95 - agg["precision"])
    r_gap = max(0, 0.95 - agg["recall"])
    h_gap = max(0, agg["hallucination"] - 0.05)
    status = "PASS" if p_gap == 0 and r_gap == 0 and h_gap == 0 else "FAIL"
    logger.info(f"Status: {status}")
    logger.info(f"P gap: {p_gap:+.1%}  R gap: {r_gap:+.1%}  H gap: {h_gap:+.1%}")

    # Per-chunk summary table
    logger.subsection("Per-Chunk Summary")
    logger.info(
        f"{'Chunk':55s} {'P':>6s} {'R':>6s} {'H':>6s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}"
    )
    logger.info("-" * 95)
    for r in per_chunk_results:
        m = r["metrics"]
        logger.info(
            f"{r['chunk_id']:55s} {m['precision']:>5.1%} {m['recall']:>5.1%} "
            f"{m['hallucination']:>5.1%} {m['f1']:>5.3f} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d}"
        )

    # Audit summary
    logger.subsection("Audit Summary")
    status_counts: dict[str, int] = {}
    routing_counts: dict[str, int] = {}
    for a in all_audits:
        status_counts[a["final_status"]] = status_counts.get(a["final_status"], 0) + 1
        routing_counts[a["routing"]] = routing_counts.get(a["routing"], 0) + 1

    logger.info("Final status distribution:")
    for s, count in sorted(status_counts.items()):
        logger.info(f"  {s}: {count}")
    logger.info("Routing distribution:")
    for r, count in sorted(routing_counts.items()):
        logger.info(f"  {r}: {count}")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────
    results = {
        "strategy": "dplus_v2",
        "gt_version": "v2",
        "gt_total_terms": gt_v2["total_terms"],
        "num_chunks": len(chunks),
        "timestamp": timestamp,
        "aggregate_metrics": agg,
        "per_chunk_results": per_chunk_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Save audit trail
    audit_data = {
        "strategy": "dplus_v2",
        "timestamp": timestamp,
        "total_candidates": len(all_audits),
        "status_distribution": status_counts,
        "routing_distribution": routing_counts,
        "audits": all_audits,
    }
    with open(AUDIT_PATH, "w") as f:
        json.dump(audit_data, f, indent=2)
    logger.info(f"Audit trail saved to {AUDIT_PATH}")

    logger.summary()
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D+v2 benchmark")
    parser.add_argument(
        "--chunks",
        type=int,
        default=15,
        help="Number of chunks to test (default: all 15)",
    )
    args = parser.parse_args()

    run_experiment(num_chunks=args.chunks)
