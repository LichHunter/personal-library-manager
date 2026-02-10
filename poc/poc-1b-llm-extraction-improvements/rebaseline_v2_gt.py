#!/usr/bin/env python3
"""Re-baseline top extraction strategies against the v2 ground truth.

Runs the best-performing strategies from previous experiments against the
expanded v2 GT (277 terms vs 163 in v1) to establish new baselines.

Strategies tested:
1. exhaustive_haiku — best recall in v1 (95.7%)
2. simple_haiku — solid balanced performer
3. quote_haiku — best hallucination (0%)
4. ensemble_verified — best F1 in v1 (0.874)
5. vote_3 — best precision in v1 (97.8%)
6. sonnet_conservative — second best precision (98.2%)

Usage:
    python rebaseline_v2_gt.py [--chunks N]
"""

import json
import re
import sys
import time
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
RESULTS_PATH = ARTIFACTS_DIR / "rebaseline_v2_results.json"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# EXTRACTION PROMPTS
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

QUOTE_PROMPT = """Extract technical terms from this documentation. For EACH term, provide an exact quote from the text containing it.

DOCUMENTATION:
{content}

For each term:
- term: The technical term (exact as it appears)
- quote: Verbatim substring from the documentation containing this term

Output JSON:
{{"terms": [{{"term": "example", "quote": "exact text from documentation containing example"}}]}}
"""

CONSERVATIVE_PROMPT = """Extract ONLY the most important domain-specific terms from this documentation.

DOCUMENTATION:
{content}

Focus on:
- Named resources, API objects, and components (e.g., Pod, Deployment, kubelet)
- Specific tools and CLI commands
- Proper nouns for technical things
- Domain-unique concepts

Do NOT include:
- Generic technical terms (cluster, container, node) unless they are THE core topic
- Common IT vocabulary
- Infrastructure terms unless domain-specific

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# ============================================================================
# EXTRACTION STRATEGIES
# ============================================================================


def extract_simple(content: str, model: str = "haiku") -> list[str]:
    """Simple extraction with basic prompt."""
    response = call_llm(SIMPLE_PROMPT.format(content=content), model=model, max_tokens=2000)
    return parse_terms(response)


def extract_exhaustive(content: str, model: str = "haiku") -> list[str]:
    """Exhaustive extraction — maximize recall."""
    response = call_llm(EXHAUSTIVE_PROMPT.format(content=content), model=model, max_tokens=3000)
    return parse_terms(response)


def extract_quote(content: str, model: str = "haiku") -> list[str]:
    """Quote-verified extraction — minimize hallucination."""
    response = call_llm(QUOTE_PROMPT.format(content=content), model=model, max_tokens=3000)
    return parse_quote_terms(response, content)


def extract_conservative(content: str, model: str = "sonnet") -> list[str]:
    """Conservative extraction with Sonnet — maximize precision."""
    response = call_llm(CONSERVATIVE_PROMPT.format(content=content), model=model, max_tokens=2000)
    return parse_terms(response)


def extract_ensemble_verified(content: str) -> list[str]:
    """Run 3 Haiku extractions, take union, verify with Haiku."""
    terms1 = extract_simple(content, "haiku")
    terms2 = extract_exhaustive(content, "haiku")
    terms3 = extract_quote(content, "haiku")

    # Union
    all_terms = set()
    for t in terms1 + terms2 + terms3:
        all_terms.add(t)

    # Verify: only keep terms that appear in the source text
    verified = []
    for t in all_terms:
        if verify_span(t, content):
            verified.append(t)

    return sorted(verified)


def extract_vote_3(content: str) -> list[str]:
    """Run 3 Haiku extractions, only keep terms appearing in 3/3."""
    terms1 = set(extract_simple(content, "haiku"))
    terms2 = set(extract_exhaustive(content, "haiku"))
    terms3 = set(extract_quote(content, "haiku"))

    # Count votes (case-insensitive)
    vote_count: dict[str, int] = {}
    original_form: dict[str, str] = {}
    for term_set in [terms1, terms2, terms3]:
        seen_this_round: set[str] = set()
        for t in term_set:
            key = normalize_for_voting(t)
            if key not in seen_this_round:
                vote_count[key] = vote_count.get(key, 0) + 1
                original_form[key] = t
                seen_this_round.add(key)

    # Keep only terms with 3 votes
    result = [original_form[k] for k, v in vote_count.items() if v >= 3]
    return sorted(result)


# ============================================================================
# PARSING HELPERS
# ============================================================================


def parse_terms(response: str) -> list[str]:
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


def parse_quote_terms(response: str, content: str) -> list[str]:
    """Parse quote-verified terms, filtering by actual span match."""
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
        terms_data = data.get("terms", [])
        results = []
        for item in terms_data:
            if isinstance(item, dict):
                term = item.get("term", "").strip()
                quote = item.get("quote", "").strip()
                if term and verify_span(term, content):
                    results.append(term)
            elif isinstance(item, str) and item.strip():
                if verify_span(item.strip(), content):
                    results.append(item.strip())
        return results
    except (json.JSONDecodeError, ValueError):
        return []


def verify_span(term: str, content: str) -> bool:
    """Check if term exists in content (case-insensitive)."""
    content_lower = content.lower()
    term_lower = term.lower()

    if term_lower in content_lower:
        return True
    # Normalized
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    content_norm = content_lower.replace("-", " ").replace("_", " ")
    if term_norm in content_norm:
        return True
    # CamelCase split
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True
    return False


def normalize_for_voting(term: str) -> str:
    """Normalize a term for voting comparison."""
    return term.lower().strip().replace("-", " ").replace("_", " ")


# ============================================================================
# METRICS
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted: str, ground_truth: str) -> bool:
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
    # Singular/plural
    if ext_norm.endswith("s") and ext_norm[:-1] == gt_norm:
        return True
    if gt_norm.endswith("s") and gt_norm[:-1] == ext_norm:
        return True
    return False


def calculate_metrics(extracted: list[str], gt_terms: list[str]) -> dict:
    matched_gt: set[int] = set()
    tp = 0

    for ext in extracted:
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_terms(ext, gt):
                matched_gt.add(j)
                tp += 1
                break

    fp = len(extracted) - tp
    fn = len(gt_terms) - tp
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

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
    }


# ============================================================================
# MAIN
# ============================================================================

STRATEGIES = {
    "exhaustive_haiku": lambda c: extract_exhaustive(c, "haiku"),
    "simple_haiku": lambda c: extract_simple(c, "haiku"),
    "quote_haiku": lambda c: extract_quote(c, "haiku"),
    "ensemble_verified": extract_ensemble_verified,
    "vote_3": extract_vote_3,
    "sonnet_conservative": lambda c: extract_conservative(c, "sonnet"),
}


def run_rebaseline(num_chunks: int = 15):
    """Re-baseline all strategies against v2 GT."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = BenchmarkLogger(
        log_dir=LOG_DIR,
        log_file=f"rebaseline_v2_{timestamp}.log",
        console=True,
        min_level="INFO",
    )

    logger.section("Re-Baseline: Top Strategies vs V2 Ground Truth")
    logger.info(f"GT: {GT_V2_PATH}")
    logger.info(f"Strategies: {list(STRATEGIES.keys())}")
    logger.info(f"Chunks: {num_chunks}")

    # Load v2 GT
    with open(GT_V2_PATH) as f:
        gt_v2 = json.load(f)

    chunks = gt_v2["chunks"][:num_chunks]
    logger.info(f"Loaded {len(chunks)} chunks ({gt_v2['total_terms']} total terms)")

    results: dict = {}

    for strat_name, strat_fn in STRATEGIES.items():
        logger.section(f"Strategy: {strat_name}")
        per_chunk: list[dict] = []
        total_time = 0

        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            content = chunk["content"]
            gt_terms = [t["term"] for t in chunk["terms"]]

            logger.info(f"  [{i+1}/{len(chunks)}] {chunk_id} (GT={len(gt_terms)} terms)")

            start = time.time()
            try:
                extracted = strat_fn(content)
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                extracted = []
            elapsed = time.time() - start
            total_time += elapsed

            metrics = calculate_metrics(extracted, gt_terms)
            logger.info(
                f"    P={metrics['precision']:.1%} R={metrics['recall']:.1%} "
                f"H={metrics['hallucination']:.1%} F1={metrics['f1']:.3f} "
                f"({metrics['tp']}tp/{metrics['fp']}fp/{metrics['fn']}fn) "
                f"ext={len(extracted)} t={elapsed:.1f}s"
            )

            per_chunk.append(
                {
                    "chunk_id": chunk_id,
                    "metrics": metrics,
                    "extracted_terms": extracted,
                    "ground_truth_terms": gt_terms,
                    "elapsed": elapsed,
                }
            )

        # Aggregate
        agg = {
            "precision": sum(c["metrics"]["precision"] for c in per_chunk) / len(per_chunk),
            "recall": sum(c["metrics"]["recall"] for c in per_chunk) / len(per_chunk),
            "hallucination": sum(c["metrics"]["hallucination"] for c in per_chunk)
            / len(per_chunk),
            "f1": sum(c["metrics"]["f1"] for c in per_chunk) / len(per_chunk),
            "total_time": total_time,
            "avg_time": total_time / len(per_chunk),
        }

        logger.section(f"AGGREGATE: {strat_name}")
        logger.info(
            f"  P={agg['precision']:.1%}  R={agg['recall']:.1%}  "
            f"H={agg['hallucination']:.1%}  F1={agg['f1']:.3f}  "
            f"Total={agg['total_time']:.1f}s"
        )

        results[strat_name] = {
            "aggregate": agg,
            "per_chunk": per_chunk,
        }

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "gt_version": "v2",
                "gt_total_terms": gt_v2["total_terms"],
                "num_chunks": len(chunks),
                "strategies": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Final comparison table
    logger.section("FINAL COMPARISON TABLE")
    logger.info(
        f"{'Strategy':25s} {'P':>7s} {'R':>7s} {'H':>7s} {'F1':>7s} {'Time':>7s}"
    )
    logger.info("-" * 60)
    for strat_name, strat_results in results.items():
        agg = strat_results["aggregate"]
        logger.info(
            f"{strat_name:25s} {agg['precision']:>6.1%} {agg['recall']:>6.1%} "
            f"{agg['hallucination']:>6.1%} {agg['f1']:>6.3f} {agg['total_time']:>6.1f}s"
        )

    # Target assessment
    logger.section("TARGET ASSESSMENT (95% P, 95% R, <5% H)")
    for strat_name, strat_results in results.items():
        agg = strat_results["aggregate"]
        p_gap = max(0, 0.95 - agg["precision"])
        r_gap = max(0, 0.95 - agg["recall"])
        h_gap = max(0, agg["hallucination"] - 0.05)
        status = "PASS" if p_gap == 0 and r_gap == 0 and h_gap == 0 else "FAIL"
        logger.info(
            f"  {strat_name:25s}: {status}  "
            f"P_gap={p_gap:+.1%} R_gap={r_gap:+.1%} H_gap={h_gap:+.1%}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-baseline strategies vs v2 GT")
    parser.add_argument(
        "--chunks",
        type=int,
        default=15,
        help="Number of chunks to test (default: all 15)",
    )
    args = parser.parse_args()

    run_rebaseline(num_chunks=args.chunks)
