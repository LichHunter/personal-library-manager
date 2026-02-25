#!/usr/bin/env python3
"""Test combined strategies to achieve both high recall AND low hallucination.

Current best results:
- ensemble_haiku: 92% recall, 48% hallucination
- quote_haiku: 75% recall, 21% hallucination

Target: 95%+ recall, <10% hallucination

Key insight: "hallucination" here means terms not in ground truth, but they're
still valid terms (span verification ensures they exist in text). The real
question is: are they RELEVANT technical terms worth indexing?

Strategies to test:
1. High-recall extraction + LLM verification (filter out non-domain terms)
2. Intersection voting (keep terms that multiple strategies agree on)
3. Confidence-weighted extraction (only keep high-confidence terms)
4. Two-pass: liberal extraction → conservative filtering
5. Expanded ground truth comparison (are "hallucinations" actually valid?)
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

# Add paths
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)

from utils.llm_provider import call_llm

print("POC-1b: Combined Strategies for High Recall + Low Hallucination", flush=True)
print("=" * 70, flush=True)

# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


# ============================================================================
# LOAD GROUND TRUTH
# ============================================================================


def load_ground_truth() -> list[dict]:
    """Load the existing ground truth."""
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")

    with open(GROUND_TRUTH_PATH) as f:
        data = json.load(f)

    return data["chunks"]


# ============================================================================
# TERM VERIFICATION
# ============================================================================


def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content (strict)."""
    if not term or len(term) < 2:
        return False

    content_lower = content.lower()
    term_lower = term.lower().strip()

    # Exact match
    if term_lower in content_lower:
        return True

    # Handle underscores/hyphens
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True

    # CamelCase split
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True

    return False


# ============================================================================
# EXTRACTION PROMPTS
# ============================================================================

SIMPLE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

CHUNK:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}"""


QUOTE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

For EACH term, provide the exact quote where it appears.

CHUNK:
{content}

Output JSON: {{"terms": [
  {{"quote": "exact text from chunk", "term": "TermName"}}
]}}"""


EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

Be EXHAUSTIVE. Include:
- Resource types (Pod, Service, Deployment, etc.)
- Components (kubelet, kubectl, etcd, etc.)  
- Concepts (namespace, label, selector, etc.)
- Feature gates (CamelCase names)
- Lifecycle stages (alpha, beta, stable)
- CLI flags (--flag-name)
- API terms (spec, status, metadata)

CHUNK:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}"""


# Verification prompt - given extracted terms, filter to only domain-relevant ones
VERIFY_PROMPT = """You are filtering a list of extracted terms from Kubernetes documentation.

ORIGINAL CHUNK:
{content}

EXTRACTED TERMS:
{terms}

Your task: Keep ONLY terms that are:
1. Kubernetes-specific technical terms (resources, components, concepts)
2. Domain-relevant terms (container, orchestration, cluster concepts)
3. Configuration/API terms specific to K8s

REMOVE:
- Generic English words (unless K8s-specific in context)
- YAML structure keywords (title, stages, defaultValue)
- Version numbers alone
- File paths or URLs
- Terms that don't add value for documentation search

Output JSON with ONLY the filtered terms:
{{"terms": ["term1", "term2", ...]}}"""


# High-precision extraction - explicitly conservative
CONSERVATIVE_PROMPT = """Extract ONLY the most important Kubernetes technical terms from this chunk.

CHUNK:
{content}

Be CONSERVATIVE. Only extract terms that are:
1. Core Kubernetes resources (Pod, Service, Deployment, Node, etc.)
2. Key components (kubelet, kube-apiserver, etcd, etc.)
3. Essential concepts (namespace, control plane, cluster, etc.)

DO NOT extract:
- Generic terms (memory, network, process)
- Implied concepts not explicitly mentioned
- Variations of the same term (just pick canonical form)

Output JSON: {{"terms": ["term1", "term2", ...]}}"""


# Two-pass: first extract liberally, then ask LLM to rate importance
RATE_PROMPT = """Rate each extracted term's importance for a Kubernetes documentation search index.

CHUNK CONTEXT:
{content}

EXTRACTED TERMS:
{terms}

For each term, rate as:
- ESSENTIAL (1): Core K8s resource, component, or must-know concept
- IMPORTANT (2): Significant technical term, should be indexed
- OPTIONAL (3): Nice-to-have, but not critical for search
- NOISE (4): Generic word, not worth indexing

Output JSON:
{{"ratings": [
  {{"term": "Pod", "rating": 1}},
  {{"term": "scheduling", "rating": 2}}
]}}"""


# ============================================================================
# PARSING HELPERS
# ============================================================================


def parse_terms_response(response: str, require_quotes: bool = False) -> list[str]:
    """Parse LLM response to get term list."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms_data = data.get("terms", [])

            if require_quotes:
                return [t.get("term", "") for t in terms_data if isinstance(t, dict)]
            elif isinstance(terms_data, list):
                if terms_data and isinstance(terms_data[0], dict):
                    return [t.get("term", "") for t in terms_data]
                else:
                    return [str(t) for t in terms_data]
    except (json.JSONDecodeError, KeyError):
        pass
    return []


def parse_ratings_response(response: str) -> dict[str, int]:
    """Parse rating response to get term -> rating map."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            ratings = data.get("ratings", [])
            return {
                r["term"]: r["rating"] for r in ratings if "term" in r and "rating" in r
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {}


# ============================================================================
# EXTRACTION STRATEGIES
# ============================================================================


def extract_simple(content: str, model: str = "claude-haiku") -> list[str]:
    """Simple extraction strategy."""
    prompt = SIMPLE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_quote(content: str, model: str = "claude-haiku") -> list[str]:
    """Quote-based extraction strategy."""
    prompt = QUOTE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    terms = parse_terms_response(response, require_quotes=True)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_exhaustive(content: str, model: str = "claude-haiku") -> list[str]:
    """Exhaustive extraction strategy."""
    prompt = EXHAUSTIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_conservative(content: str, model: str = "claude-haiku") -> list[str]:
    """Conservative extraction - only essential terms."""
    prompt = CONSERVATIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


# ============================================================================
# COMBINED STRATEGIES
# ============================================================================


def strategy_ensemble_verified(content: str, model: str = "claude-haiku") -> list[str]:
    """High-recall ensemble + LLM verification to filter noise.

    1. Run multiple extraction strategies (high recall)
    2. Union all results
    3. Use LLM to filter out non-domain terms
    """
    # Step 1: High-recall extraction
    all_terms = set()
    all_terms.update(extract_simple(content, model))
    all_terms.update(extract_quote(content, model))
    all_terms.update(extract_exhaustive(content, model))

    if not all_terms:
        return []

    # Step 2: LLM verification/filtering
    terms_list = sorted(all_terms)
    prompt = VERIFY_PROMPT.format(content=content[:2000], terms=json.dumps(terms_list))

    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    verified = parse_terms_response(response)

    # Final span verification (safety)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_intersection_voting(
    content: str, model: str = "claude-haiku", min_votes: int = 2
) -> list[str]:
    """Keep only terms that multiple strategies agree on.

    1. Run multiple extraction strategies
    2. Count votes for each term
    3. Keep terms with >= min_votes
    """
    # Run strategies
    simple = set(extract_simple(content, model))
    quote = set(extract_quote(content, model))
    exhaustive = set(extract_exhaustive(content, model))
    conservative = set(extract_conservative(content, model))

    # Count votes (case-insensitive)
    term_votes = {}
    for term_set in [simple, quote, exhaustive, conservative]:
        for term in term_set:
            key = term.lower()
            if key not in term_votes:
                term_votes[key] = {"canonical": term, "votes": 0}
            term_votes[key]["votes"] += 1

    # Keep terms with enough votes
    result = [
        info["canonical"] for info in term_votes.values() if info["votes"] >= min_votes
    ]

    return result


def strategy_rated_extraction(
    content: str, model: str = "claude-haiku", max_rating: int = 2
) -> list[str]:
    """Extract liberally, then rate and filter by importance.

    1. Run exhaustive extraction (high recall)
    2. Ask LLM to rate each term's importance
    3. Keep only essential + important terms (rating <= max_rating)
    """
    # Step 1: Liberal extraction
    all_terms = set()
    all_terms.update(extract_simple(content, model))
    all_terms.update(extract_exhaustive(content, model))

    if not all_terms:
        return []

    # Step 2: Rate terms
    terms_list = sorted(all_terms)
    prompt = RATE_PROMPT.format(content=content[:2000], terms=json.dumps(terms_list))

    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    ratings = parse_ratings_response(response)

    # Step 3: Filter by rating
    result = []
    for term in terms_list:
        rating = ratings.get(term, 3)  # Default to OPTIONAL if not rated
        if rating <= max_rating:
            result.append(term)

    return result


def strategy_conservative_union(content: str, model: str = "claude-haiku") -> list[str]:
    """Union of conservative extractions from multiple models.

    Use conservative prompts with both Haiku and Sonnet, union results.
    """
    haiku_terms = set(extract_conservative(content, "claude-haiku"))
    sonnet_terms = set(extract_conservative(content, "claude-sonnet"))

    return list(haiku_terms | sonnet_terms)


def strategy_quote_exhaustive_intersect(
    content: str, model: str = "claude-haiku"
) -> list[str]:
    """Intersection of quote-based and exhaustive extraction.

    Quote-based has lower hallucination, exhaustive has higher recall.
    Intersection should have both benefits.
    """
    quote = set(t.lower() for t in extract_quote(content, model))
    exhaustive_terms = extract_exhaustive(content, model)

    # Keep exhaustive terms that also appear in quote extraction
    result = []
    for term in exhaustive_terms:
        if term.lower() in quote:
            result.append(term)

    # Also add all quote terms (they're high confidence)
    quote_terms = extract_quote(content, model)
    for term in quote_terms:
        if term not in result:
            result.append(term)

    return result


def strategy_multi_model_consensus(content: str) -> list[str]:
    """Run same extraction on multiple models, keep consensus.

    Terms that both Haiku and Sonnet extract are likely valid.
    """
    haiku_simple = set(t.lower() for t in extract_simple(content, "claude-haiku"))
    sonnet_simple = set(t.lower() for t in extract_simple(content, "claude-sonnet"))

    # Keep canonical form from Haiku
    haiku_terms = extract_simple(content, "claude-haiku")
    result = [t for t in haiku_terms if t.lower() in sonnet_simple]

    # Add Sonnet terms that Haiku agreed with
    sonnet_terms = extract_simple(content, "claude-sonnet")
    for term in sonnet_terms:
        if term.lower() in haiku_simple and term not in result:
            result.append(term)

    return result


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

    # Fuzzy match
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True

    # Token overlap for multi-word terms
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True

    return False


def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
    """Calculate precision, recall, hallucination."""
    gt_terms = [t.get("term", "") for t in ground_truth]

    matched_gt = set()
    matched_ext = set()
    tp = 0

    for i, ext in enumerate(extracted):
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_terms(ext, gt):
                matched_gt.add(j)
                matched_ext.add(i)
                tp += 1
                break

    fp = len(extracted) - tp
    fn = len(gt_terms) - tp

    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    missed = [gt_terms[i] for i in range(len(gt_terms)) if i not in matched_gt]
    false_pos = [extracted[i] for i in range(len(extracted)) if i not in matched_ext]

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
        "missed": missed,
        "false_positives": false_pos,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment(num_chunks: int = 10):
    """Run combined strategy experiments."""

    # Load ground truth
    ground_truth = load_ground_truth()
    test_chunks = ground_truth[:num_chunks]

    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)
    avg_words = sum(len(c["content"].split()) for c in test_chunks) / len(test_chunks)
    avg_terms = sum(c["term_count"] for c in test_chunks) / len(test_chunks)
    print(f"  Avg chunk size: {avg_words:.0f} words", flush=True)
    print(f"  Avg terms/chunk: {avg_terms:.1f}", flush=True)

    # Define strategies
    strategies = {
        # Baselines
        "simple_haiku": lambda c: extract_simple(c, "claude-haiku"),
        "quote_haiku": lambda c: extract_quote(c, "claude-haiku"),
        "exhaustive_haiku": lambda c: extract_exhaustive(c, "claude-haiku"),
        "conservative_haiku": lambda c: extract_conservative(c, "claude-haiku"),
        # Combined strategies
        "ensemble_verified": lambda c: strategy_ensemble_verified(c, "claude-haiku"),
        "intersection_vote2": lambda c: strategy_intersection_voting(
            c, "claude-haiku", min_votes=2
        ),
        "intersection_vote3": lambda c: strategy_intersection_voting(
            c, "claude-haiku", min_votes=3
        ),
        "rated_essential": lambda c: strategy_rated_extraction(
            c, "claude-haiku", max_rating=1
        ),
        "rated_important": lambda c: strategy_rated_extraction(
            c, "claude-haiku", max_rating=2
        ),
        "conservative_union": lambda c: strategy_conservative_union(c),
        "quote_exhaust_intersect": lambda c: strategy_quote_exhaustive_intersect(
            c, "claude-haiku"
        ),
        "multi_model_consensus": lambda c: strategy_multi_model_consensus(c),
    }

    # Run experiments
    results = {name: [] for name in strategies}

    print(f"\n{'=' * 70}", flush=True)
    print("RUNNING COMBINED STRATEGY EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)

    for chunk in test_chunks:
        print(
            f"\n{chunk['chunk_id']} (GT: {chunk['term_count']} terms):",
            flush=True,
        )

        for name, extractor in strategies.items():
            try:
                start = time.time()
                extracted = extractor(chunk["content"])
                elapsed = time.time() - start

                metrics = calculate_metrics(extracted, chunk["terms"])
                metrics["elapsed"] = elapsed
                results[name].append(metrics)

                # Markers for targets
                r_mark = (
                    "✓"
                    if metrics["recall"] >= 0.95
                    else ("~" if metrics["recall"] >= 0.85 else " ")
                )
                h_mark = (
                    "✓"
                    if metrics["hallucination"] < 0.10
                    else ("~" if metrics["hallucination"] < 0.20 else " ")
                )

                print(
                    f"  {name:<25}: R={metrics['recall']:>5.0%}{r_mark} H={metrics['hallucination']:>5.0%}{h_mark} ({metrics['extracted_count']:>2} ext, {elapsed:.1f}s)",
                    flush=True,
                )

            except Exception as e:
                print(f"  {name:<25}: ERROR - {e}", flush=True)

    # Aggregate results
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)

    print(
        f"{'Strategy':<27} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 70, flush=True)

    summary = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            continue

        avg_p = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_r = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_h = sum(m["hallucination"] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)

        r_mark = "✓" if avg_r >= 0.95 else ("~" if avg_r >= 0.85 else "  ")
        h_mark = "✓" if avg_h < 0.10 else ("~" if avg_h < 0.20 else "  ")

        print(
            f"{name:<27} {avg_p:>10.1%} {avg_r:>8.1%} {r_mark} {avg_h:>8.1%} {h_mark} {avg_f1:>7.1%}",
            flush=True,
        )

        summary[name] = {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
        }

    print(f"\nTargets: Recall 95%+ [✓], Hallucination <10% [✓]", flush=True)
    print(f"Close:   Recall 85%+ [~],  Hallucination <20% [~]", flush=True)

    # Find best strategy meeting both targets
    print(f"\n{'=' * 70}", flush=True)
    print("BEST STRATEGIES", flush=True)
    print("=" * 70, flush=True)

    # Sort by recall (descending), then by hallucination (ascending)
    sorted_strategies = sorted(
        summary.items(), key=lambda x: (-x[1]["recall"], x[1]["hallucination"])
    )

    # Find strategies meeting targets
    meeting_both = [
        (name, metrics)
        for name, metrics in sorted_strategies
        if metrics["recall"] >= 0.95 and metrics["hallucination"] < 0.10
    ]

    if meeting_both:
        print(
            f"\n✅ Strategies meeting BOTH targets (95%+ recall, <10% hallucination):"
        )
        for name, metrics in meeting_both:
            print(
                f"   {name}: R={metrics['recall']:.1%}, H={metrics['hallucination']:.1%}"
            )
    else:
        print(f"\n❌ No strategy met both targets. Best candidates:")
        for name, metrics in sorted_strategies[:5]:
            print(
                f"   {name}: R={metrics['recall']:.1%}, H={metrics['hallucination']:.1%}"
            )

    # Save results
    results_path = ARTIFACTS_DIR / "combined_strategy_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=5)
