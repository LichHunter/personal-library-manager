#!/usr/bin/env python3
"""Fast test of key combined strategies.

Based on initial results, the most promising strategies are:
- ensemble_verified: High recall with LLM filtering
- intersection_vote3: High precision through voting
- rated_essential: Good balance through importance rating

This script tests fewer strategies on more chunks for better statistical significance.
"""

import json
import re
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

# Add paths
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)

from utils.llm_provider import call_llm

print("POC-1b: Fast Combined Strategy Test", flush=True)
print("=" * 70, flush=True)

# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]


def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content."""
    if not term or len(term) < 2:
        return False
    content_lower = content.lower()
    term_lower = term.lower().strip()
    if term_lower in content_lower:
        return True
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True
    return False


def parse_terms(response: str, require_quotes: bool = False) -> list[str]:
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms = data.get("terms", [])
            if require_quotes:
                return [t.get("term", "") for t in terms if isinstance(t, dict)]
            elif isinstance(terms, list):
                if terms and isinstance(terms[0], dict):
                    return [t.get("term", "") for t in terms]
                return [str(t) for t in terms]
    except:
        pass
    return []


def parse_ratings(response: str) -> dict[str, int]:
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
    except:
        pass
    return {}


# ============================================================================
# PROMPTS
# ============================================================================

SIMPLE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

QUOTE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
For EACH term, provide the exact quote where it appears.
CHUNK:
{content}
Output JSON: {{"terms": [{{"quote": "exact text", "term": "Term"}}]}}"""

EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates, lifecycle stages, CLI flags, API terms.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

CONSERVATIVE_PROMPT = """Extract ONLY the most important Kubernetes technical terms.
Be CONSERVATIVE. Only core resources, key components, essential concepts.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

VERIFY_PROMPT = """Filter this list of extracted terms from Kubernetes docs.
Keep ONLY Kubernetes-specific technical terms. Remove generic English words.
CHUNK: {content}
TERMS: {terms}
Output JSON: {{"terms": ["term1", ...]}}"""

RATE_PROMPT = """Rate each term's importance for K8s documentation search.
CHUNK: {content}
TERMS: {terms}
Ratings: 1=ESSENTIAL (core K8s), 2=IMPORTANT (should index), 3=OPTIONAL, 4=NOISE
Output JSON: {{"ratings": [{{"term": "X", "rating": 1}}]}}"""


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def extract_simple(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = SIMPLE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_quote(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = QUOTE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    terms = parse_terms(response, require_quotes=True)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_exhaustive(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = EXHAUSTIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_conservative(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = CONSERVATIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


# ============================================================================
# COMBINED STRATEGIES
# ============================================================================


def strategy_ensemble_verified(content: str) -> list[str]:
    """High recall ensemble + LLM verification."""
    all_terms = set()
    all_terms.update(extract_simple(content))
    all_terms.update(extract_quote(content))
    all_terms.update(extract_exhaustive(content))

    if not all_terms:
        return []

    prompt = VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_intersection_vote(content: str, min_votes: int = 2) -> list[str]:
    """Keep terms with multiple strategy votes."""
    simple = set(extract_simple(content))
    quote = set(extract_quote(content))
    exhaustive = set(extract_exhaustive(content))
    conservative = set(extract_conservative(content))

    term_votes = {}
    for term_set in [simple, quote, exhaustive, conservative]:
        for term in term_set:
            key = term.lower()
            if key not in term_votes:
                term_votes[key] = {"canonical": term, "votes": 0}
            term_votes[key]["votes"] += 1

    return [
        info["canonical"] for info in term_votes.values() if info["votes"] >= min_votes
    ]


def strategy_rated(content: str, max_rating: int = 2) -> list[str]:
    """Extract liberally, rate, keep important."""
    all_terms = set()
    all_terms.update(extract_simple(content))
    all_terms.update(extract_exhaustive(content))

    if not all_terms:
        return []

    prompt = RATE_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1500)
    ratings = parse_ratings(response)

    return [t for t in all_terms if ratings.get(t, 3) <= max_rating]


def strategy_sonnet_conservative(content: str) -> list[str]:
    """Use Sonnet for more accurate conservative extraction."""
    return extract_conservative(content, "claude-sonnet")


def strategy_quote_verified(content: str) -> list[str]:
    """Quote extraction (high precision) + verification."""
    quote_terms = extract_quote(content)
    if not quote_terms:
        return []

    prompt = VERIFY_PROMPT.format(content=content[:2000], terms=json.dumps(quote_terms))
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_union_conservative(content: str) -> list[str]:
    """Union of conservative extractions from multiple approaches."""
    cons_haiku = set(extract_conservative(content, "claude-haiku"))
    cons_sonnet = set(extract_conservative(content, "claude-sonnet"))
    quote_terms = set(extract_quote(content))

    # Return union - conservative should have low hallucination
    return list(cons_haiku | cons_sonnet)


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
    return False


def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
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


def run_experiment(num_chunks: int = 10):
    ground_truth = load_ground_truth()
    test_chunks = ground_truth[:num_chunks]

    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)

    strategies = {
        # Baselines
        "simple_haiku": lambda c: extract_simple(c),
        "quote_haiku": lambda c: extract_quote(c),
        "exhaustive_haiku": lambda c: extract_exhaustive(c),
        # Combined - targeting high recall + low hallucination
        "ensemble_verified": strategy_ensemble_verified,
        "vote_2": lambda c: strategy_intersection_vote(c, 2),
        "vote_3": lambda c: strategy_intersection_vote(c, 3),
        "rated_important": lambda c: strategy_rated(c, 2),
        "sonnet_conservative": strategy_sonnet_conservative,
        "quote_verified": strategy_quote_verified,
        "union_conservative": strategy_union_conservative,
    }

    results = {name: [] for name in strategies}

    print(f"\n{'=' * 70}", flush=True)

    for i, chunk in enumerate(test_chunks):
        print(
            f"\n[{i + 1}/{len(test_chunks)}] {chunk['chunk_id']} (GT: {chunk['term_count']} terms)",
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
                    f"  {name:<20}: R={metrics['recall']:>5.0%}{r_mark} H={metrics['hallucination']:>5.0%}{h_mark} ({metrics['extracted_count']:>2} ext)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<20}: ERROR - {e}", flush=True)

    # Aggregate
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Strategy':<22} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 55, flush=True)

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
            f"{name:<22} {avg_p:>10.1%} {avg_r:>8.1%} {r_mark} {avg_h:>8.1%} {h_mark} {avg_f1:>7.1%}",
            flush=True,
        )
        summary[name] = {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
        }

    # Best strategies
    print(f"\n{'=' * 70}", flush=True)
    print("TARGET CHECK: 95%+ recall AND <10% hallucination", flush=True)
    print("=" * 70, flush=True)

    meeting_both = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.95 and m["hallucination"] < 0.10
    ]
    close_to_both = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.85 and m["hallucination"] < 0.20
    ]

    if meeting_both:
        print("✅ STRATEGIES MEETING BOTH TARGETS:")
        for name, m in sorted(meeting_both, key=lambda x: -x[1]["recall"]):
            print(
                f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}, F1={m['f1']:.1%}"
            )
    else:
        print("❌ No strategy met both targets.")
        print("\nClose candidates (85%+ recall, <20% hallucination):")
        for name, m in sorted(
            close_to_both, key=lambda x: (-x[1]["recall"], x[1]["hallucination"])
        )[:5]:
            print(f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}")

    # Save
    results_path = ARTIFACTS_DIR / "fast_combined_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {results_path}", flush=True)

    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=5)
