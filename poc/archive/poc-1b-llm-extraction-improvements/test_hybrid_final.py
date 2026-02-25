#!/usr/bin/env python3
"""Final hybrid strategy attempt to hit both targets.

Current best:
- ensemble_verified: 88.9% recall, 10.7% hallucination (very close!)
- vote_3: 75.1% recall, 2.2% hallucination

Strategy: Combine high-recall extraction with multi-stage verification.

New approaches:
1. Ensemble + Sonnet verification (use smarter model for filtering)
2. Ensemble + vote threshold (keep terms from ensemble that also appear in vote_2)
3. Multi-pass with Opus verification
4. Exhaustive Sonnet + Haiku verification
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

print("POC-1b: Hybrid Final Strategy Test", flush=True)
print("=" * 70, flush=True)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]


def strict_span_verify(term, content):
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


def parse_terms(response, require_quotes=False):
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


# Prompts
SIMPLE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates, lifecycle stages, CLI flags, API terms.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

QUOTE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
For EACH term, provide the exact quote where it appears.
CHUNK:
{content}
Output JSON: {{"terms": [{{"quote": "exact text", "term": "Term"}}]}}"""

CONSERVATIVE_PROMPT = """Extract ONLY the most important Kubernetes technical terms.
Be CONSERVATIVE. Only core resources, key components, essential concepts.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

# More precise verification prompt
STRICT_VERIFY_PROMPT = """You are a Kubernetes documentation expert filtering extracted terms.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS:
{terms}

Your task: Keep ONLY terms that meet ALL criteria:
1. Is a Kubernetes-specific technical term (not generic English)
2. Would be valuable in a documentation search index
3. Represents a specific K8s concept, resource, component, or feature

STRICTLY REMOVE:
- Generic words (memory, network, process, information, section)
- YAML/JSON structural keywords (title, stages, value, content)
- Version numbers or dates
- File paths, URLs
- Common verbs/adjectives used in tech writing

Be CONSERVATIVE. When in doubt, REMOVE the term.

Output ONLY the filtered terms as JSON:
{{"terms": ["term1", "term2", ...]}}"""


def extract_simple(content, model="claude-haiku"):
    prompt = SIMPLE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_exhaustive(content, model="claude-haiku"):
    prompt = EXHAUSTIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_quote(content, model="claude-haiku"):
    prompt = QUOTE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    terms = parse_terms(response, require_quotes=True)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_conservative(content, model="claude-haiku"):
    prompt = CONSERVATIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


# ============================================================================
# HYBRID STRATEGIES
# ============================================================================


def strategy_ensemble_sonnet_verify(content):
    """Ensemble (Haiku) + Sonnet verification."""
    all_terms = set()
    all_terms.update(extract_simple(content, "claude-haiku"))
    all_terms.update(extract_quote(content, "claude-haiku"))
    all_terms.update(extract_exhaustive(content, "claude-haiku"))

    if not all_terms:
        return []

    # Verify with Sonnet (smarter model)
    prompt = STRICT_VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-sonnet", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_sonnet_exhaustive_haiku_verify(content):
    """Exhaustive Sonnet + Haiku verification."""
    # High-recall extraction with Sonnet
    terms = extract_exhaustive(content, "claude-sonnet")

    if not terms:
        return []

    # Verify with Haiku
    prompt = STRICT_VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(terms)
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_ensemble_vote_hybrid(content):
    """Ensemble but weighted by vote count."""
    # Get all extractions
    simple = extract_simple(content, "claude-haiku")
    quote = extract_quote(content, "claude-haiku")
    exhaustive = extract_exhaustive(content, "claude-haiku")
    conservative = extract_conservative(content, "claude-haiku")

    # Count votes
    term_votes = {}
    for term_set in [simple, quote, exhaustive, conservative]:
        for term in term_set:
            key = term.lower()
            if key not in term_votes:
                term_votes[key] = {
                    "canonical": term,
                    "votes": 0,
                    "in_conservative": False,
                }
            term_votes[key]["votes"] += 1

    # Mark conservative terms
    for term in conservative:
        key = term.lower()
        if key in term_votes:
            term_votes[key]["in_conservative"] = True

    # Keep terms with 2+ votes OR in conservative
    result = [
        info["canonical"]
        for info in term_votes.values()
        if info["votes"] >= 2 or info["in_conservative"]
    ]

    return result


def strategy_multi_extraction_consensus(content):
    """Run same extraction 3x, keep consensus terms."""
    # Run simple extraction 3 times with slight prompt variations
    extractions = []

    for i in range(3):
        terms = extract_simple(content, "claude-haiku")
        extractions.append(set(t.lower() for t in terms))

    # Also get exhaustive
    exhaustive = set(t.lower() for t in extract_exhaustive(content, "claude-haiku"))
    extractions.append(exhaustive)

    # Get canonical forms from first extraction
    canonical = {t.lower(): t for t in extract_simple(content, "claude-haiku")}
    canonical.update(
        {t.lower(): t for t in extract_exhaustive(content, "claude-haiku")}
    )

    # Keep terms appearing in 3+ extractions
    term_counts = {}
    for ext_set in extractions:
        for term in ext_set:
            term_counts[term] = term_counts.get(term, 0) + 1

    result = [canonical.get(t, t) for t, count in term_counts.items() if count >= 3]
    return result


def strategy_union_verified(content):
    """Union of high-precision strategies + verification."""
    # High precision: quote + conservative
    all_terms = set()
    all_terms.update(extract_quote(content, "claude-haiku"))
    all_terms.update(extract_conservative(content, "claude-haiku"))
    all_terms.update(extract_conservative(content, "claude-sonnet"))

    if not all_terms:
        return []

    # Light verification
    prompt = STRICT_VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]


def strategy_exhaustive_double_verify(content):
    """Exhaustive extraction + double verification."""
    # Step 1: High-recall extraction
    all_terms = set()
    all_terms.update(extract_exhaustive(content, "claude-haiku"))
    all_terms.update(extract_exhaustive(content, "claude-sonnet"))

    if not all_terms:
        return []

    # Step 2: First verification with Haiku
    prompt = STRICT_VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    first_verified = parse_terms(response)

    if not first_verified:
        return []

    # Step 3: Second verification with Sonnet
    prompt = STRICT_VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(first_verified)
    )
    response = call_llm(prompt, model="claude-sonnet", temperature=0, max_tokens=1000)
    final = parse_terms(response)

    return [t for t in final if strict_span_verify(t, content)]


# ============================================================================
# METRICS
# ============================================================================


def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted, ground_truth):
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


def calculate_metrics(extracted, ground_truth):
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


def run_experiment(num_chunks=8):
    ground_truth = load_ground_truth()
    test_chunks = ground_truth[:num_chunks]

    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)

    strategies = {
        # Baselines for comparison
        "exhaustive_haiku": lambda c: extract_exhaustive(c, "claude-haiku"),
        "exhaustive_sonnet": lambda c: extract_exhaustive(c, "claude-sonnet"),
        # New hybrid strategies
        "ensemble_sonnet_verify": strategy_ensemble_sonnet_verify,
        "sonnet_exh_haiku_verify": strategy_sonnet_exhaustive_haiku_verify,
        "ensemble_vote_hybrid": strategy_ensemble_vote_hybrid,
        "multi_consensus": strategy_multi_extraction_consensus,
        "union_verified": strategy_union_verified,
        "exhaustive_double_verify": strategy_exhaustive_double_verify,
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
                    f"  {name:<25}: R={metrics['recall']:>5.0%}{r_mark} H={metrics['hallucination']:>5.0%}{h_mark} ({metrics['extracted_count']:>2} ext, {elapsed:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<25}: ERROR - {e}", flush=True)

    # Aggregate
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Strategy':<27} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 68, flush=True)

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

    # Check targets
    print(f"\n{'=' * 70}", flush=True)
    print("TARGET CHECK: 95%+ recall AND <10% hallucination", flush=True)
    print("=" * 70, flush=True)

    meeting_both = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.95 and m["hallucination"] < 0.10
    ]
    close_recall = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.90 and m["hallucination"] < 0.15
    ]

    if meeting_both:
        print("✅ STRATEGIES MEETING BOTH TARGETS:")
        for name, m in sorted(meeting_both, key=lambda x: -x[1]["recall"]):
            print(
                f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}, F1={m['f1']:.1%}"
            )
    else:
        print("❌ No strategy met both targets.")
        if close_recall:
            print("\n Close candidates (90%+ recall, <15% hallucination):")
            for name, m in sorted(
                close_recall, key=lambda x: (-x[1]["recall"], x[1]["hallucination"])
            ):
                print(f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}")

    # Save
    results_path = ARTIFACTS_DIR / "hybrid_final_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {results_path}", flush=True)

    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=8)
