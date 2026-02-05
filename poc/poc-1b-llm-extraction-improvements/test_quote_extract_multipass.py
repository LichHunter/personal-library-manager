#!/usr/bin/env python3
"""Test Quote-Extract with multi-pass gleaning and strict span verification.

Key insight from experiments:
- Quote-Extract achieves 92.6% precision, 7.4% hallucination (BEST)
- But only 53.7% recall (PROBLEM)
- Quote-Verify boosts recall to 82.5% but kills precision

Solution: Multi-pass Quote-Extract with:
1. Initial Quote-Extract (high precision base)
2. Category-specific Quote-Extract passes (targeted recall boost)
3. STRICT span verification after EVERY pass (prevent hallucination cascade)

Target: 90%+ precision, 80%+ recall, <5% hallucination
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

print("POC-1b: Quote-Extract Multi-Pass", flush=True)
print("=" * 70, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# STRICT SPAN VERIFICATION
# ============================================================================


def strict_span_verify(term: str, quote: str, source_text: str) -> bool:
    """Strictly verify that both term and quote exist in source."""
    source_lower = source_text.lower()

    # Quote must exist (fuzzy match for minor variations)
    if quote:
        quote_lower = quote.lower().strip()
        if len(quote_lower) >= 5:
            if quote_lower not in source_lower:
                # Try fuzzy partial match
                if fuzz.partial_ratio(quote_lower, source_lower) < 85:
                    return False

    # Term must exist in source
    term_lower = term.lower().strip()
    if term_lower not in source_lower:
        # Handle CamelCase → space separated
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
        if spaced not in source_lower:
            return False

    return True


# ============================================================================
# QUOTE-EXTRACT PROMPTS
# ============================================================================

INITIAL_EXTRACT_PROMPT = """You are a Kubernetes documentation expert.

TASK: Extract ALL Kubernetes-specific terms from the text below.

For EACH term you extract:
1. QUOTE: Copy the EXACT text (5-30 words) from the source containing the term
2. TERM: The normalized Kubernetes term
3. TYPE: One of [resource, component, concept, feature_gate, lifecycle]

EXTRACT:
- Resources: Pod, Deployment, Service, ConfigMap, Secret, Ingress, etc.
- Components: kubelet, kubectl, kube-proxy, etcd, etc.
- Concepts: namespace, label, selector, annotation, controller, webhook, etc.
- Feature gates: CamelCase names like ServiceAppProtocol, JobPodFailurePolicy
- Lifecycle: alpha, beta, stable, GA, deprecated (when describing K8s features)

DO NOT EXTRACT:
- YAML metadata keywords (title, id, date, stages, defaultValue, fromVersion)
- Generic English words unless K8s-specific in context
- Version numbers alone (1.18, 1.19)
- File paths or URLs

<text>
{chunk_text}
</text>

Output ONLY valid JSON:
{{"entities": [
  {{"quote": "exact text from source containing term", "term": "TermName", "type": "resource"}}
]}}"""


CATEGORY_PROMPTS = {
    "resources": """Extract ONLY Kubernetes RESOURCE TYPES from this text.
Resource types include: Pod, Service, Deployment, ConfigMap, Secret, Ingress, 
PersistentVolume, StatefulSet, DaemonSet, ReplicaSet, Job, CronJob, etc.

For EACH resource found:
1. QUOTE: Exact text (5-30 words) where it appears
2. TERM: The resource name

<text>
{text}
</text>

Output JSON: {{"entities": [{{"quote": "...", "term": "..."}}]}}""",
    "components": """Extract ONLY Kubernetes COMPONENT/PROCESS names from this text.
Components include: kubelet, kubectl, kubeadm, kube-proxy, kube-apiserver, 
kube-scheduler, kube-controller-manager, etcd, coredns, containerd, etc.

For EACH component found:
1. QUOTE: Exact text (5-30 words) where it appears
2. TERM: The component name

<text>
{text}
</text>

Output JSON: {{"entities": [{{"quote": "...", "term": "..."}}]}}""",
    "feature_gates": """Extract ONLY Kubernetes FEATURE GATE names from this text.
Feature gates are CamelCase names like: ServiceAppProtocol, JobPodFailurePolicy,
CSIVolumeHealth, CSINodeInfo, etc.

For EACH feature gate found:
1. QUOTE: Exact text (5-30 words) where it appears
2. TERM: The feature gate name

<text>
{text}
</text>

Output JSON: {{"entities": [{{"quote": "...", "term": "..."}}]}}""",
    "lifecycle": """Extract ONLY Kubernetes LIFECYCLE/MATURITY terms from this text.
Lifecycle terms include: alpha, beta, stable, GA, deprecated, removed
(ONLY when describing Kubernetes feature maturity, not general usage)

For EACH lifecycle term found:
1. QUOTE: Exact text (5-30 words) showing it describes K8s feature maturity
2. TERM: The lifecycle stage

<text>
{text}
</text>

Output JSON: {{"entities": [{{"quote": "...", "term": "..."}}]}}""",
    "concepts": """Extract ONLY Kubernetes CONCEPTS and API TERMS from this text.
Concepts include: namespace, label, selector, annotation, taint, toleration,
affinity, controller, operator, webhook, admission, finalizer, object, spec, 
status, watch, stream, etc.

For EACH concept found:
1. QUOTE: Exact text (5-30 words) where it appears as K8s concept
2. TERM: The concept name

<text>
{text}
</text>

Output JSON: {{"entities": [{{"quote": "...", "term": "..."}}]}}""",
}


GLEANING_PROMPT = """You previously extracted these Kubernetes terms: {previous_terms}

Review the text again. What Kubernetes terms did you MISS?

Focus on:
- Feature gate names (CamelCase like ServiceAppProtocol)
- Component names (kubelet, etcd, etc.)
- Resource types (Pod, Service, etc.)
- Lifecycle stages (alpha, beta, stable) describing K8s features
- API concepts (Watch, controller, object, etc.)

For EACH missed term:
1. QUOTE: Exact text (5-30 words) where it appears
2. TERM: The term name

<text>
{text}
</text>

Output JSON: {{"missed": [{{"quote": "...", "term": "..."}}]}}
If nothing was missed, return: {{"missed": []}}"""


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def parse_extraction_response(response: str) -> list[dict]:
    """Parse JSON response from extraction prompt."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("entities", data.get("missed", []))
    except (json.JSONDecodeError, KeyError):
        pass
    return []


def extract_initial(text: str, model: str = "claude-haiku") -> list[dict]:
    """Initial comprehensive extraction pass."""
    prompt = INITIAL_EXTRACT_PROMPT.format(chunk_text=text[:3500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)
    return parse_extraction_response(response)


def extract_category(
    text: str, category: str, model: str = "claude-haiku"
) -> list[dict]:
    """Extract terms of a specific category."""
    if category not in CATEGORY_PROMPTS:
        return []

    prompt = CATEGORY_PROMPTS[category].format(text=text[:3000])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=500)
    return parse_extraction_response(response)


def extract_gleaning(
    text: str, existing_terms: list[str], model: str = "claude-haiku"
) -> list[dict]:
    """Gleaning pass to find missed terms."""
    prompt = GLEANING_PROMPT.format(
        previous_terms=json.dumps(existing_terms),
        text=text[:3000],
    )
    response = call_llm(prompt, model=model, temperature=0, max_tokens=500)
    return parse_extraction_response(response)


# ============================================================================
# MULTI-PASS PIPELINE
# ============================================================================


def extract_multipass_quote(
    text: str,
    model: str = "claude-haiku",
    categories: list[str] = None,
    use_gleaning: bool = True,
) -> dict:
    """Multi-pass Quote-Extract with strict verification after each pass."""

    if categories is None:
        categories = [
            "resources",
            "components",
            "feature_gates",
            "lifecycle",
            "concepts",
        ]

    all_terms = set()
    verified_entities = []

    # Pass 1: Initial comprehensive extraction
    initial_results = extract_initial(text, model)
    for entity in initial_results:
        term = entity.get("term", "")
        quote = entity.get("quote", "")
        if term and strict_span_verify(term, quote, text):
            if term.lower() not in {t.lower() for t in all_terms}:
                all_terms.add(term)
                verified_entities.append(
                    {"term": term, "quote": quote, "source": "initial"}
                )

    initial_count = len(all_terms)

    # Pass 2: Category-specific passes
    category_count = 0
    for category in categories:
        cat_results = extract_category(text, category, model)
        for entity in cat_results:
            term = entity.get("term", "")
            quote = entity.get("quote", "")
            if term and strict_span_verify(term, quote, text):
                if term.lower() not in {t.lower() for t in all_terms}:
                    all_terms.add(term)
                    verified_entities.append(
                        {"term": term, "quote": quote, "source": f"category:{category}"}
                    )
                    category_count += 1

    # Pass 3: Gleaning (optional)
    gleaning_count = 0
    if use_gleaning:
        gleaning_results = extract_gleaning(text, list(all_terms), model)
        for entity in gleaning_results:
            term = entity.get("term", "")
            quote = entity.get("quote", "")
            if term and strict_span_verify(term, quote, text):
                if term.lower() not in {t.lower() for t in all_terms}:
                    all_terms.add(term)
                    verified_entities.append(
                        {"term": term, "quote": quote, "source": "gleaning"}
                    )
                    gleaning_count += 1

    return {
        "final_terms": list(all_terms),
        "verified_entities": verified_entities,
        "initial_count": initial_count,
        "category_count": category_count,
        "gleaning_count": gleaning_count,
    }


# ============================================================================
# METRICS
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted: str, ground_truth: str) -> str:
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    if ext_norm == gt_norm:
        return "exact"

    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return "partial"

    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return "fuzzy"

    return "no_match"


def calculate_metrics(extracted_terms: list[str], gt_terms: list[dict]) -> dict:
    gt_list = [t.get("term", "") for t in gt_terms]
    gt_tiers = {t.get("term", ""): t.get("tier", 1) for t in gt_terms}

    matched_gt = set()
    matched_extracted = set()
    tp = 0

    for i, ext in enumerate(extracted_terms):
        for j, gt in enumerate(gt_list):
            if j in matched_gt:
                continue
            if match_terms(ext, gt) != "no_match":
                matched_gt.add(j)
                matched_extracted.add(i)
                tp += 1
                break

    fp = len(extracted_terms) - tp
    fn = len(gt_list) - len(matched_gt)

    precision = tp / len(extracted_terms) if extracted_terms else 0
    recall = tp / len(gt_list) if gt_list else 0
    hallucination = fp / len(extracted_terms) if extracted_terms else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    missed = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
    false_positives = [
        extracted_terms[i]
        for i in range(len(extracted_terms))
        if i not in matched_extracted
    ]

    # Tier breakdown
    tier_metrics = {}
    for tier in [1, 2, 3]:
        tier_gt = [
            gt_list[i]
            for i in range(len(gt_list))
            if gt_tiers.get(gt_list[i], 1) == tier
        ]
        tier_matched = sum(1 for i in matched_gt if gt_tiers.get(gt_list[i], 1) == tier)
        if tier_gt:
            tier_metrics[f"tier_{tier}_recall"] = tier_matched / len(tier_gt)
            tier_metrics[f"tier_{tier}_count"] = len(tier_gt)

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted": len(extracted_terms),
        "ground_truth": len(gt_list),
        "missed": missed,
        "false_positives": false_positives,
        "tier_metrics": tier_metrics,
    }


# ============================================================================
# EXPERIMENTS
# ============================================================================


def run_multipass_experiment(model: str = "claude-haiku", num_chunks: int = 20):
    """Run multi-pass Quote-Extract experiment."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"MULTI-PASS QUOTE-EXTRACT ({model}, {num_chunks} chunks)", flush=True)
    print("=" * 70, flush=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunks = gt["chunks"][:num_chunks]
    all_metrics = []

    for chunk in chunks:
        start = time.time()
        result = extract_multipass_quote(chunk["text"], model=model)
        elapsed = time.time() - start

        metrics = calculate_metrics(result["final_terms"], chunk["terms"])
        all_metrics.append(metrics)

        p_mark = (
            "OK"
            if metrics["precision"] >= 0.95
            else ("~" if metrics["precision"] >= 0.85 else "")
        )
        r_mark = (
            "OK"
            if metrics["recall"] >= 0.95
            else ("~" if metrics["recall"] >= 0.70 else "")
        )
        h_mark = (
            "OK"
            if metrics["hallucination"] < 0.01
            else ("~" if metrics["hallucination"] < 0.05 else "")
        )

        print(
            f"  {chunk['chunk_id']}: P={metrics['precision']:.0%}{p_mark} R={metrics['recall']:.0%}{r_mark} H={metrics['hallucination']:.0%}{h_mark} "
            f"({elapsed:.1f}s, init={result['initial_count']}, cat={result['category_count']}, glean={result['gleaning_count']})",
            flush=True,
        )

        if metrics["missed"] and len(metrics["missed"]) <= 5:
            print(f"    Missed: {metrics['missed']}", flush=True)
        if metrics["false_positives"] and len(metrics["false_positives"]) <= 3:
            print(f"    FP: {metrics['false_positives']}", flush=True)

    # Aggregate
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    # Tier aggregation
    tier_recalls = {1: [], 2: [], 3: []}
    for m in all_metrics:
        for tier in [1, 2, 3]:
            key = f"tier_{tier}_recall"
            if key in m.get("tier_metrics", {}):
                tier_recalls[tier].append(m["tier_metrics"][key])

    print(f"\n{'=' * 70}", flush=True)
    print(f"RESULTS: Multi-Pass Quote-Extract ({model})", flush=True)
    print("=" * 70, flush=True)

    p_mark = "OK" if avg_p >= 0.95 else ("~" if avg_p >= 0.85 else "")
    r_mark = "OK" if avg_r >= 0.95 else ("~" if avg_r >= 0.70 else "")
    h_mark = "OK" if avg_h < 0.01 else ("~" if avg_h < 0.05 else "")

    print(f"  Precision:     {avg_p:.1%} {p_mark}", flush=True)
    print(f"  Recall:        {avg_r:.1%} {r_mark}", flush=True)
    print(f"  Hallucination: {avg_h:.1%} {h_mark}", flush=True)
    print(f"  F1:            {avg_f1:.1%}", flush=True)

    print(f"\n  Recall by Tier:", flush=True)
    for tier in [1, 2, 3]:
        if tier_recalls[tier]:
            avg_tier = sum(tier_recalls[tier]) / len(tier_recalls[tier])
            print(f"    Tier {tier}: {avg_tier:.1%}", flush=True)

    return {
        "precision": avg_p,
        "recall": avg_r,
        "hallucination": avg_h,
        "f1": avg_f1,
        "tier_recalls": {
            k: sum(v) / len(v) if v else 0 for k, v in tier_recalls.items()
        },
    }


def run_all_experiments():
    """Run all experiments and compare."""
    print("\n" + "=" * 70, flush=True)
    print("MULTI-PASS QUOTE-EXTRACT EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    print("\nTarget: P>90%, R>80%, H<5%", flush=True)

    results = {}

    # Haiku full experiment
    results["multipass_haiku"] = run_multipass_experiment(
        model="claude-haiku", num_chunks=20
    )

    # Sonnet (fewer due to cost)
    results["multipass_sonnet"] = run_multipass_experiment(
        model="claude-sonnet", num_chunks=10
    )

    # Final comparison
    print("\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON (All Approaches)", flush=True)
    print("=" * 70, flush=True)

    previous = {
        "quote_extract_original": {
            "precision": 0.926,
            "recall": 0.537,
            "hallucination": 0.074,
        },
        "gleaning_2x": {"precision": 0.627, "recall": 0.684, "hallucination": 0.323},
        "combined_sonnet": {
            "precision": 0.857,
            "recall": 0.609,
            "hallucination": 0.093,
        },
        "discrimination_full": {
            "precision": 0.640,
            "recall": 0.625,
            "hallucination": 0.310,
        },
        "quote_verify_haiku": {
            "precision": 0.408,
            "recall": 0.825,
            "hallucination": 0.592,
        },
    }

    all_results = {**previous, **results}

    print(
        f"{'Approach':<25} {'Precision':>12} {'Recall':>12} {'Hallucination':>14} {'F1':>8}",
        flush=True,
    )
    print("-" * 75, flush=True)

    for name, data in all_results.items():
        p = data.get("precision", 0)
        r = data.get("recall", 0)
        h = data.get("hallucination", 0)
        f1 = data.get("f1", 2 * p * r / (p + r) if (p + r) > 0 else 0)

        p_mark = "OK" if p >= 0.95 else ("~" if p >= 0.85 else "  ")
        r_mark = "OK" if r >= 0.95 else ("~" if r >= 0.70 else "  ")
        h_mark = "OK" if h < 0.01 else ("~" if h < 0.05 else "  ")

        print(
            f"{name:<25} {p:>10.1%} {p_mark} {r:>10.1%} {r_mark} {h:>12.1%} {h_mark} {f1:>7.1%}",
            flush=True,
        )

    # Save results
    results_path = (
        Path(__file__).parent / "artifacts" / "multipass_quote_extract_results.json"
    )
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return results


if __name__ == "__main__":
    run_all_experiments()
