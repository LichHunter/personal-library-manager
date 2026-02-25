#!/usr/bin/env python3
"""Test the Quote-Verify approach based on Oracle's analysis.

Key insight: Quote-Extract achieves 92.6% precision because it forces grounding
BEFORE extraction. We extend this to all verification:

Architecture:
1. EXHAUSTIVE CANDIDATE EXTRACTION: All possible terms (patterns + known vocabulary)
2. KNOWN-TERM VOCABULARY: Common K8s terms bypass LLM entirely (100% precision)
3. QUOTE-VERIFY: For each candidate, ask LLM to QUOTE where it appears as K8s concept
4. GAP-FILLING QUOTE-EXTRACT: One pass asking "what did I miss?" with quotes required

Target: 95%+ precision, 95%+ recall, <1% hallucination
Intermediate: 90%+ precision, 75%+ recall, <5% hallucination
"""

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Quote-Verify Approach", flush=True)
print("=" * 70, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# KNOWN KUBERNETES VOCABULARY (Bypass LLM - 100% precision)
# ============================================================================

# Common K8s terms that we can match directly without LLM
K8S_KNOWN_TERMS = {
    # Core resources (most common)
    "pod",
    "pods",
    "deployment",
    "deployments",
    "service",
    "services",
    "configmap",
    "configmaps",
    "secret",
    "secrets",
    "namespace",
    "namespaces",
    "node",
    "nodes",
    "container",
    "containers",
    "volume",
    "volumes",
    "ingress",
    "endpoint",
    "endpoints",
    "replicaset",
    "statefulset",
    "daemonset",
    "job",
    "jobs",
    "cronjob",
    "persistentvolume",
    "pv",
    "persistentvolumeclaim",
    "pvc",
    "serviceaccount",
    # Components
    "kubelet",
    "kubectl",
    "kubeadm",
    "kube-proxy",
    "kube-apiserver",
    "kube-scheduler",
    "kube-controller-manager",
    "etcd",
    "coredns",
    "kubernetes",
    "api server",
    "api-server",
    # Concepts
    "cluster",
    "label",
    "labels",
    "selector",
    "selectors",
    "annotation",
    "annotations",
    "taint",
    "taints",
    "toleration",
    "tolerations",
    "affinity",
    "replica",
    "replicas",
    "rollout",
    "rollback",
    "workload",
    "controller",
    "operator",
    "webhook",
    "admission",
    "finalizer",
    "object",
    "objects",
    "spec",
    "status",
    "metadata",
    # Lifecycle stages
    "alpha",
    "beta",
    "stable",
    "ga",
    "deprecated",
    "removed",
    # Important concepts
    "feature gate",
    "feature_gate",
    "api",
    "cni",
    "csi",
    "cri",
    "runtime",
    "watch",
    "stream",
}

# Normalize for matching
K8S_KNOWN_TERMS_NORMALIZED = {
    t.lower().replace("-", " ").replace("_", " "): t for t in K8S_KNOWN_TERMS
}


def find_known_terms_in_text(text: str) -> set[str]:
    """Find known K8s terms that appear in the text (case-insensitive)."""
    text_lower = text.lower()
    found = set()

    for normalized, original in K8S_KNOWN_TERMS_NORMALIZED.items():
        if normalized in text_lower:
            found.add(original)

    return found


# ============================================================================
# EXHAUSTIVE CANDIDATE EXTRACTION
# ============================================================================


def extract_all_candidates_exhaustive(text: str) -> set[str]:
    """Extract ALL possible candidates - be liberal, let verification filter."""
    candidates = set()

    # 1. Backticked terms (high confidence)
    backticked = re.findall(r"`([^`]+)`", text)
    for term in backticked:
        term = term.strip()
        if 2 <= len(term) <= 50 and not re.match(r"^[\d\.\-/]+$", term):
            candidates.add(term)

    # 2. CamelCase terms
    camelcase = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b", text)
    candidates.update(camelcase)

    # 3. Single capitalized words (Kubernetes, Pod, etc.)
    capitalized = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
    # Filter common English words that are often capitalized at sentence start
    common_sentence_starters = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "it",
        "they",
        "we",
        "you",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "for",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "any",
        "some",
        "no",
        "none",
        "one",
        "two",
        "three",
        "few",
        "many",
        "much",
        "more",
        "most",
        "other",
        "another",
        "such",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "if",
        "then",
        "else",
        "because",
        "although",
        "though",
        "unless",
        "until",
        "while",
        "as",
        "after",
        "before",
        "since",
        "during",
        "about",
        "against",
        "between",
        "into",
        "through",
        "above",
        "below",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "see",
        "use",
        "used",
        "using",
        "uses",
        "allow",
        "allows",
        "enable",
        "enables",
        "make",
        "makes",
        "made",
        "note",
        "example",
        "however",
        "therefore",
        "thus",
        "hence",
        "instead",
        "rather",
        "below",
        "above",
        "following",
        "previous",
        "next",
        "first",
        "second",
        "last",
        "new",
        "old",
        "different",
        "similar",
        "specific",
        "general",
        "common",
        "typical",
        "default",
        "current",
        "available",
        "possible",
        "required",
        "optional",
        "additional",
        "existing",
        "given",
        "provided",
        "based",
        "related",
        "associated",
        "corresponding",
        "respective",
        "relevant",
    }
    for word in capitalized:
        if word.lower() not in common_sentence_starters:
            candidates.add(word)

    # 4. YAML keys that might be K8s concepts (don't filter feature_gate!)
    # Only filter obviously non-K8s YAML metadata
    yaml_metadata_only = {
        "title",
        "content_type",
        "id",
        "date",
        "full_link",
        "short_description",
        "aka",
        "tags",
        "_build",
        "list",
        "render",
        "stages",
        "stage",
        "defaultvalue",
        "fromversion",
        "toversion",
        "removed",
    }
    yaml_keys = re.findall(r"^\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*:", text, re.MULTILINE)
    for key in yaml_keys:
        if key.lower() not in yaml_metadata_only:
            candidates.add(key)

    # 5. Terms after "glossary_tooltip" (these are always K8s terms)
    glossary_terms = re.findall(r'term_id="([^"]+)"', text)
    candidates.update(glossary_terms)
    glossary_text = re.findall(r'text="([^"]+)"', text)
    candidates.update(glossary_text)

    # 6. Hyphenated technical terms
    hyphenated = re.findall(r"\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b", text.lower())
    non_technical = {
        "built-in",
        "well-known",
        "so-called",
        "real-time",
        "up-to-date",
        "e-g",
        "i-e",
        "non-nil",
        "non-zero",
        "pre-existing",
    }
    for term in hyphenated:
        if term not in non_technical:
            candidates.add(term)

    # 7. Lowercase K8s terms that appear in text (pod, container, etc.)
    text_lower = text.lower()
    for known in K8S_KNOWN_TERMS:
        known_lower = known.lower()
        if known_lower in text_lower:
            candidates.add(known)

    return candidates


# ============================================================================
# QUOTE-VERIFY: Ask LLM to quote where term appears as K8s concept
# ============================================================================

QUOTE_VERIFY_PROMPT = """You are a Kubernetes documentation expert.

TASK: For each candidate term below, determine if it's used as a Kubernetes concept in the text.

For each term:
- If it IS a K8s concept in this text: provide the EXACT quote (5-30 words) where it appears in that context
- If it is NOT a K8s concept (generic word, YAML metadata, version number, etc.): respond "NOT_K8S"

CONTEXT (source document):
{context}

CANDIDATE TERMS TO VERIFY:
{candidates}

OUTPUT FORMAT (JSON array):
[
  {{"term": "Pod", "status": "K8S_CONCEPT", "quote": "containers are running in a Pod"}},
  {{"term": "title", "status": "NOT_K8S", "quote": null}},
  {{"term": "alpha", "status": "K8S_CONCEPT", "quote": "stage: alpha"}}
]

RULES:
- The quote MUST be an EXACT substring from the text above
- Only mark as K8S_CONCEPT if the term is genuinely a Kubernetes technical term
- Generic English words used in technical context still count if they're K8s-specific
- Version numbers, YAML metadata keys (title, id, date), and formatting are NOT K8s concepts"""


def quote_verify_candidates(
    candidates: list[str],
    context: str,
    model: str = "claude-haiku",
    batch_size: int = 12,
) -> list[dict]:
    """Use LLM to verify candidates by requiring quotes."""

    if not candidates:
        return []

    all_results = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        candidates_str = "\n".join(f"- {term}" for term in batch)

        prompt = QUOTE_VERIFY_PROMPT.format(
            context=context[:3500],
            candidates=candidates_str,
        )

        response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)

        try:
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                results = json.loads(json_match.group())
                all_results.extend(results)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Failed to parse quote-verify response: {e}", flush=True)

    return all_results


def verify_quotes_in_source(results: list[dict], source_text: str) -> list[str]:
    """Filter results to only those with valid quotes in source."""
    verified_terms = []
    source_lower = source_text.lower()

    for r in results:
        if r.get("status") != "K8S_CONCEPT":
            continue

        term = r.get("term", "")
        quote = r.get("quote", "")

        if not term:
            continue

        # Verify quote exists in source
        if quote:
            quote_lower = quote.lower()
            if quote_lower in source_lower:
                verified_terms.append(term)
            # Fuzzy match for minor variations
            elif fuzz.partial_ratio(quote_lower, source_lower) >= 90:
                verified_terms.append(term)
        # If no quote but term exists in source, accept with lower confidence
        elif term.lower() in source_lower:
            verified_terms.append(term)

    return verified_terms


# ============================================================================
# GAP-FILLING QUOTE-EXTRACT
# ============================================================================

GAP_FILL_PROMPT = """You are a Kubernetes documentation expert.

I already extracted these K8s terms: {existing_terms}

TASK: Find any Kubernetes-specific terms I MISSED in the text below.

For EACH missed term:
1. Provide the EXACT quote (5-30 words) where it appears
2. State the term

Focus on:
- Resource types (Pod, Service, Deployment, ConfigMap, etc.)
- Components (kubelet, kubectl, etcd, etc.)
- Concepts (namespace, label, selector, controller, etc.)
- Feature gates (CamelCase names like ServiceAppProtocol)
- Lifecycle stages (alpha, beta, stable) when describing K8s features

DO NOT include:
- YAML metadata (title, id, date, stages)
- Version numbers
- Generic English words unless K8s-specific in context

TEXT:
{text}

OUTPUT (JSON array):
[
  {{"quote": "exact text from source", "term": "missed_term"}}
]

If no terms were missed, return: []"""


def gap_fill_with_quotes(
    text: str,
    existing_terms: list[str],
    model: str = "claude-haiku",
) -> list[str]:
    """Find missed terms using quote-constrained extraction."""

    prompt = GAP_FILL_PROMPT.format(
        existing_terms=json.dumps(existing_terms),
        text=text[:3500],
    )

    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)

    additional_terms = []
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            results = json.loads(json_match.group())
            text_lower = text.lower()

            for r in results:
                term = r.get("term", "")
                quote = r.get("quote", "")

                if not term:
                    continue

                # Verify quote exists
                if quote and quote.lower() in text_lower:
                    additional_terms.append(term)
                elif term.lower() in text_lower:
                    additional_terms.append(term)
    except (json.JSONDecodeError, KeyError):
        pass

    return additional_terms


# ============================================================================
# FULL PIPELINE
# ============================================================================


def extract_quote_verify_pipeline(
    text: str,
    model: str = "claude-haiku",
    use_gap_fill: bool = True,
) -> dict:
    """Full Quote-Verify extraction pipeline."""

    # Phase 1: Find known terms (bypass LLM - 100% precision)
    known_found = find_known_terms_in_text(text)

    # Phase 2: Extract all candidates exhaustively
    all_candidates = extract_all_candidates_exhaustive(text)

    # Remove already-known terms from candidates
    candidates_to_verify = [
        c for c in all_candidates if c.lower() not in {k.lower() for k in known_found}
    ]

    # Phase 3: Quote-verify candidates with LLM
    verification_results = quote_verify_candidates(candidates_to_verify, text, model)
    verified_terms = verify_quotes_in_source(verification_results, text)

    # Combine known + verified
    all_terms = set(known_found) | set(verified_terms)

    # Phase 4: Gap-fill with quote-constrained extraction
    gap_terms = []
    if use_gap_fill:
        gap_terms = gap_fill_with_quotes(text, list(all_terms), model)
        all_terms.update(gap_terms)

    # Final span verification
    text_lower = text.lower()
    final_terms = [t for t in all_terms if t.lower() in text_lower]

    return {
        "known_found": list(known_found),
        "candidates": len(all_candidates),
        "verified_terms": verified_terms,
        "gap_terms": gap_terms,
        "final_terms": final_terms,
    }


# ============================================================================
# METRICS WITH TIER SUPPORT
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


def calculate_metrics(
    extracted_terms: list[str], gt_terms: list[dict], by_tier: bool = False
) -> dict:
    """Calculate metrics, optionally broken down by tier."""
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

    result = {
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
    }

    # Tier breakdown
    if by_tier:
        tier_metrics = {}
        for tier in [1, 2, 3]:
            tier_gt = [t for t in gt_terms if t.get("tier", 1) == tier]
            if tier_gt:
                tier_matched = sum(
                    1 for i in matched_gt if gt_tiers.get(gt_list[i], 1) == tier
                )
                tier_total = len(tier_gt)
                tier_metrics[f"tier_{tier}_recall"] = (
                    tier_matched / tier_total if tier_total > 0 else 0
                )
                tier_metrics[f"tier_{tier}_count"] = tier_total
        result["tier_metrics"] = tier_metrics

    return result


# ============================================================================
# EXPERIMENTS
# ============================================================================


def run_quote_verify_experiment(model: str = "claude-haiku", num_chunks: int = 20):
    """Test Quote-Verify pipeline."""
    print("\n" + "=" * 70, flush=True)
    print(f"QUOTE-VERIFY EXPERIMENT ({model}, {num_chunks} chunks)", flush=True)
    print("=" * 70, flush=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunks = gt["chunks"][:num_chunks]
    all_metrics = []

    for chunk in chunks:
        start = time.time()
        result = extract_quote_verify_pipeline(
            chunk["text"],
            model=model,
            use_gap_fill=True,
        )
        elapsed = time.time() - start

        metrics = calculate_metrics(result["final_terms"], chunk["terms"], by_tier=True)
        all_metrics.append(metrics)

        # Show results
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
            f"({elapsed:.1f}s, known={len(result['known_found'])}, verified={len(result['verified_terms'])}, gap={len(result['gap_terms'])})",
            flush=True,
        )

        if metrics["missed"] and len(metrics["missed"]) <= 5:
            print(f"    Missed: {metrics['missed']}", flush=True)
        if metrics["false_positives"] and len(metrics["false_positives"]) <= 5:
            print(f"    FP: {metrics['false_positives']}", flush=True)

    # Aggregate
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    # Tier breakdown
    tier_recalls = {1: [], 2: [], 3: []}
    for m in all_metrics:
        if "tier_metrics" in m:
            for tier in [1, 2, 3]:
                key = f"tier_{tier}_recall"
                if key in m["tier_metrics"]:
                    tier_recalls[tier].append(m["tier_metrics"][key])

    print(f"\n{'=' * 70}", flush=True)
    print(f"RESULTS: Quote-Verify Pipeline ({model})", flush=True)
    print(f"{'=' * 70}", flush=True)

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

    print(f"\n  Targets: P>95% [OK], R>95% [OK], H<1% [OK]", flush=True)
    print(f"  Close:   P>85% [~],  R>70% [~],  H<5% [~]", flush=True)

    return {
        "precision": avg_p,
        "recall": avg_r,
        "hallucination": avg_h,
        "f1": avg_f1,
        "tier_recalls": {
            k: sum(v) / len(v) if v else 0 for k, v in tier_recalls.items()
        },
    }


def run_comparison_experiment():
    """Compare Quote-Verify with previous approaches."""
    print("\n" + "=" * 70, flush=True)
    print("COMPARISON EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    results = {}

    # Quote-Verify with Haiku
    results["quote_verify_haiku"] = run_quote_verify_experiment(
        model="claude-haiku", num_chunks=20
    )

    # Quote-Verify with Sonnet (fewer chunks due to cost)
    results["quote_verify_sonnet"] = run_quote_verify_experiment(
        model="claude-sonnet", num_chunks=10
    )

    # Summary comparison
    print("\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("=" * 70, flush=True)

    # Include previous results for comparison
    previous = {
        "quote_extract_haiku": {
            "precision": 0.926,
            "recall": 0.537,
            "hallucination": 0.074,
        },
        "gleaning_haiku": {"precision": 0.627, "recall": 0.684, "hallucination": 0.323},
        "combined_sonnet": {
            "precision": 0.857,
            "recall": 0.609,
            "hallucination": 0.093,
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
    results_path = Path(__file__).parent / "artifacts" / "quote_verify_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return results


if __name__ == "__main__":
    run_comparison_experiment()
