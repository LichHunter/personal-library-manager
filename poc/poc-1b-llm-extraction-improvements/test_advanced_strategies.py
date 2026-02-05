#!/usr/bin/env python3
"""Test advanced extraction strategies based on research findings.

Key strategies from research:
1. Quote-then-Extract: Force LLM to quote source before extracting
2. Gleaning: Multi-pass extraction ("Did you miss anything?")
3. Cross-encoder verification: NLI-based validation
4. Knowledge base validation: Check against K8s API reference

Target: >95% precision, >95% recall, <1% hallucination
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Advanced Extraction Strategies", flush=True)
print("=" * 60, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# KUBERNETES KNOWLEDGE BASE (For validation)
# ============================================================================

K8S_VALID_TERMS = {
    # Core resources
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
    "ingress",
    "ingresses",
    "namespace",
    "namespaces",
    "node",
    "nodes",
    "persistentvolume",
    "pv",
    "persistentvolumeclaim",
    "pvc",
    "statefulset",
    "statefulsets",
    "daemonset",
    "daemonsets",
    "replicaset",
    "replicasets",
    "job",
    "jobs",
    "cronjob",
    "cronjobs",
    "serviceaccount",
    "endpoints",
    "endpoint",
    "container",
    "containers",
    "volume",
    "volumes",
    "networkpolicy",
    "resourcequota",
    "limitrange",
    "horizontalpodautoscaler",
    "hpa",
    "poddisruptionbudget",
    "pdb",
    "clusterrole",
    "clusterrolebinding",
    "role",
    "rolebinding",
    "customresourcedefinition",
    "crd",
    # Components
    "kubelet",
    "kubectl",
    "kube-proxy",
    "kube-apiserver",
    "kube-scheduler",
    "kube-controller-manager",
    "etcd",
    "coredns",
    "helm",
    "kubernetes",
    "api server",
    "api-server",
    "controller",
    "scheduler",
    # Concepts
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
    "cluster",
    "workload",
    "replica",
    "replicas",
    "rollout",
    "rollback",
    "watch",
    "feature_gate",
    "feature gate",
    "alpha",
    "beta",
    "stable",
    "ga",
    "deprecated",
    "object",
    "objects",
    "api",
    "apiversion",
    "kind",
    "metadata",
    "spec",
    "status",
    "finalizer",
    "webhook",
    "admission",
    "cni",
    "csi",
    "cri",
    "runtime",
    "podspec",
    "podspecs",
    # Feature gates (common)
    "serviceappprotocol",
    "jobpodfailurepolicy",
    "poddisruptionbudget",
    "csivolume",
    "csinodeinfo",
    "honorpvreclaimmpolicy",
    "csistoragecapacity",
}


# ============================================================================
# STRATEGY 1: QUOTE-THEN-EXTRACT
# ============================================================================

PROMPT_QUOTE_THEN_EXTRACT = """You are a Kubernetes documentation expert.

TASK: Extract Kubernetes-specific terms from the text below.

CRITICAL INSTRUCTIONS:
For EACH entity you extract, you MUST:
1. QUOTE: Copy the EXACT text from the source containing the entity (verbatim, no modifications)
2. REASON: Explain briefly why this is a Kubernetes concept
3. ENTITY: The normalized entity name
4. TYPE: One of [resource, component, concept, feature_gate, lifecycle]

Only extract genuine Kubernetes concepts:
- Resources: Pod, Deployment, Service, ConfigMap, etc.
- Components: kubelet, kubectl, etcd, API server, etc.
- Concepts: namespace, label, selector, watch, etc.
- Feature gates: ServiceAppProtocol, JobPodFailurePolicy, etc.
- Lifecycle stages: alpha, beta, stable (when describing K8s features)

DO NOT extract:
- YAML metadata keywords (title, content_type, stage, removed, list, render)
- Version numbers (1.18, 1.19, fromVersion, toVersion)
- Generic words unless they're K8s-specific in context

<text>
{chunk_text}
</text>

Output ONLY valid JSON:
{{"entities": [
  {{"quote": "exact text from source", "reason": "why it's K8s", "entity": "name", "type": "type"}}
]}}"""


def extract_quote_then_extract(text: str, model: str = "claude-haiku") -> list[str]:
    """Quote-then-extract approach - forces grounding in source."""
    prompt = PROMPT_QUOTE_THEN_EXTRACT.format(chunk_text=text[:3500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)

    # Parse response
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    try:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            entities = data.get("entities", [])

            # Verify quotes exist in source
            verified = []
            text_lower = text.lower()
            for ent in entities:
                quote = ent.get("quote", "")
                entity = ent.get("entity", "")

                # Check if quote exists in source (case-insensitive)
                if quote.lower() in text_lower:
                    verified.append(entity)
                # Fallback: check if entity itself exists
                elif entity.lower() in text_lower:
                    verified.append(entity)

            return verified
    except json.JSONDecodeError:
        pass

    return []


# ============================================================================
# STRATEGY 2: GLEANING (Multi-pass extraction)
# ============================================================================

PROMPT_INITIAL = """Extract ALL Kubernetes-specific terms from this text.

Include: resources (Pod, Service, etc.), components (kubelet, etc.), concepts (namespace, label, etc.), feature gates, lifecycle stages (alpha, beta, stable).

Exclude: YAML keywords (title, content_type, stage, removed), version numbers, generic words.

<text>
{chunk_text}
</text>

Output ONLY valid JSON:
{{"terms": ["term1", "term2", ...]}}"""

PROMPT_GLEANING = """You previously extracted these Kubernetes terms:
{previous_terms}

MANY entities were likely missed in the last extraction.

Review the text again and identify ANY additional Kubernetes terms you missed.
Focus on:
- Feature gate names (often CamelCase like ServiceAppProtocol)
- Component names (kubelet, kubectl, etcd, etc.)
- Lifecycle stages (alpha, beta, stable) when describing K8s features
- API concepts (Watch, controller, etc.)

<text>
{chunk_text}
</text>

Output ONLY valid JSON with ADDITIONAL terms (not already listed):
{{"additional_terms": ["term1", "term2", ...]}}

If no additional terms, return: {{"additional_terms": []}}"""

PROMPT_GLEANING_CHECK = """Based on the text and extracted terms, are there still Kubernetes entities that need to be added?

Previously extracted: {all_terms}

<text>
{chunk_text}
</text>

Answer with ONLY 'Y' (yes, more entities exist) or 'N' (no, extraction is complete)."""


def extract_with_gleaning(
    text: str, model: str = "claude-haiku", max_gleanings: int = 2
) -> list[str]:
    """Multi-pass extraction with gleaning (GraphRAG technique)."""

    # Initial extraction
    prompt = PROMPT_INITIAL.format(chunk_text=text[:3500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)

    all_terms = set()
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            all_terms.update(data.get("terms", []))
    except json.JSONDecodeError:
        pass

    # Gleaning passes
    for i in range(max_gleanings):
        # Ask for more entities
        prompt = PROMPT_GLEANING.format(
            previous_terms=json.dumps(list(all_terms)), chunk_text=text[:3000]
        )
        response = call_llm(prompt, model=model, temperature=0, max_tokens=500)

        try:
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                additional = data.get("additional_terms", [])
                if additional:
                    all_terms.update(additional)
                else:
                    break  # No more entities found
        except json.JSONDecodeError:
            break

        # Check if more entities exist (optional - can skip for efficiency)
        if i < max_gleanings - 1:
            check_prompt = PROMPT_GLEANING_CHECK.format(
                all_terms=json.dumps(list(all_terms)), chunk_text=text[:2000]
            )
            check_response = call_llm(
                check_prompt, model=model, temperature=0, max_tokens=5
            )
            if "N" in check_response.upper():
                break

    # Verify terms exist in source
    text_lower = text.lower()
    verified = [t for t in all_terms if t.lower() in text_lower]

    return verified


# ============================================================================
# STRATEGY 3: COMBINED - Quote + Gleaning + KB Validation
# ============================================================================


def extract_combined_strategy(text: str, model: str = "claude-haiku") -> list[str]:
    """
    Combined best strategies:
    1. Quote-then-extract (reduces hallucination)
    2. Gleaning pass (improves recall)
    3. KB validation (ensures K8s relevance)
    4. Source verification (final filter)
    """

    # Stage 1: Quote-then-extract (high precision)
    quote_terms = set(extract_quote_then_extract(text, model))

    # Stage 2: Gleaning for missed entities
    gleaning_prompt = f"""You extracted these K8s terms: {json.dumps(list(quote_terms))}

Review again - what Kubernetes terms did you MISS? Look for:
- Feature gates (CamelCase names like ServiceAppProtocol, JobPodFailurePolicy)
- API concepts (Watch, controller, object, stream)
- Components (kubelet, kubectl, etcd)
- Lifecycle (alpha, beta, stable when describing K8s)

<text>
{text[:3000]}
</text>

For each MISSED term, provide quote and entity:
{{"missed": [{{"quote": "exact text", "entity": "name"}}]}}"""

    response = call_llm(gleaning_prompt, model=model, temperature=0, max_tokens=500)

    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            text_lower = text.lower()
            for item in data.get("missed", []):
                entity = item.get("entity", "")
                quote = item.get("quote", "")
                # Verify quote or entity exists
                if quote.lower() in text_lower or entity.lower() in text_lower:
                    quote_terms.add(entity)
    except json.JSONDecodeError:
        pass

    # Stage 3: KB validation (filter non-K8s terms)
    kb_validated = []
    for term in quote_terms:
        term_lower = term.lower().replace("-", " ").replace("_", " ")
        # Check if term or any word in term is in K8s vocabulary
        if term_lower in K8S_VALID_TERMS:
            kb_validated.append(term)
        elif any(word in K8S_VALID_TERMS for word in term_lower.split()):
            kb_validated.append(term)
        else:
            # Keep if it looks like a feature gate (CamelCase)
            if re.match(r"^[A-Z][a-z]+([A-Z][a-z]+)+$", term):
                kb_validated.append(term)
            # Keep alpha/beta/stable
            elif term_lower in {"alpha", "beta", "stable", "ga", "deprecated"}:
                kb_validated.append(term)

    return kb_validated


# ============================================================================
# STRATEGY 4: SONNET WITH FULL PIPELINE
# ============================================================================


def extract_sonnet_full_pipeline(text: str) -> list[str]:
    """Use Sonnet (stronger model) with full pipeline."""
    return extract_combined_strategy(text, model="claude-sonnet")


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
    matched_gt = set()
    tp = 0

    for ext in extracted_terms:
        for i, gt in enumerate(gt_list):
            if i in matched_gt:
                continue
            if match_terms(ext, gt) != "no_match":
                matched_gt.add(i)
                tp += 1
                break

    fp = len(extracted_terms) - tp
    fn = len(gt_list) - len(matched_gt)

    precision = tp / len(extracted_terms) if extracted_terms else 0
    recall = tp / len(gt_list) if gt_list else 0
    hallucination = fp / len(extracted_terms) if extracted_terms else 0

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted": len(extracted_terms),
        "ground_truth": len(gt_list),
        "missed": [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt],
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment():
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    NUM_CHUNKS = 20
    chunks = gt["chunks"][:NUM_CHUNKS]

    print(f"\n{'=' * 70}", flush=True)
    print(f"ADVANCED STRATEGIES COMPARISON ({NUM_CHUNKS} chunks)", flush=True)
    print(f"{'=' * 70}", flush=True)

    approaches = {
        "1. Quote-Extract (Haiku)": lambda t: extract_quote_then_extract(
            t, "claude-haiku"
        ),
        "2. Gleaning 2x (Haiku)": lambda t: extract_with_gleaning(t, "claude-haiku", 2),
        "3. Combined (Haiku)": lambda t: extract_combined_strategy(t, "claude-haiku"),
        "4. Combined (Sonnet)": lambda t: extract_sonnet_full_pipeline(t),
    }

    results = {name: {"metrics": [], "times": []} for name in approaches}

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        gt_terms = chunk["terms"]

        print(f"\n{chunk_id} (GT: {len(gt_terms)} terms):", flush=True)

        for name, extractor in approaches.items():
            try:
                start = time.time()
                terms = extractor(chunk_text)
                elapsed = time.time() - start

                metrics = calculate_metrics(terms, gt_terms)
                results[name]["metrics"].append(metrics)
                results[name]["times"].append(elapsed)

                print(
                    f"  {name:<24} P={metrics['precision']:>5.0%} R={metrics['recall']:>5.0%} H={metrics['hallucination']:>5.0%} ({len(terms)} terms, {elapsed:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<24} ERROR: {e}", flush=True)

    # Aggregate results
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(
        f"{'Approach':<26} {'Precision':>10} {'Recall':>10} {'Hallucination':>14} {'F1':>8} {'Avg Time':>10}",
        flush=True,
    )
    print("-" * 82, flush=True)

    best_f1 = 0
    best_approach = None

    for name, data in results.items():
        if data["metrics"]:
            avg_p = sum(m["precision"] for m in data["metrics"]) / len(data["metrics"])
            avg_r = sum(m["recall"] for m in data["metrics"]) / len(data["metrics"])
            avg_h = sum(m["hallucination"] for m in data["metrics"]) / len(
                data["metrics"]
            )
            avg_t = sum(data["times"]) / len(data["times"])
            f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_approach = name

            p_status = "OK" if avg_p >= 0.95 else ("~" if avg_p >= 0.85 else "  ")
            r_status = "OK" if avg_r >= 0.95 else ("~" if avg_r >= 0.70 else "  ")
            h_status = "OK" if avg_h < 0.01 else ("~" if avg_h < 0.05 else "  ")

            print(
                f"{name:<26} {avg_p:>8.1%} {p_status} {avg_r:>8.1%} {r_status} {avg_h:>12.1%} {h_status} {f1:>7.1%} {avg_t:>9.1f}s",
                flush=True,
            )

    print(f"\nBest F1: {best_approach} ({best_f1:.1%})", flush=True)
    print(f"\nTargets: P>95% [OK], R>95% [OK], H<1% [OK]", flush=True)
    print(f"Close:   P>85% [~],  R>70% [~],  H<5% [~]", flush=True)

    # Save results
    results_path = (
        Path(__file__).parent / "artifacts" / "advanced_strategies_results.json"
    )
    results_path.parent.mkdir(exist_ok=True)

    summary = {}
    for name, data in results.items():
        if data["metrics"]:
            summary[name] = {
                "precision": sum(m["precision"] for m in data["metrics"])
                / len(data["metrics"]),
                "recall": sum(m["recall"] for m in data["metrics"])
                / len(data["metrics"]),
                "hallucination": sum(m["hallucination"] for m in data["metrics"])
                / len(data["metrics"]),
                "f1": 2
                * summary[name]["precision"]
                * summary[name]["recall"]
                / (summary[name]["precision"] + summary[name]["recall"])
                if name in summary
                else 0,
                "avg_time": sum(data["times"]) / len(data["times"]),
                "num_chunks": len(data["metrics"]),
            }

    with open(results_path, "w") as f:
        json.dump(
            {"summary": summary, "best_approach": best_approach, "best_f1": best_f1},
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_path}", flush=True)

    return results


if __name__ == "__main__":
    run_experiment()
