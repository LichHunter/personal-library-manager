#!/usr/bin/env python3
"""Test Pattern + LLM expansion approach.

Key insight from previous tests:
- Patterns: High precision (75%+), limited recall (58%)
- LLMs: Variable precision, higher recall but hallucinate
- GLiNER: Doesn't work well for K8s domain

New approach: Use patterns as foundation, LLM to expand recall while
keeping hallucination low by constraining LLM to only add terms that
are clearly present in the source text.

Target: >90% precision, >85% recall, <5% hallucination
"""

import json
import re
import sys
from pathlib import Path

import spacy
from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Pattern + LLM Expansion Test", flush=True)
print("=" * 60, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# EXPANDED KUBERNETES PATTERNS
# ============================================================================

# Core resources
K8S_RESOURCES = {
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
    "serviceaccounts",
    "networkpolicy",
    "networkpolicies",
    "resourcequota",
    "limitrange",
    "horizontalpodautoscaler",
    "hpa",
    "poddisruptionbudget",
    "pdb",
    "endpoints",
    "endpoint",
    "container",
    "containers",
    "volume",
    "volumes",
    "clusterrole",
    "clusterrolebinding",
    "role",
    "rolebinding",
    "customresourcedefinition",
    "crd",
    "validatingwebhookconfiguration",
    "mutatingwebhookconfiguration",
    "storageclass",
    "volumesnapshot",
}

# Components and tools
K8S_COMPONENTS = {
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
    "controller-manager",
    "cloud-controller-manager",
    "kube-dns",
    "metrics-server",
}

# API and concepts
K8S_CONCEPTS = {
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
    "nodeaffinity",
    "podaffinity",
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
    "api version",
    "apiversion",
    "kind",
    "metadata",
    "spec",
    "status",
    "finalizer",
    "finalizers",
    "owner reference",
    "ownerreference",
    "admission controller",
    "admission-controller",
    "webhook",
    "resource quota",
    "limit range",
    "priority class",
    "priorityclass",
    "pod security",
    "podsecurity",
    "network policy",
    "ingress",
    "egress",
    "cni",
    "csi",
    "cri",
    "oci",
    "runtime",
    "container runtime",
}

# Feature gates (common ones)
K8S_FEATURE_GATES = {
    "serviceappprotocol",
    "jobpodfailurepolicy",
    "poddisruptionbudget",
    "csivolume",
    "csinodeinfo",
    "honorpvreclaimmpolicy",
    "podspec",
    "csistoragecapacity",
    "genericephemeralvolume",
    "localstoragecapacityisolation",
}

ALL_K8S_PATTERNS = K8S_RESOURCES | K8S_COMPONENTS | K8S_CONCEPTS | K8S_FEATURE_GATES

print(f"Loaded {len(ALL_K8S_PATTERNS)} patterns", flush=True)


# ============================================================================
# SPACY SETUP
# ============================================================================

NLP = spacy.blank("en")
RULER = NLP.add_pipe("entity_ruler")

patterns = []
for term in ALL_K8S_PATTERNS:
    # Add lowercase, titlecase, and uppercase versions
    patterns.append({"label": "K8S", "pattern": term})
    patterns.append({"label": "K8S", "pattern": term.title()})
    patterns.append({"label": "K8S", "pattern": term.upper()})
    # Handle compound terms with different separators
    if " " in term:
        patterns.append({"label": "K8S", "pattern": term.replace(" ", "-")})
        patterns.append({"label": "K8S", "pattern": term.replace(" ", "_")})

RULER.add_patterns(patterns)
print(f"Added {len(patterns)} SpaCy patterns", flush=True)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def extract_with_patterns(text: str) -> set[str]:
    """Pattern-based extraction."""
    doc = NLP(text)
    return {ent.text for ent in doc.ents}


def extract_with_llm_expansion(
    text: str, pattern_terms: set[str], model: str = "claude-haiku"
) -> list[str]:
    """
    Use LLM to find K8s terms that patterns missed.

    Key constraints:
    1. Only extract terms ACTUALLY PRESENT in the text
    2. Only K8s-specific terms, not generic words
    3. Provide the already-found terms to avoid duplicates
    """

    prompt = f"""You are a Kubernetes documentation expert.

I've already extracted these Kubernetes terms from the text using pattern matching:
{json.dumps(list(pattern_terms))}

Your task: Find any ADDITIONAL Kubernetes-specific terms I MISSED.

RULES:
1. ONLY include terms that appear EXACTLY in the source text (copy exact spelling)
2. ONLY include genuine Kubernetes concepts (resources, components, APIs, etc.)
3. DO NOT include:
   - Generic words (title, content, stage, removed, etc.)
   - YAML structure keywords
   - Version numbers
   - Terms I already found

<text>
{text[:3500]}
</text>

Output ONLY valid JSON:
{{"additional_terms": ["term1", "term2", ...]}}

If no additional terms found, return: {{"additional_terms": []}}"""

    try:
        response = call_llm(prompt, model=model, temperature=0, max_tokens=500)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            additional = data.get("additional_terms", [])

            # Verify each additional term exists in text (post-verification)
            verified = []
            text_lower = text.lower()
            for term in additional:
                if term.lower() in text_lower:
                    verified.append(term)

            return verified
    except Exception as e:
        print(f"    LLM expansion error: {e}", flush=True)

    return []


def extract_combined(text: str, model: str = "claude-haiku") -> list[str]:
    """
    Combined approach:
    1. Pattern matching (high precision foundation)
    2. LLM expansion (boost recall)
    3. Source verification (eliminate hallucinations)
    """
    # Stage 1: Pattern extraction
    pattern_terms = extract_with_patterns(text)

    # Stage 2: LLM expansion
    additional_terms = extract_with_llm_expansion(text, pattern_terms, model)

    # Combine
    all_terms = set(normalize_term(t) for t in pattern_terms)
    all_terms.update(normalize_term(t) for t in additional_terms)

    return list(all_terms)


def extract_llm_only_constrained(text: str, model: str = "claude-haiku") -> list[str]:
    """
    LLM extraction with strict source verification.
    """
    prompt = f"""Extract Kubernetes-specific terms from this text.

RULES:
1. ONLY include terms that appear EXACTLY in the source text
2. ONLY include genuine Kubernetes concepts:
   - Resources: Pod, Deployment, Service, ConfigMap, etc.
   - Components: kubelet, kubectl, etcd, etc.  
   - Concepts: namespace, label, selector, etc.
   - Feature gates: ServiceAppProtocol, JobPodFailurePolicy, etc.
   - Lifecycle stages: alpha, beta, stable (when describing K8s features)
3. DO NOT include:
   - YAML keywords: title, content_type, stage, removed, list, render
   - Version numbers: 1.18, fromVersion, toVersion
   - Generic words: object (unless "Kubernetes object"), stream

<text>
{text[:3500]}
</text>

Output ONLY valid JSON:
{{"terms": ["term1", "term2", ...]}}"""

    try:
        response = call_llm(prompt, model=model, temperature=0, max_tokens=500)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms = data.get("terms", [])

            # Verify each term exists in text
            verified = []
            text_lower = text.lower()
            for term in terms:
                if term.lower() in text_lower:
                    verified.append(term)

            return verified
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)

    return []


# ============================================================================
# METRICS
# ============================================================================


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
    print(f"PATTERN + LLM EXPANSION ({NUM_CHUNKS} chunks)", flush=True)
    print(f"{'=' * 70}", flush=True)

    approaches = {
        "1. Patterns (expanded)": lambda t: list(extract_with_patterns(t)),
        "2. Pattern + Haiku": lambda t: extract_combined(t, "claude-haiku"),
        "3. Pattern + Sonnet": lambda t: extract_combined(t, "claude-sonnet"),
        "4. Haiku constrained": lambda t: extract_llm_only_constrained(
            t, "claude-haiku"
        ),
        "5. Sonnet constrained": lambda t: extract_llm_only_constrained(
            t, "claude-sonnet"
        ),
    }

    results = {name: {"metrics": []} for name in approaches}

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        gt_terms = chunk["terms"]

        print(f"\n{chunk_id} (GT: {len(gt_terms)} terms):", flush=True)

        for name, extractor in approaches.items():
            try:
                terms = extractor(chunk_text)
                metrics = calculate_metrics(terms, gt_terms)
                results[name]["metrics"].append(metrics)

                print(
                    f"  {name:<22} P={metrics['precision']:>5.0%} R={metrics['recall']:>5.0%} H={metrics['hallucination']:>5.0%} ({len(terms)} terms)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<22} ERROR: {e}", flush=True)

    # Aggregate
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(
        f"{'Approach':<24} {'Precision':>10} {'Recall':>10} {'Hallucination':>14} {'F1':>8}",
        flush=True,
    )
    print("-" * 70, flush=True)

    best_f1 = 0
    best_approach = None

    for name, data in results.items():
        if data["metrics"]:
            avg_p = sum(m["precision"] for m in data["metrics"]) / len(data["metrics"])
            avg_r = sum(m["recall"] for m in data["metrics"]) / len(data["metrics"])
            avg_h = sum(m["hallucination"] for m in data["metrics"]) / len(
                data["metrics"]
            )
            f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_approach = name

            p_status = "OK" if avg_p >= 0.90 else ("~" if avg_p >= 0.80 else "  ")
            r_status = "OK" if avg_r >= 0.85 else ("~" if avg_r >= 0.70 else "  ")
            h_status = "OK" if avg_h < 0.05 else ("~" if avg_h < 0.10 else "  ")

            print(
                f"{name:<24} {avg_p:>8.1%} {p_status} {avg_r:>8.1%} {r_status} {avg_h:>12.1%} {h_status} {f1:>7.1%}",
                flush=True,
            )

    print(f"\nBest F1: {best_approach} ({best_f1:.1%})", flush=True)
    print(f"\nTargets: P>90% [OK], R>85% [OK], H<5% [OK]", flush=True)
    print(f"Relaxed: P>80% [~],  R>70% [~],  H<10% [~]", flush=True)

    return results


if __name__ == "__main__":
    run_experiment()
