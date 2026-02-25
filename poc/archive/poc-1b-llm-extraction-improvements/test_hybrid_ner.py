#!/usr/bin/env python3
"""Test hybrid NER+LLM approaches for high-precision K8s term extraction.

Approaches tested:
1. GLiNER zero-shot NER
2. SpaCy pattern-based extraction
3. Hybrid: Pattern + GLiNER
4. Hybrid + LLM verification
5. Local LLMs (Qwen2.5, Llama3.1, Mistral)
6. Ensemble voting across all approaches

Target: >95% precision, >95% recall, <1% hallucination
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama
import spacy
from gliner import GLiNER
from rapidfuzz import fuzz

# Add POC-1 utils to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Hybrid NER+LLM Extraction Test", flush=True)
print("=" * 60, flush=True)

# Paths
POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# KUBERNETES PATTERNS (High precision, curated list)
# ============================================================================

K8S_RESOURCES = [
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
]

K8S_COMPONENTS = [
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
    "controller",
    "scheduler",
    "api-server",
]

K8S_CONCEPTS = [
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
]

K8S_ALL_TERMS = set(t.lower() for t in K8S_RESOURCES + K8S_COMPONENTS + K8S_CONCEPTS)


# ============================================================================
# GLINER SETUP
# ============================================================================

print("Loading GLiNER model...", flush=True)
try:
    GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    GLINER_AVAILABLE = True
    print("  GLiNER loaded successfully", flush=True)
except Exception as e:
    print(f"  GLiNER failed to load: {e}", flush=True)
    GLINER_AVAILABLE = False
    GLINER_MODEL = None

GLINER_LABELS = [
    "kubernetes resource",
    "kubernetes component",
    "kubernetes concept",
    "kubernetes tool",
    "kubernetes api object",
    "feature gate",
]


# ============================================================================
# SPACY PATTERN SETUP
# ============================================================================

print("Setting up SpaCy patterns...", flush=True)
NLP = spacy.blank("en")
RULER = NLP.add_pipe("entity_ruler")

# Build patterns from our curated lists
patterns = []
for term in K8S_RESOURCES:
    patterns.append({"label": "K8S_RESOURCE", "pattern": term})
    patterns.append({"label": "K8S_RESOURCE", "pattern": term.title()})

for term in K8S_COMPONENTS:
    patterns.append({"label": "K8S_COMPONENT", "pattern": term})
    patterns.append({"label": "K8S_COMPONENT", "pattern": term.title()})

for term in K8S_CONCEPTS:
    patterns.append({"label": "K8S_CONCEPT", "pattern": term})
    patterns.append({"label": "K8S_CONCEPT", "pattern": term.title()})

RULER.add_patterns(patterns)
print(f"  Loaded {len(patterns)} patterns", flush=True)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def normalize_term(term: str) -> str:
    """Normalize term for comparison."""
    return term.lower().strip().replace("-", " ").replace("_", " ")


def extract_with_patterns(text: str) -> list[str]:
    """Extract using SpaCy pattern matching (high precision)."""
    doc = NLP(text)
    terms = set()
    for ent in doc.ents:
        terms.add(ent.text)
    return list(terms)


def extract_with_gliner(text: str, threshold: float = 0.5) -> list[str]:
    """Extract using GLiNER zero-shot NER."""
    if not GLINER_AVAILABLE:
        return []

    try:
        entities = GLINER_MODEL.predict_entities(
            text, GLINER_LABELS, threshold=threshold, flat_ner=True
        )
        return [ent["text"] for ent in entities]
    except Exception as e:
        print(f"    GLiNER error: {e}", flush=True)
        return []


def extract_with_ollama(text: str, model: str = "qwen2.5:7b") -> list[str]:
    """Extract using local Ollama model."""
    prompt = f"""Extract all Kubernetes-specific terms from this text.

Output ONLY a JSON object with this exact format:
{{"terms": ["term1", "term2", ...]}}

Kubernetes terms include:
- Resources: Pod, Deployment, Service, ConfigMap, etc.
- Components: kubelet, kubectl, etcd, etc.
- Concepts: namespace, label, selector, etc.
- Feature gates and lifecycle stages: alpha, beta, stable

Text:
{text[:3000]}

JSON:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": 1024,
            },
        )

        result = json.loads(response["response"])
        return result.get("terms", [])
    except Exception as e:
        print(f"    Ollama ({model}) error: {e}", flush=True)
        return []


def extract_hybrid_pattern_gliner(
    text: str, gliner_threshold: float = 0.5
) -> list[str]:
    """Combine pattern matching (high precision) with GLiNER (high recall)."""
    pattern_terms = set(normalize_term(t) for t in extract_with_patterns(text))
    gliner_terms = set(
        normalize_term(t) for t in extract_with_gliner(text, gliner_threshold)
    )

    # Union of both approaches
    all_terms = pattern_terms | gliner_terms
    return list(all_terms)


def extract_hybrid_with_verification(
    text: str, gliner_threshold: float = 0.5, verify_model: str = "claude-haiku"
) -> list[str]:
    """
    Hybrid extraction with LLM verification for low-confidence terms.

    1. Pattern matching (keep all - 100% precision)
    2. GLiNER (verify terms not in patterns)
    3. LLM verification for uncertain terms
    """
    # Stage 1: Pattern matching (high precision)
    pattern_terms = set(extract_with_patterns(text))

    # Stage 2: GLiNER
    gliner_terms = set(extract_with_gliner(text, gliner_threshold))

    # Terms from GLiNER that weren't found by patterns need verification
    novel_terms = gliner_terms - set(normalize_term(t) for t in pattern_terms)

    # Stage 3: Verify novel terms with LLM
    if novel_terms and verify_model:
        verified = verify_terms_with_llm(list(novel_terms), text, verify_model)
        verified_terms = set(verified)
    else:
        verified_terms = novel_terms

    # Combine pattern terms + verified GLiNER terms
    final_terms = pattern_terms | verified_terms
    return list(final_terms)


def verify_terms_with_llm(
    terms: list[str], context: str, model: str = "claude-haiku"
) -> list[str]:
    """Use LLM to verify if terms are valid K8s concepts."""
    if not terms:
        return []

    prompt = f"""Given these candidate Kubernetes terms extracted from documentation, classify each as VALID or INVALID.

A term is VALID if it's a genuine Kubernetes concept (resource, component, API concept, etc.)
A term is INVALID if it's a generic word, YAML keyword, or not K8s-specific.

Candidate terms: {json.dumps(terms)}

Context (source text):
{context[:2000]}

Output ONLY valid JSON:
{{"valid": ["term1", "term2"], "invalid": ["term3"]}}"""

    try:
        response = call_llm(prompt, model=model, temperature=0, max_tokens=500)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("valid", [])
    except Exception as e:
        print(f"    Verification error: {e}", flush=True)

    return terms  # Return all if verification fails


def extract_ensemble(text: str, min_votes: int = 2) -> list[str]:
    """
    Ensemble voting: run multiple extractors, keep terms with >= min_votes agreement.
    """
    from collections import Counter

    # Run all extractors
    extractors = {
        "patterns": extract_with_patterns(text),
        "gliner": extract_with_gliner(text, threshold=0.4),
        "qwen": extract_with_ollama(text, "qwen2.5:7b"),
    }

    # Normalize and count votes
    vote_counts = Counter()
    for name, terms in extractors.items():
        for term in terms:
            normalized = normalize_term(term)
            vote_counts[normalized] += 1

    # Keep terms with >= min_votes
    ensemble_terms = [term for term, count in vote_counts.items() if count >= min_votes]
    return ensemble_terms


# ============================================================================
# METRICS
# ============================================================================


def match_terms(extracted: str, ground_truth: str) -> str:
    """Check if extracted term matches ground truth."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    if ext_norm == gt_norm:
        return "exact"

    # Check token overlap
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return "partial"

    # Fuzzy match
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return "fuzzy"

    return "no_match"


def calculate_metrics(extracted_terms: list[str], gt_terms: list[dict]) -> dict:
    """Calculate precision, recall, hallucination."""
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
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment():
    """Run comprehensive comparison of all extraction approaches."""

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    NUM_CHUNKS = 15
    chunks = gt["chunks"][:NUM_CHUNKS]

    print(f"\n{'=' * 70}", flush=True)
    print(f"HYBRID NER+LLM COMPARISON ({NUM_CHUNKS} chunks)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Define approaches to test
    approaches = {
        "1. Patterns Only": lambda t: extract_with_patterns(t),
        "2. GLiNER (0.5)": lambda t: extract_with_gliner(t, 0.5),
        "3. GLiNER (0.3)": lambda t: extract_with_gliner(t, 0.3),
        "4. Hybrid (P+G)": lambda t: extract_hybrid_pattern_gliner(t, 0.5),
        "5. Hybrid+Verify": lambda t: extract_hybrid_with_verification(
            t, 0.5, "claude-haiku"
        ),
        "6. Qwen2.5:7b": lambda t: extract_with_ollama(t, "qwen2.5:7b"),
        "7. Llama3.1:8b": lambda t: extract_with_ollama(t, "llama3.1:8b"),
        "8. Ensemble (2+)": lambda t: extract_ensemble(t, min_votes=2),
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
                    f"  {name:<20} P={metrics['precision']:>5.0%} R={metrics['recall']:>5.0%} H={metrics['hallucination']:>5.0%} ({len(terms)} terms)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<20} ERROR: {e}", flush=True)

    # Aggregate results
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(
        f"{'Approach':<22} {'Precision':>10} {'Recall':>10} {'Hallucination':>14} {'F1':>8}",
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

            p_status = "OK" if avg_p >= 0.95 else ("~" if avg_p >= 0.85 else "  ")
            r_status = "OK" if avg_r >= 0.95 else ("~" if avg_r >= 0.85 else "  ")
            h_status = "OK" if avg_h < 0.01 else ("~" if avg_h < 0.05 else "  ")

            print(
                f"{name:<22} {avg_p:>8.1%} {p_status} {avg_r:>8.1%} {r_status} {avg_h:>12.1%} {h_status} {f1:>7.1%}",
                flush=True,
            )

    print(f"\nBest F1: {best_approach} ({best_f1:.1%})", flush=True)
    print(f"\nTargets: P>95% [OK], R>95% [OK], H<1% [OK]", flush=True)
    print(f"Close:   P>85% [~],  R>85% [~],  H<5% [~]", flush=True)

    return results


if __name__ == "__main__":
    run_experiment()
