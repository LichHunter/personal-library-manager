#!/usr/bin/env python3
"""Test improved recall strategies based on research findings.

Key strategies from production systems (GraphRAG, LangExtract, fast-graphrag):
1. Gleaning: "MANY entities were missed" follow-up prompts
2. Category-by-category extraction with taxonomy
3. Exhaustive prompting with explicit completeness instructions
4. Chain-of-thought structured thinking
5. Contrastive examples (DO/DON'T)
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator
from rapidfuzz import fuzz

# Add POC-1 utils to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Improved Recall Strategies Test", flush=True)
print("=" * 60, flush=True)

# Paths
POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# Validation Model
# ============================================================================


class ExtractedTermVerified(BaseModel):
    term: str
    span: str
    confidence: str

    @model_validator(mode="before")
    @classmethod
    def validate_span_exists(cls, values):
        source_text = values.pop("_source_text", "")
        span = values.get("span", "")
        term = values.get("term", "")
        if source_text and span:
            if span not in source_text:
                raise ValueError(f"Span not found")
            if term.lower() not in span.lower():
                raise ValueError(f"Term not in span")
        return values


# ============================================================================
# IMPROVED PROMPTS (Based on Research)
# ============================================================================

# Strategy 1: Exhaustive Prompt with Taxonomy (from Graphiti + Claude docs)
PROMPT_EXHAUSTIVE_TAXONOMY = """You are an expert Kubernetes terminology annotator with 10+ years of experience.
Your specialty is comprehensive, HIGH-RECALL extraction of Kubernetes-specific terms.

<task>
Extract ALL Kubernetes-specific terms from the text below. 
PRIORITIZE RECALL over precision - it's better to extract a borderline term than to miss a valid one.
</task>

<workflow>
1. FIRST: Read the ENTIRE text to understand context
2. SECOND: Go through EACH category in the taxonomy below
3. THIRD: For each category, extract ALL matching terms
4. FOURTH: Check for variations (plurals, abbreviations, synonyms)
5. FIFTH: Review - "Did I miss any terms mentioned explicitly OR implicitly?"
</workflow>

<taxonomy>
EXTRACT terms from these Kubernetes categories:
- WORKLOAD RESOURCES: Pod, Deployment, StatefulSet, DaemonSet, Job, CronJob, ReplicaSet, container
- NETWORK RESOURCES: Service, Services, Ingress, NetworkPolicy, Endpoint, Endpoints, EndpointSlice, DNS
- CONFIG & STORAGE: ConfigMap, Secret, Volume, PV, PVC, StorageClass, emptyDir, object
- CLUSTER COMPONENTS: Node, Namespace, ResourceQuota, LimitRange, API Server, etcd, Kubernetes
- FEATURE GATES: feature_gate, alpha, beta, stable, GA, deprecated, removed (feature stages)
- SECURITY: RBAC, Role, ClusterRole, ServiceAccount, SecurityContext, PodSecurityPolicy
- CLI & TOOLS: kubectl, kubeadm, kubelet, kube-proxy, helm, kustomize
- API CONCEPTS: Watch, watch, stream, API verb, controller, polling, detection
- CONCEPTS: label, selector, annotation, taint, toleration, affinity, scheduling, replica
- STATUS & ERRORS: Running, Pending, CrashLoopBackOff, OOMKilled, ImagePullBackOff
- DOCUMENT METADATA: title, content_type values (e.g., feature_gate), stage names
</taxonomy>

<contrastive_examples>
EXTRACT (Kubernetes-specific):
✅ "pod" - core K8s resource
✅ "kubectl apply" - K8s CLI command
✅ "rolling update" - K8s deployment strategy
✅ "kube-system" - K8s system namespace
✅ "container runtime" - when in K8s context

DO NOT EXTRACT (too generic without context):
❌ "server" alone (unless "API server" or "metrics-server")
❌ "cluster" alone (unless clearly "Kubernetes cluster")
❌ "container" alone (unless K8s-specific context)
</contrastive_examples>

<text>
{chunk_text}
</text>

<instructions>
Think step-by-step:
1. What categories from the taxonomy are present in this text?
2. For EACH category present, what terms should I extract?
3. Are there any IMPLICIT references (e.g., "it" referring to a Pod)?
4. Are there abbreviated or variant forms (e.g., "k8s" = "Kubernetes")?
5. Final check: Have I extracted EVERYTHING?

Output ONLY valid JSON (no markdown):
{{"terms": [{{"term": "...", "span": "exact quote containing term", "confidence": "HIGH|MEDIUM|LOW"}}]}}
</instructions>"""


# Strategy 2: Gleaning Prompt (from Microsoft GraphRAG + fast-graphrag)
PROMPT_GLEANING = """You previously extracted these Kubernetes terms: {extracted_terms}

MANY entities and terms were MISSED in the last extraction.

Review the text AGAIN and add ALL terms that were missed:
- Terms you may have overlooked
- Implicit references
- Abbreviated forms
- Multi-word terms
- Terms in subordinate clauses or examples

<text>
{chunk_text}
</text>

IMPORTANT: You will be evaluated on RECALL. Missing a term is worse than including an uncertain one.

Output ONLY the ADDITIONAL terms (not already extracted) as valid JSON:
{{"terms": [{{"term": "...", "span": "exact quote", "confidence": "HIGH|MEDIUM|LOW"}}]}}"""


# Strategy 3: Category-by-Category Extraction
PROMPT_CATEGORY = """Extract ONLY {category} terms from this Kubernetes documentation.

{category_description}

<text>
{chunk_text}
</text>

Think: What {category} terms appear in this text? Look for both explicit mentions and implicit references.

Output ONLY {category} terms as valid JSON:
{{"terms": [{{"term": "...", "span": "exact quote", "confidence": "HIGH|MEDIUM|LOW"}}]}}"""

CATEGORIES = {
    "WORKLOAD RESOURCES": "Pod, Deployment, StatefulSet, DaemonSet, Job, CronJob, ReplicaSet, container",
    "NETWORK RESOURCES": "Service, Services, Ingress, NetworkPolicy, Endpoint, Endpoints, DNS",
    "CONFIG & STORAGE": "ConfigMap, Secret, Volume, PersistentVolume, PVC, StorageClass, object",
    "CLUSTER & ADMIN": "Node, Namespace, Kubernetes, ResourceQuota, LimitRange, API Server, etcd",
    "FEATURE GATES & STAGES": "feature_gate, alpha, beta, stable, GA, deprecated, removed",
    "API CONCEPTS": "Watch, watch, stream, API verb, controller, polling, detection",
    "CONCEPTS & METADATA": "label, selector, annotation, title, content_type, stage",
}


# Strategy 4: Verification/Completeness Check (from fast-graphrag gleaning)
PROMPT_VERIFICATION = """You have extracted these terms: {extracted_terms}

Retrospectively check if ALL Kubernetes terms have been correctly identified.

Review the text one more time:
<text>
{chunk_text}
</text>

Questions to verify completeness:
1. Are there any Kubernetes resources mentioned I missed?
2. Are there any CLI commands or tools?
3. Are there any status conditions or error states?
4. Are there any configuration options?
5. Are there any abbreviations I didn't expand?

If you find ANY missed terms, output them. If extraction is truly complete, output empty list.

Output as valid JSON:
{{"terms": [{{"term": "...", "span": "...", "confidence": "..."}}], "is_complete": true|false}}"""


# ============================================================================
# Helpers
# ============================================================================


def parse_json_response(response: str) -> list[dict]:
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("terms", [])
        except json.JSONDecodeError:
            pass
    return []


def validate_terms_with_source(terms: list[dict], source_text: str) -> list[dict]:
    validated = []
    for t in terms:
        try:
            t_with_source = {**t, "_source_text": source_text}
            term = ExtractedTermVerified(**t_with_source)
            validated.append(
                {"term": term.term, "span": term.span, "confidence": term.confidence}
            )
        except (ValidationError, ValueError):
            pass
    return validated


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
    matches = {"exact": 0, "partial": 0, "fuzzy": 0}

    for ext in extracted_terms:
        best_match = None
        best_type = "no_match"

        for i, gt in enumerate(gt_list):
            if i in matched_gt:
                continue
            mt = match_terms(ext, gt)
            if mt == "exact":
                best_match, best_type = i, "exact"
                break
            elif mt == "partial" and best_type not in ["exact", "partial"]:
                best_match, best_type = i, "partial"
            elif mt == "fuzzy" and best_type == "no_match":
                best_match, best_type = i, "fuzzy"

        if best_match is not None:
            matched_gt.add(best_match)
            matches[best_type] += 1

    tp = sum(matches.values())
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
# Strategies
# ============================================================================


def strategy_exhaustive_taxonomy(chunk_text: str, model: str) -> list[str]:
    """Strategy: Exhaustive prompt with full taxonomy."""
    prompt = PROMPT_EXHAUSTIVE_TAXONOMY.format(chunk_text=chunk_text[:4000])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    return [t["term"] for t in validated]


def strategy_exhaustive_plus_gleaning(chunk_text: str, model: str) -> list[str]:
    """Strategy: Exhaustive extraction + gleaning follow-up."""
    # Initial extraction
    terms = set(strategy_exhaustive_taxonomy(chunk_text, model))

    # Gleaning round
    print("      Gleaning...", end=" ", flush=True)
    prompt = PROMPT_GLEANING.format(
        extracted_terms=list(terms), chunk_text=chunk_text[:4000]
    )
    response = call_llm(prompt, model=model, temperature=0.3, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    new_terms = [t["term"] for t in validated]
    print(f"+{len(new_terms)}", flush=True)
    terms.update(new_terms)

    return list(terms)


def strategy_category_by_category(chunk_text: str, model: str) -> list[str]:
    """Strategy: Extract by category, then merge."""
    all_terms = set()

    for category, description in CATEGORIES.items():
        print(f"      {category}...", end=" ", flush=True)
        prompt = PROMPT_CATEGORY.format(
            category=category,
            category_description=description,
            chunk_text=chunk_text[:4000],
        )
        response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
        raw = parse_json_response(response)
        validated = validate_terms_with_source(raw, chunk_text)
        terms = [t["term"] for t in validated]
        all_terms.update(terms)
        print(f"{len(terms)}", flush=True)

    return list(all_terms)


def strategy_exhaustive_gleaning_verify(chunk_text: str, model: str) -> list[str]:
    """Strategy: Exhaustive + Gleaning + Verification loop."""
    # Initial
    terms = set(strategy_exhaustive_taxonomy(chunk_text, model))

    # Gleaning
    print("      Gleaning...", end=" ", flush=True)
    prompt = PROMPT_GLEANING.format(
        extracted_terms=list(terms), chunk_text=chunk_text[:4000]
    )
    response = call_llm(prompt, model=model, temperature=0.3, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    terms.update(t["term"] for t in validated)
    print(f"+{len(validated)}", flush=True)

    # Verification
    print("      Verifying...", end=" ", flush=True)
    prompt = PROMPT_VERIFICATION.format(
        extracted_terms=list(terms), chunk_text=chunk_text[:4000]
    )
    response = call_llm(prompt, model=model, temperature=0.2, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    terms.update(t["term"] for t in validated)
    print(f"+{len(validated)}", flush=True)

    return list(terms)


def strategy_combined_max_recall(chunk_text: str, model: str) -> list[str]:
    """Strategy: Combine multiple approaches for maximum recall."""
    all_terms = set()

    # Method 1: Exhaustive taxonomy
    print("      Exhaustive...", end=" ", flush=True)
    terms1 = strategy_exhaustive_taxonomy(chunk_text, model)
    all_terms.update(terms1)
    print(f"{len(terms1)}", flush=True)

    # Method 2: Category-by-category
    terms2 = strategy_category_by_category(chunk_text, model)
    all_terms.update(terms2)

    # Method 3: Gleaning on combined results
    print("      Final gleaning...", end=" ", flush=True)
    prompt = PROMPT_GLEANING.format(
        extracted_terms=list(all_terms), chunk_text=chunk_text[:4000]
    )
    response = call_llm(prompt, model=model, temperature=0.3, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    all_terms.update(t["term"] for t in validated)
    print(f"+{len(validated)}", flush=True)

    return list(all_terms)


# ============================================================================
# Main
# ============================================================================


def main():
    # Load chunks
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    # Test first 5 chunks
    NUM_CHUNKS = 5
    chunks = gt["chunks"][:NUM_CHUNKS]

    # Only test the best strategy (A) on multiple chunks
    model = "claude-haiku"
    print(
        f"\nTesting Strategy A (Exhaustive+Taxonomy) on {NUM_CHUNKS} chunks", flush=True
    )
    print(f"{'=' * 70}", flush=True)

    all_metrics = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        gt_terms = chunk["terms"]

        print(f"\n  {chunk_id}: GT={len(gt_terms)} terms", end=" ", flush=True)

        try:
            start = time.time()
            terms = strategy_exhaustive_taxonomy(chunk_text, model)
            elapsed = time.time() - start
            metrics = calculate_metrics(terms, gt_terms)
            all_metrics.append(metrics)

            print(
                f"→ P={metrics['precision']:.0%} R={metrics['recall']:.0%} H={metrics['hallucination']:.0%} ({elapsed:.1f}s)",
                flush=True,
            )
            if metrics["missed"]:
                print(
                    f"    Missed: {metrics['missed'][:3]}{'...' if len(metrics['missed']) > 3 else ''}",
                    flush=True,
                )
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Aggregate metrics
    if all_metrics:
        avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
        avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
        avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)

        print(f"\n{'=' * 70}", flush=True)
        print(f"AGGREGATE RESULTS (Strategy A on {NUM_CHUNKS} chunks)", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  Average Precision:    {avg_p:.1%}", flush=True)
        print(f"  Average Recall:       {avg_r:.1%}", flush=True)
        print(f"  Average Hallucination: {avg_h:.1%}", flush=True)

        # Check if we hit targets
        print(f"\n  Targets: P>95%, R>95%, H<1%", flush=True)
        print(
            f"  Status: P={'PASS' if avg_p >= 0.95 else 'FAIL'}, R={'PASS' if avg_r >= 0.95 else 'FAIL'}, H={'PASS' if avg_h < 0.01 else 'FAIL'}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
