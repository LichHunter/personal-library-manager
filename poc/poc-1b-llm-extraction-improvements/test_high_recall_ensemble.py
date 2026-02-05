#!/usr/bin/env python3
"""High-recall ensemble extraction approach.

Strategy:
1. Run MULTIPLE extraction methods in parallel
2. Take UNION of all results (maximize candidates)
3. Apply STRICT span verification (deterministic, no LLM)
4. Output to human review queue

Target: 90%+ recall, <10% hallucination
Precision is not a concern (human review will filter)

Extraction methods:
- Quote-Extract (baseline, grounded)
- Quote-Extract + Gleaning (catch missed terms)
- Multiple category prompts (different angles)
- Liberal pattern extraction
- Use Sonnet for better comprehension
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: High-Recall Ensemble Extraction", flush=True)
print("=" * 70, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# STRICT SPAN VERIFICATION (Deterministic, No LLM)
# ============================================================================


def strict_span_verify(term: str, source_text: str) -> bool:
    """
    Verify term exists in source text. NO LLM involved.
    This is the hallucination firewall.
    """
    text_lower = source_text.lower()
    term_lower = term.lower().strip()

    if not term_lower or len(term_lower) < 2:
        return False

    # 1. Exact match (case-insensitive)
    if term_lower in text_lower:
        return True

    # 2. Handle underscores/hyphens/spaces interchangeably
    normalized_term = term_lower.replace("_", " ").replace("-", " ")
    normalized_text = text_lower.replace("_", " ").replace("-", " ")
    if normalized_term in normalized_text:
        return True

    # 3. Handle CamelCase split
    camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel_split != term_lower and camel_split in text_lower:
        return True

    # 4. Handle plural/singular for single words
    if " " not in term_lower:
        if term_lower.endswith("s") and term_lower[:-1] in text_lower:
            return True
        if (term_lower + "s") in text_lower:
            return True

    return False


def fuzzy_span_verify(term: str, source_text: str, threshold: int = 90) -> bool:
    """
    Fuzzy verification for edge cases.
    Still deterministic, but more lenient.
    """
    if strict_span_verify(term, source_text):
        return True

    text_lower = source_text.lower()
    term_lower = term.lower().strip()

    # Sliding window fuzzy match
    words = text_lower.split()
    term_word_count = max(1, len(term_lower.split()))

    for i in range(len(words) - term_word_count + 1):
        window = " ".join(words[i : i + term_word_count + 1])  # +1 for flexibility
        ratio = fuzz.ratio(term_lower, window)
        if ratio >= threshold:
            return True

    return False


# ============================================================================
# EXTRACTION PROMPTS
# ============================================================================

QUOTE_EXTRACT_PROMPT = """You are a Kubernetes documentation expert.

TASK: Extract ALL Kubernetes-specific terms from the text below.

For EACH term:
1. QUOTE: Copy the EXACT text (5-50 words) from the source containing the term
2. TERM: The Kubernetes term exactly as it appears

Include:
- Resource types (Pod, Service, Deployment, ConfigMap, etc.)
- Components (kubelet, kubectl, etcd, kube-proxy, etc.)
- Concepts (namespace, label, selector, controller, watch, etc.)
- Feature gates (CamelCase names)
- Lifecycle stages (alpha, beta, stable, GA, deprecated)
- API terms (object, spec, status, metadata, etc.)
- Error states (CrashLoopBackOff, OOMKilled, etc.)

Be EXHAUSTIVE - extract every technical term you can find.

<text>
{text}
</text>

Output JSON:
{{"entities": [{{"quote": "exact text", "term": "term"}}]}}"""


GLEANING_PROMPT = """You extracted: {previous_terms}

IMPORTANT: Many terms were likely missed. Review CAREFULLY and find MORE.

Look for:
- Feature gates (CamelCase like ServiceAppProtocol)
- Components (kubelet, etcd, coredns)
- Resources (Pod, Service, ConfigMap) 
- Concepts (namespace, label, annotation, watch)
- Lifecycle (alpha, beta, stable, deprecated)
- API terms (object, spec, status)
- Abbreviations (PV, PVC, HPA, CRD)

<text>
{text}
</text>

For EACH missed term, provide quote and term:
{{"missed": [{{"quote": "exact text", "term": "term"}}]}}"""


CATEGORY_PROMPTS = {
    "resources_and_objects": """Extract ALL Kubernetes resources and API objects:
- Core resources: Pod, Service, Deployment, ConfigMap, Secret, Namespace, Node
- Workloads: StatefulSet, DaemonSet, ReplicaSet, Job, CronJob
- Storage: PersistentVolume, PersistentVolumeClaim, StorageClass
- Networking: Ingress, NetworkPolicy, Endpoint
- Config: ConfigMap, Secret, ServiceAccount
- CRDs and custom resources

<text>
{text}
</text>

JSON: {{"entities": [{{"quote": "exact text", "term": "resource"}}]}}""",
    "components_and_tools": """Extract ALL Kubernetes components and tools:
- Node components: kubelet, kube-proxy, container runtime
- Control plane: kube-apiserver, kube-scheduler, kube-controller-manager, etcd
- CLI tools: kubectl, kubeadm, helm
- Add-ons: coredns, metrics-server
- Container runtimes: containerd, CRI-O, Docker

<text>
{text}
</text>

JSON: {{"entities": [{{"quote": "exact text", "term": "component"}}]}}""",
    "concepts_and_api": """Extract ALL Kubernetes concepts and API terms:
- Core concepts: namespace, label, selector, annotation
- Scheduling: taint, toleration, affinity, nodeSelector
- Controllers: controller, operator, reconciler
- API: object, spec, status, metadata, apiVersion, kind
- Lifecycle: watch, list, get, create, update, delete
- Advanced: finalizer, webhook, admission controller

<text>
{text}
</text>

JSON: {{"entities": [{{"quote": "exact text", "term": "concept"}}]}}""",
    "feature_gates_and_lifecycle": """Extract ALL feature gates and lifecycle terms:
- Feature gates: CamelCase names (ServiceAppProtocol, JobPodFailurePolicy, etc.)
- Lifecycle stages: alpha, beta, stable, GA, deprecated, removed
- Version references: v1, v1beta1, v1alpha1

<text>
{text}
</text>

JSON: {{"entities": [{{"quote": "exact text", "term": "term"}}]}}""",
    "technical_terms": """Extract ALL technical/domain terms that a reader might need defined:
- Any term that would benefit from a glossary entry
- Any abbreviation or acronym (CNI, CSI, CRI, PV, PVC, HPA)
- Any term specific to Kubernetes or container orchestration
- Any term that appears in backticks or code formatting

<text>
{text}
</text>

JSON: {{"entities": [{{"quote": "exact text", "term": "term"}}]}}""",
}


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def parse_response(response: str) -> list[dict]:
    """Parse LLM response to extract entities."""
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


def extract_with_prompt(text: str, prompt: str, model: str) -> list[str]:
    """Run extraction with a prompt and return verified terms."""
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)
    entities = parse_response(response)

    terms = []
    for e in entities:
        term = e.get("term", "")
        if term and strict_span_verify(term, text):
            terms.append(term)

    return terms


def extract_quote_extract(text: str, model: str = "claude-sonnet") -> list[str]:
    """Basic quote-extract."""
    prompt = QUOTE_EXTRACT_PROMPT.format(text=text[:4000])
    return extract_with_prompt(text, prompt, model)


def extract_with_gleaning(
    text: str, model: str = "claude-sonnet", rounds: int = 2
) -> list[str]:
    """Quote-extract with gleaning rounds."""
    all_terms = set()

    # Initial extraction
    initial = extract_quote_extract(text, model)
    all_terms.update(initial)

    # Gleaning rounds
    for _ in range(rounds):
        prompt = GLEANING_PROMPT.format(
            previous_terms=json.dumps(list(all_terms)),
            text=text[:3500],
        )
        response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
        entities = parse_response(response)

        new_terms = 0
        for e in entities:
            term = e.get("term", "")
            if term and strict_span_verify(term, text):
                if term.lower() not in {t.lower() for t in all_terms}:
                    all_terms.add(term)
                    new_terms += 1

        if new_terms == 0:
            break

    return list(all_terms)


def extract_category(
    text: str, category: str, model: str = "claude-sonnet"
) -> list[str]:
    """Extract terms for a specific category."""
    if category not in CATEGORY_PROMPTS:
        return []

    prompt = CATEGORY_PROMPTS[category].format(text=text[:3500])
    return extract_with_prompt(text, prompt, model)


def extract_all_categories(text: str, model: str = "claude-sonnet") -> list[str]:
    """Run all category prompts and return union."""
    all_terms = set()

    for category in CATEGORY_PROMPTS:
        terms = extract_category(text, category, model)
        all_terms.update(terms)

    return list(all_terms)


# ============================================================================
# ENSEMBLE PIPELINE
# ============================================================================


def extract_ensemble(
    text: str,
    model: str = "claude-sonnet",
    use_gleaning: bool = True,
    use_categories: bool = True,
    gleaning_rounds: int = 2,
) -> dict:
    """
    Ensemble extraction: union of multiple methods.
    All results pass through strict span verification.
    """
    all_terms = set()
    sources = {}  # Track where each term came from

    # Method 1: Quote-Extract baseline
    baseline_terms = extract_quote_extract(text, model)
    for t in baseline_terms:
        if t.lower() not in {x.lower() for x in all_terms}:
            all_terms.add(t)
            sources[t.lower()] = "baseline"
    baseline_count = len(baseline_terms)

    # Method 2: Gleaning
    gleaning_count = 0
    if use_gleaning:
        gleaning_terms = extract_with_gleaning(text, model, gleaning_rounds)
        for t in gleaning_terms:
            if t.lower() not in {x.lower() for x in all_terms}:
                all_terms.add(t)
                sources[t.lower()] = "gleaning"
                gleaning_count += 1

    # Method 3: Category prompts
    category_count = 0
    if use_categories:
        category_terms = extract_all_categories(text, model)
        for t in category_terms:
            if t.lower() not in {x.lower() for x in all_terms}:
                all_terms.add(t)
                sources[t.lower()] = "category"
                category_count += 1

    # Final verification pass (should already be verified, but double-check)
    final_terms = [t for t in all_terms if strict_span_verify(t, text)]

    return {
        "final_terms": final_terms,
        "baseline_count": baseline_count,
        "gleaning_added": gleaning_count,
        "category_added": category_count,
        "total_before_verify": len(all_terms),
        "total_after_verify": len(final_terms),
        "sources": sources,
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
        tier_gt_indices = [
            i for i in range(len(gt_list)) if gt_tiers.get(gt_list[i], 1) == tier
        ]
        tier_matched = sum(1 for i in tier_gt_indices if i in matched_gt)
        if tier_gt_indices:
            tier_metrics[f"tier_{tier}_recall"] = tier_matched / len(tier_gt_indices)
            tier_metrics[f"tier_{tier}_count"] = len(tier_gt_indices)

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


def run_ensemble_experiment(
    model: str = "claude-sonnet",
    num_chunks: int = 20,
    use_gleaning: bool = True,
    use_categories: bool = True,
):
    """Run ensemble extraction experiment."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"ENSEMBLE EXTRACTION ({model})", flush=True)
    print(f"  Gleaning: {use_gleaning}, Categories: {use_categories}", flush=True)
    print("=" * 70, flush=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunks = gt["chunks"][:num_chunks]
    all_metrics = []

    for chunk in chunks:
        start = time.time()
        result = extract_ensemble(
            chunk["text"],
            model=model,
            use_gleaning=use_gleaning,
            use_categories=use_categories,
        )
        elapsed = time.time() - start

        metrics = calculate_metrics(result["final_terms"], chunk["terms"])
        all_metrics.append(metrics)

        # Status markers
        r_mark = (
            "OK"
            if metrics["recall"] >= 0.90
            else ("~" if metrics["recall"] >= 0.80 else "")
        )
        h_mark = (
            "OK"
            if metrics["hallucination"] < 0.10
            else ("~" if metrics["hallucination"] < 0.15 else "")
        )

        print(
            f"  {chunk['chunk_id']}: R={metrics['recall']:.0%}{r_mark} H={metrics['hallucination']:.0%}{h_mark} "
            f"({elapsed:.1f}s, base={result['baseline_count']}, +glean={result['gleaning_added']}, +cat={result['category_added']})",
            flush=True,
        )

        if metrics["missed"] and len(metrics["missed"]) <= 3:
            print(f"    Missed: {metrics['missed']}", flush=True)

    # Aggregate
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    # Count chunks meeting targets
    chunks_90_recall = sum(1 for m in all_metrics if m["recall"] >= 0.90)
    chunks_10_halluc = sum(1 for m in all_metrics if m["hallucination"] < 0.10)
    chunks_both = sum(
        1 for m in all_metrics if m["recall"] >= 0.90 and m["hallucination"] < 0.10
    )

    print(f"\n{'=' * 70}", flush=True)
    print(f"RESULTS: Ensemble ({model})", flush=True)
    print("=" * 70, flush=True)

    r_mark = "OK" if avg_r >= 0.90 else ("~" if avg_r >= 0.80 else "")
    h_mark = "OK" if avg_h < 0.10 else ("~" if avg_h < 0.15 else "")

    print(f"  Precision:     {avg_p:.1%}", flush=True)
    print(f"  Recall:        {avg_r:.1%} {r_mark}  (Target: 90%+)", flush=True)
    print(f"  Hallucination: {avg_h:.1%} {h_mark}  (Target: <10%)", flush=True)
    print(f"  F1:            {avg_f1:.1%}", flush=True)

    print(f"\n  Chunks with R>=90%: {chunks_90_recall}/{num_chunks}", flush=True)
    print(f"  Chunks with H<10%:  {chunks_10_halluc}/{num_chunks}", flush=True)
    print(f"  Chunks meeting BOTH: {chunks_both}/{num_chunks}", flush=True)

    # Tier breakdown
    tier_recalls = {1: [], 2: [], 3: []}
    for m in all_metrics:
        for tier in [1, 2, 3]:
            key = f"tier_{tier}_recall"
            if key in m.get("tier_metrics", {}):
                tier_recalls[tier].append(m["tier_metrics"][key])

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
        "chunks_90_recall": chunks_90_recall,
        "chunks_10_halluc": chunks_10_halluc,
        "chunks_both": chunks_both,
        "tier_recalls": {
            k: sum(v) / len(v) if v else 0 for k, v in tier_recalls.items()
        },
    }


def run_all_experiments():
    """Run multiple configurations to find best high-recall approach."""
    print("\n" + "=" * 70, flush=True)
    print("HIGH-RECALL ENSEMBLE EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)
    print("\nTarget: Recall 90%+, Hallucination <10%", flush=True)

    results = {}

    # Configuration 1: Sonnet with full ensemble (20 chunks)
    print("\n\n>>> EXPERIMENT 1: Sonnet Full Ensemble", flush=True)
    results["sonnet_full"] = run_ensemble_experiment(
        model="claude-sonnet",
        num_chunks=20,
        use_gleaning=True,
        use_categories=True,
    )

    # Configuration 2: Sonnet baseline only (no gleaning/categories)
    print("\n\n>>> EXPERIMENT 2: Sonnet Baseline Only", flush=True)
    results["sonnet_baseline"] = run_ensemble_experiment(
        model="claude-sonnet",
        num_chunks=20,
        use_gleaning=False,
        use_categories=False,
    )

    # Configuration 3: Sonnet with gleaning only
    print("\n\n>>> EXPERIMENT 3: Sonnet + Gleaning", flush=True)
    results["sonnet_gleaning"] = run_ensemble_experiment(
        model="claude-sonnet",
        num_chunks=20,
        use_gleaning=True,
        use_categories=False,
    )

    # Final comparison
    print("\n\n" + "=" * 70, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("=" * 70, flush=True)

    print(
        f"{'Configuration':<25} {'Recall':>10} {'Halluc':>10} {'Precision':>10} {'Chunks R>=90%':>15}",
        flush=True,
    )
    print("-" * 75, flush=True)

    for name, data in results.items():
        r_mark = (
            "OK" if data["recall"] >= 0.90 else ("~" if data["recall"] >= 0.80 else "")
        )
        h_mark = (
            "OK"
            if data["hallucination"] < 0.10
            else ("~" if data["hallucination"] < 0.15 else "")
        )

        print(
            f"{name:<25} {data['recall']:>8.1%} {r_mark} {data['hallucination']:>8.1%} {h_mark} {data['precision']:>10.1%} {data['chunks_90_recall']:>10}/20",
            flush=True,
        )

    # Save results
    results_path = (
        Path(__file__).parent / "artifacts" / "high_recall_ensemble_results.json"
    )
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return results


if __name__ == "__main__":
    run_all_experiments()
