#!/usr/bin/env python3
"""Test the discrimination-based extraction approach.

Key insight from Oracle: LLMs are better discriminators than generators.

Architecture:
1. PATTERN EXTRACTION: Extract candidates via patterns (backticks, CamelCase, code blocks)
2. LLM DISCRIMINATION: Ask LLM to classify each candidate (yes/no) - cannot hallucinate
3. CATEGORY-TARGETED EXPANSION: Narrow prompts for specific term types (feature gates, resources, etc.)
4. SPAN VERIFICATION: Final filter - reject anything not in source text

Target: 95%+ precision, 95%+ recall, <1% hallucination
"""

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Discrimination-Based Extraction", flush=True)
print("=" * 70, flush=True)

POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"


# ============================================================================
# PHASE 1: PATTERN-BASED CANDIDATE EXTRACTION
# ============================================================================


def extract_backticked_terms(text: str) -> set[str]:
    """Extract terms in backticks - high confidence technical terms."""
    pattern = r"`([^`]+)`"
    matches = re.findall(pattern, text)
    # Filter out obvious non-terms
    terms = set()
    for m in matches:
        m = m.strip()
        # Skip if too long (likely code), too short, or contains obvious non-term patterns
        if len(m) > 50 or len(m) < 2:
            continue
        if re.match(r"^[\d\.\-]+$", m):  # Pure version numbers
            continue
        if m.startswith("/") and "/" in m[1:]:  # Paths
            continue
        terms.add(m)
    return terms


def extract_camelcase_terms(text: str) -> set[str]:
    """Extract CamelCase terms - feature gates, resource types."""
    # Match CamelCase: starts with capital, has at least one more capital
    pattern = r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b"
    matches = re.findall(pattern, text)
    return set(matches)


def extract_code_block_identifiers(text: str) -> set[str]:
    """Extract identifiers from code blocks (YAML, shell, etc.)."""
    terms = set()

    # Find code blocks
    code_block_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_block_pattern, text)

    for block in code_blocks:
        # Extract YAML keys (left side of colon)
        yaml_keys = re.findall(
            r"^\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*:", block, re.MULTILINE
        )
        for key in yaml_keys:
            # Skip generic YAML keys
            if key.lower() not in {
                "title",
                "content_type",
                "stage",
                "defaultvalue",
                "fromversion",
                "toversion",
                "list",
                "render",
                "build",
                "id",
                "date",
                "full_link",
                "short_description",
                "aka",
                "tags",
                "removed",
                "stages",
            }:
                terms.add(key)

        # Extract command names (kubectl, kubeadm, etc.)
        commands = re.findall(r"\b(kubectl|kubeadm|kubelet|kube-\w+|helm)\b", block)
        terms.update(commands)

        # Extract resource references in kubectl commands
        kubectl_resources = re.findall(r"kubectl\s+\w+\s+(\w+)", block)
        terms.update(kubectl_resources)

    return terms


def extract_capitalized_phrases(text: str) -> set[str]:
    """Extract capitalized multi-word phrases that might be technical terms."""
    terms = set()

    # Two-word capitalized phrases (e.g., "Service Account", "Deployment Controller")
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    matches = re.findall(pattern, text)

    for m in matches:
        # Filter out sentence starts by checking if preceded by period/newline
        # This is a heuristic - not perfect
        if len(m.split()) <= 3:  # Max 3 words
            terms.add(m)

    return terms


def extract_hyphenated_compounds(text: str) -> set[str]:
    """Extract hyphenated technical terms."""
    pattern = r"\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b"
    matches = re.findall(pattern, text.lower())

    terms = set()
    for m in matches:
        # Filter common non-technical hyphenated words
        if m not in {
            "built-in",
            "well-known",
            "so-called",
            "real-time",
            "up-to-date",
            "e-g",
            "i-e",
        }:
            terms.add(m)

    return terms


def extract_all_candidates(text: str) -> dict[str, set[str]]:
    """Extract all candidate terms using pattern-based methods."""
    return {
        "backticked": extract_backticked_terms(text),
        "camelcase": extract_camelcase_terms(text),
        "code_blocks": extract_code_block_identifiers(text),
        "capitalized": extract_capitalized_phrases(text),
        "hyphenated": extract_hyphenated_compounds(text),
    }


def merge_candidates(candidates_by_type: dict[str, set[str]]) -> list[str]:
    """Merge all candidates, removing duplicates (case-insensitive)."""
    seen_lower = set()
    merged = []

    # Priority order: backticked > camelcase > code_blocks > capitalized > hyphenated
    for source in [
        "backticked",
        "camelcase",
        "code_blocks",
        "capitalized",
        "hyphenated",
    ]:
        for term in candidates_by_type.get(source, []):
            if term.lower() not in seen_lower:
                seen_lower.add(term.lower())
                merged.append(term)

    return merged


# ============================================================================
# PHASE 2: LLM DISCRIMINATION (Binary Classification)
# ============================================================================

DISCRIMINATION_PROMPT = """You are a technical documentation expert.

TASK: For each candidate term below, determine if it's a technical term worth indexing.

CONTEXT (source document):
{context}

CANDIDATE TERMS:
{candidates}

RULES:
- Answer "yes" ONLY for domain-specific technical terms (Kubernetes resources, components, concepts, feature gates, etc.)
- Answer "no" for generic words, YAML keywords, version numbers, or non-technical terms
- A term must appear in the context to be valid

OUTPUT FORMAT (JSON array):
[
  {{"term": "Pod", "verdict": "yes", "reason": "K8s resource"}},
  {{"term": "title", "verdict": "no", "reason": "YAML metadata"}}
]

Be conservative - when in doubt, say "no"."""


def discriminate_candidates_llm(
    candidates: list[str],
    context: str,
    model: str = "claude-haiku",
    batch_size: int = 15,
) -> list[str]:
    """Use LLM to discriminate which candidates are real technical terms."""

    if not candidates:
        return []

    approved = []

    # Process in batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        candidates_str = "\n".join(f"- {term}" for term in batch)

        prompt = DISCRIMINATION_PROMPT.format(
            context=context[:3000],
            candidates=candidates_str,
        )

        response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)

        # Parse response
        try:
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)

            # Find JSON array
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                verdicts = json.loads(json_match.group())
                for v in verdicts:
                    if v.get("verdict", "").lower() == "yes":
                        approved.append(v.get("term", ""))
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"  Warning: Failed to parse discrimination response: {e}", flush=True
            )

    return approved


# ============================================================================
# PHASE 3: CATEGORY-TARGETED EXTRACTION
# ============================================================================

CATEGORY_PROMPTS = {
    "resources": """Extract ONLY Kubernetes resource types mentioned in this text.
Examples of resource types: Pod, Service, Deployment, ConfigMap, Secret, Ingress, etc.

Text:
{text}

Output JSON: {{"resources": ["Resource1", "Resource2"]}}
Only include terms that APPEAR EXACTLY in the text above.""",
    "components": """Extract ONLY Kubernetes component/process names mentioned in this text.
Examples: kubelet, kubectl, kube-proxy, kube-apiserver, etcd, coredns, etc.

Text:
{text}

Output JSON: {{"components": ["component1", "component2"]}}
Only include terms that APPEAR EXACTLY in the text above.""",
    "feature_gates": """Extract ONLY Kubernetes feature gate names mentioned in this text.
Feature gates are CamelCase names like: ServiceAppProtocol, JobPodFailurePolicy, etc.

Text:
{text}

Output JSON: {{"feature_gates": ["FeatureGate1", "FeatureGate2"]}}
Only include terms that APPEAR EXACTLY in the text above.""",
    "concepts": """Extract ONLY Kubernetes concepts and API terms mentioned in this text.
Examples: namespace, label, selector, annotation, watch, controller, object, spec, status, etc.

Text:
{text}

Output JSON: {{"concepts": ["concept1", "concept2"]}}
Only include terms that APPEAR EXACTLY in the text above.""",
    "lifecycle": """Extract ONLY lifecycle/maturity stage terms mentioned in this text.
Examples: alpha, beta, stable, GA, deprecated, removed, etc.

Text:
{text}

Output JSON: {{"lifecycle": ["stage1", "stage2"]}}
Only include terms that APPEAR EXACTLY in the text above.""",
}


def extract_category_targeted(
    text: str,
    model: str = "claude-haiku",
) -> dict[str, list[str]]:
    """Run category-targeted extraction prompts."""
    results = {}

    for category, prompt_template in CATEGORY_PROMPTS.items():
        prompt = prompt_template.format(text=text[:3000])
        response = call_llm(prompt, model=model, temperature=0, max_tokens=500)

        try:
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                terms = data.get(category, [])
                # Span verify - only keep terms that exist in text
                text_lower = text.lower()
                verified = [t for t in terms if t.lower() in text_lower]
                results[category] = verified
        except (json.JSONDecodeError, KeyError):
            results[category] = []

    return results


# ============================================================================
# PHASE 4: SPAN VERIFICATION
# ============================================================================


def span_verify(terms: list[str], source_text: str) -> list[str]:
    """Final filter - only keep terms that appear in source text."""
    text_lower = source_text.lower()
    verified = []

    for term in terms:
        # Exact match (case-insensitive)
        if term.lower() in text_lower:
            verified.append(term)
        # Handle whitespace variations (e.g., "CrashLoopBackOff" vs "crash loop back off")
        elif re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower() in text_lower:
            verified.append(term)

    return verified


# ============================================================================
# FULL PIPELINE
# ============================================================================


def extract_discrimination_pipeline(
    text: str,
    model: str = "claude-haiku",
    use_categories: bool = True,
) -> dict:
    """Full discrimination-based extraction pipeline."""

    # Phase 1: Pattern-based candidate extraction
    candidates_by_type = extract_all_candidates(text)
    all_candidates = merge_candidates(candidates_by_type)

    # Phase 2: LLM discrimination
    discriminated = discriminate_candidates_llm(all_candidates, text, model)

    # Phase 3: Category-targeted extraction (for terms patterns might miss)
    category_terms = []
    if use_categories:
        category_results = extract_category_targeted(text, model)
        for category, terms in category_results.items():
            category_terms.extend(terms)

    # Phase 4: Merge and deduplicate
    all_terms = set(discriminated)
    all_terms.update(category_terms)

    # Final span verification
    final_terms = span_verify(list(all_terms), text)

    return {
        "candidates_by_type": {k: list(v) for k, v in candidates_by_type.items()},
        "total_candidates": len(all_candidates),
        "discriminated": discriminated,
        "category_terms": category_terms,
        "final_terms": final_terms,
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

    # Get missed terms for analysis
    missed = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
    false_positives = [
        extracted_terms[i]
        for i in range(len(extracted_terms))
        if i not in matched_extracted
    ]

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
    }


# ============================================================================
# EXPERIMENTS
# ============================================================================


def run_pattern_only_baseline():
    """Test pattern extraction alone (no LLM)."""
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Pattern Extraction Baseline (No LLM)", flush=True)
    print("=" * 70, flush=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunks = gt["chunks"]
    all_metrics = []

    for chunk in chunks:
        candidates_by_type = extract_all_candidates(chunk["text"])
        all_candidates = merge_candidates(candidates_by_type)

        # Span verify all candidates
        verified = span_verify(all_candidates, chunk["text"])

        metrics = calculate_metrics(verified, chunk["terms"])
        all_metrics.append(metrics)

        print(
            f"  {chunk['chunk_id']}: P={metrics['precision']:.0%} R={metrics['recall']:.0%} "
            f"(candidates={len(all_candidates)}, verified={len(verified)}, gt={len(chunk['terms'])})",
            flush=True,
        )

    # Aggregate
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)

    print(f"\nPattern Baseline: P={avg_p:.1%} R={avg_r:.1%} H={avg_h:.1%}", flush=True)
    return {"precision": avg_p, "recall": avg_r, "hallucination": avg_h}


def run_discrimination_experiment(
    model: str = "claude-haiku", use_categories: bool = True, num_chunks: int = 20
):
    """Test full discrimination pipeline."""
    print("\n" + "=" * 70, flush=True)
    print(
        f"EXPERIMENT 2: Discrimination Pipeline ({model}, categories={use_categories})",
        flush=True,
    )
    print("=" * 70, flush=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunks = gt["chunks"][:num_chunks]
    all_metrics = []
    all_results = []

    for chunk in chunks:
        start = time.time()
        result = extract_discrimination_pipeline(
            chunk["text"],
            model=model,
            use_categories=use_categories,
        )
        elapsed = time.time() - start

        metrics = calculate_metrics(result["final_terms"], chunk["terms"])
        all_metrics.append(metrics)

        all_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "result": result,
                "metrics": metrics,
                "elapsed": elapsed,
            }
        )

        print(
            f"  {chunk['chunk_id']}: P={metrics['precision']:.0%} R={metrics['recall']:.0%} H={metrics['hallucination']:.0%} "
            f"({elapsed:.1f}s, terms={len(result['final_terms'])})",
            flush=True,
        )

        if metrics["missed"]:
            print(
                f"    Missed: {metrics['missed'][:5]}{'...' if len(metrics['missed']) > 5 else ''}",
                flush=True,
            )
        if metrics["false_positives"]:
            print(
                f"    FP: {metrics['false_positives'][:5]}{'...' if len(metrics['false_positives']) > 5 else ''}",
                flush=True,
            )

    # Aggregate
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_h = sum(m["hallucination"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    print(f"\nDiscrimination Pipeline ({model}):", flush=True)
    print(
        f"  Precision:     {avg_p:.1%} {'OK' if avg_p >= 0.95 else ('~' if avg_p >= 0.85 else '')}",
        flush=True,
    )
    print(
        f"  Recall:        {avg_r:.1%} {'OK' if avg_r >= 0.95 else ('~' if avg_r >= 0.70 else '')}",
        flush=True,
    )
    print(
        f"  Hallucination: {avg_h:.1%} {'OK' if avg_h < 0.01 else ('~' if avg_h < 0.05 else '')}",
        flush=True,
    )
    print(f"  F1:            {avg_f1:.1%}", flush=True)

    return {
        "precision": avg_p,
        "recall": avg_r,
        "hallucination": avg_h,
        "f1": avg_f1,
        "results": all_results,
    }


def run_all_experiments():
    """Run all experiments and compare."""
    print("\n" + "=" * 70, flush=True)
    print("DISCRIMINATION APPROACH - FULL EXPERIMENT SUITE", flush=True)
    print("=" * 70, flush=True)
    print("\nTarget: P>95%, R>95%, H<1%", flush=True)

    results = {}

    # Experiment 1: Pattern baseline
    results["pattern_only"] = run_pattern_only_baseline()

    # Experiment 2a: Discrimination only (no categories)
    results["discrimination_no_cat"] = run_discrimination_experiment(
        model="claude-haiku",
        use_categories=False,
        num_chunks=20,
    )

    # Experiment 2b: Full discrimination + categories
    results["discrimination_full"] = run_discrimination_experiment(
        model="claude-haiku",
        use_categories=True,
        num_chunks=20,
    )

    # Experiment 2c: Sonnet (if time permits)
    results["discrimination_sonnet"] = run_discrimination_experiment(
        model="claude-sonnet",
        use_categories=True,
        num_chunks=10,  # Fewer due to cost
    )

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Approach':<30} {'Precision':>12} {'Recall':>12} {'Hallucination':>14}",
        flush=True,
    )
    print("-" * 70, flush=True)

    for name, data in results.items():
        if isinstance(data, dict) and "precision" in data:
            p_status = (
                "OK"
                if data["precision"] >= 0.95
                else ("~" if data["precision"] >= 0.85 else "  ")
            )
            r_status = (
                "OK"
                if data["recall"] >= 0.95
                else ("~" if data["recall"] >= 0.70 else "  ")
            )
            h_status = (
                "OK"
                if data["hallucination"] < 0.01
                else ("~" if data["hallucination"] < 0.05 else "  ")
            )

            print(
                f"{name:<30} {data['precision']:>10.1%} {p_status} {data['recall']:>10.1%} {r_status} {data['hallucination']:>12.1%} {h_status}",
                flush=True,
            )

    # Save results
    results_path = Path(__file__).parent / "artifacts" / "discrimination_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Clean up non-serializable data
    clean_results = {}
    for name, data in results.items():
        if isinstance(data, dict):
            clean_results[name] = {k: v for k, v in data.items() if k != "results"}

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return results


if __name__ == "__main__":
    run_all_experiments()
