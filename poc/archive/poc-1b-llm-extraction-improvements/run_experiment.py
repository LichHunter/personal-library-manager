#!/usr/bin/env python3
"""POC-1b: LLM Term Extraction Improvements

Tests 5 strategies to achieve 95%+ precision/recall with <1% hallucination:
- E: Structured output with Pydantic validation
- F: Structured output + span verification
- G: Self-consistency voting (N=10, 70% agreement)
- H: Multi-pass extraction ("what did I miss?")
- I: Combined pipeline (F + G + H)

Uses same 45 chunks and ground truth from POC-1.
Uses the same LLM provider as POC-1 (OAuth-based httpx client).
"""

import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from rapidfuzz import fuzz

# Add POC-1 utils to path for LLM provider
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm, AnthropicProvider

print("POC-1b: LLM Term Extraction Improvements", flush=True)
print("=" * 50, flush=True)

# Paths
POC_DIR = Path(__file__).parent
POC1_DIR = POC_DIR.parent / "poc-1-llm-extraction-guardrails"
ARTIFACTS_DIR = POC_DIR / "artifacts"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"

# Configuration
TRIALS = 3
MODELS = ["claude-haiku", "claude-sonnet"]
MODEL_MAP = {
    "claude-haiku": "claude-3-5-haiku-latest",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
}

# Self-consistency configuration
SC_NUM_SAMPLES = 10
SC_AGREEMENT_THRESHOLD = 0.7  # 70% must agree
SC_TEMPERATURE = 0.8


# ============================================================================
# Pydantic Models for Validation
# ============================================================================


class ExtractedTermBasic(BaseModel):
    """Basic extracted term."""

    term: str = Field(description="Kubernetes domain-specific term")
    span: str = Field(description="Exact quote from source containing the term")
    confidence: str = Field(pattern="^(HIGH|MEDIUM|LOW)$")


class ExtractedTermVerified(BaseModel):
    """Extracted term with span verification."""

    term: str
    span: str
    confidence: str

    # Store source text for validation
    _source_text: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_span_exists(cls, values):
        """Verify span exists in source text."""
        source_text = values.pop("_source_text", "")
        span = values.get("span", "")
        term = values.get("term", "")

        if source_text and span:
            if span not in source_text:
                raise ValueError(f"Span '{span[:50]}...' not found in source text")
            if term.lower() not in span.lower():
                raise ValueError(f"Term '{term}' not found in its span")

        return values


# ============================================================================
# Prompt Templates
# ============================================================================

PROMPT_STRUCTURED = """Extract Kubernetes domain-specific terms from this documentation chunk.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide an EXACT text span (10-50 words) where it appears
3. Assign confidence: HIGH (K8s-specific), MEDIUM (technical), LOW (generic)
4. Maximum 15 terms per chunk
5. Prefer multi-word terms over single words

TEXT:
---
{chunk_text}
---

OUTPUT FORMAT (JSON only, no markdown):
{{"terms": [{{"term": "example", "span": "exact quote containing example", "confidence": "HIGH"}}]}}"""

PROMPT_MULTI_PASS_1 = """Extract ALL Kubernetes domain-specific terms from this text.

Be EXHAUSTIVE - include every single relevant term, even if uncertain.
When in doubt, INCLUDE IT. You will be penalized for MISSING terms.

For each term, provide:
- The exact term
- An exact quote (10-50 words) from the text containing it
- Confidence: HIGH (K8s-specific), MEDIUM (technical), LOW (generic)

TEXT:
---
{chunk_text}
---

OUTPUT FORMAT (JSON only):
{{"terms": [{{"term": "...", "span": "...", "confidence": "HIGH|MEDIUM|LOW"}}]}}"""

PROMPT_MULTI_PASS_2 = """I previously extracted these Kubernetes terms: {extracted_terms}

Review the text AGAIN and identify ANY terms I might have MISSED.
You will be penalized for missing terms, not for including extras.

Focus on:
- Abbreviations and acronyms (K8s, CRD, HPA, etc.)
- Multi-word technical terms (Pod Security Policy, etc.)
- Configuration fields and API objects (spec.containers, etc.)
- Error states (CrashLoopBackOff, OOMKilled, etc.)

TEXT:
---
{chunk_text}
---

Return ONLY the additional terms I missed (JSON format):
{{"terms": [{{"term": "...", "span": "...", "confidence": "..."}}]}}"""

PROMPT_MULTI_PASS_3 = """Already found: {extracted_terms}

Final sweep for specific Kubernetes categories:
- Resource types (Pod, Service, Deployment, StatefulSet)
- Commands and tools (kubectl, kubeadm, kubelet)
- Namespace and labels (kube-system, app=nginx)
- Status conditions and phases

TEXT:
---
{chunk_text}
---

Any additional terms? (JSON format):
{{"terms": [{{"term": "...", "span": "...", "confidence": "..."}}]}}"""


# ============================================================================
# Response Parsing
# ============================================================================


def parse_json_response(response: str) -> list[dict]:
    """Parse JSON response, handling markdown code blocks."""
    response = response.strip()

    # Remove markdown code blocks
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    # Try to find JSON object
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("terms", [])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array
    json_match = re.search(r"\[[\s\S]*\]", response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return []


def validate_terms_basic(terms: list[dict]) -> list[dict]:
    """Validate terms using basic Pydantic model."""
    validated = []
    for t in terms:
        try:
            term = ExtractedTermBasic(**t)
            validated.append(
                {"term": term.term, "span": term.span, "confidence": term.confidence}
            )
        except ValidationError:
            pass
    return validated


def validate_terms_with_source(terms: list[dict], source_text: str) -> list[dict]:
    """Validate terms and verify spans exist in source."""
    validated = []
    for t in terms:
        try:
            # Add source text for validation
            t_with_source = {**t, "_source_text": source_text}
            term = ExtractedTermVerified(**t_with_source)
            validated.append(
                {"term": term.term, "span": term.span, "confidence": term.confidence}
            )
        except (ValidationError, ValueError):
            pass
    return validated


# ============================================================================
# Metrics Calculation
# ============================================================================


def normalize_term(term: str) -> str:
    """Normalize term for comparison."""
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted: str, ground_truth: str) -> str:
    """Three-level matching: exact, partial, fuzzy."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    if ext_norm == gt_norm:
        return "exact"

    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens:
        overlap = len(ext_tokens & gt_tokens) / len(gt_tokens)
        if overlap >= 0.8:
            return "partial"

    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return "fuzzy"

    return "no_match"


@dataclass
class MetricsResult:
    precision: float
    recall: float
    hallucination_rate: float
    exact_matches: int
    partial_matches: int
    fuzzy_matches: int
    false_positives: int
    false_negatives: int
    total_extracted: int
    total_ground_truth: int


def calculate_metrics(
    extracted_terms: list[str], ground_truth_terms: list[dict]
) -> MetricsResult:
    """Calculate precision, recall, hallucination rate."""
    gt_terms = [t.get("term", "") for t in ground_truth_terms]

    matched_gt = set()
    exact_matches = 0
    partial_matches = 0
    fuzzy_matches = 0

    for ext in extracted_terms:
        best_match = None
        best_match_type = "no_match"

        for i, gt in enumerate(gt_terms):
            if i in matched_gt:
                continue
            match_type = match_terms(ext, gt)
            if match_type == "exact":
                best_match = i
                best_match_type = "exact"
                break
            elif match_type == "partial" and best_match_type not in [
                "exact",
                "partial",
            ]:
                best_match = i
                best_match_type = "partial"
            elif match_type == "fuzzy" and best_match_type == "no_match":
                best_match = i
                best_match_type = "fuzzy"

        if best_match is not None:
            matched_gt.add(best_match)
            if best_match_type == "exact":
                exact_matches += 1
            elif best_match_type == "partial":
                partial_matches += 1
            else:
                fuzzy_matches += 1

    true_positives = exact_matches + partial_matches + fuzzy_matches
    false_positives = len(extracted_terms) - true_positives
    false_negatives = len(gt_terms) - len(matched_gt)

    precision = true_positives / len(extracted_terms) if extracted_terms else 0.0
    recall = true_positives / len(gt_terms) if gt_terms else 0.0
    hallucination_rate = (
        false_positives / len(extracted_terms) if extracted_terms else 0.0
    )

    return MetricsResult(
        precision=precision,
        recall=recall,
        hallucination_rate=hallucination_rate,
        exact_matches=exact_matches,
        partial_matches=partial_matches,
        fuzzy_matches=fuzzy_matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_extracted=len(extracted_terms),
        total_ground_truth=len(gt_terms),
    )


# ============================================================================
# Strategy Implementations
# ============================================================================


def strategy_e_structured(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy E: Structured output with basic Pydantic validation."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])

    start = time.time()
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)

    raw_terms = parse_json_response(response)
    validated = validate_terms_basic(raw_terms)
    terms = [t["term"] for t in validated]

    latency = (time.time() - start) * 1000
    return terms, latency


def strategy_f_span_verify(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy F: Structured output + span verification."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])

    start = time.time()
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)

    raw_terms = parse_json_response(response)
    validated = validate_terms_with_source(raw_terms, chunk_text)
    terms = [t["term"] for t in validated]

    latency = (time.time() - start) * 1000
    return terms, latency


def strategy_g_self_consistency(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy G: Self-consistency voting with N=10 samples."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])

    start = time.time()
    all_term_sets: list[set[str]] = []

    for i in range(SC_NUM_SAMPLES):
        response = call_llm(
            prompt, model=model, temperature=SC_TEMPERATURE, max_tokens=2000
        )
        raw_terms = parse_json_response(response)
        validated = validate_terms_with_source(raw_terms, chunk_text)
        terms = {t["term"].lower() for t in validated}
        all_term_sets.append(terms)
        time.sleep(0.3)  # Rate limiting

    # Count occurrences and apply threshold
    term_counts: Counter = Counter()
    for term_set in all_term_sets:
        term_counts.update(term_set)

    min_count = int(SC_NUM_SAMPLES * SC_AGREEMENT_THRESHOLD)
    high_confidence = [
        term for term, count in term_counts.items() if count >= min_count
    ]

    latency = (time.time() - start) * 1000
    return high_confidence, latency


def strategy_h_multi_pass(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy H: Multi-pass extraction with 'what did I miss?'"""
    start = time.time()
    all_terms: set[str] = set()

    # Pass 1: Initial exhaustive extraction
    prompt1 = PROMPT_MULTI_PASS_1.format(chunk_text=chunk_text[:3000])
    response1 = call_llm(prompt1, model=model, temperature=0.3, max_tokens=2000)
    raw1 = parse_json_response(response1)
    validated1 = validate_terms_with_source(raw1, chunk_text)
    all_terms.update(t["term"] for t in validated1)

    # Pass 2: "What did I miss?"
    prompt2 = PROMPT_MULTI_PASS_2.format(
        extracted_terms=list(all_terms), chunk_text=chunk_text[:3000]
    )
    response2 = call_llm(prompt2, model=model, temperature=0.5, max_tokens=2000)
    raw2 = parse_json_response(response2)
    validated2 = validate_terms_with_source(raw2, chunk_text)
    all_terms.update(t["term"] for t in validated2)

    # Pass 3: Category sweep
    prompt3 = PROMPT_MULTI_PASS_3.format(
        extracted_terms=list(all_terms), chunk_text=chunk_text[:3000]
    )
    response3 = call_llm(prompt3, model=model, temperature=0.5, max_tokens=2000)
    raw3 = parse_json_response(response3)
    validated3 = validate_terms_with_source(raw3, chunk_text)
    all_terms.update(t["term"] for t in validated3)

    latency = (time.time() - start) * 1000
    return list(all_terms), latency


def strategy_i_combined(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy I: Combined pipeline (multi-pass + self-consistency + verification)."""
    start = time.time()

    # Step 1: Multi-pass for high recall
    multi_pass_terms, _ = strategy_h_multi_pass(chunk_text, model)

    # Step 2: Self-consistency voting (reduced N=5)
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])
    all_term_sets: list[set[str]] = []

    for i in range(5):  # Reduced from 10
        response = call_llm(
            prompt, model=model, temperature=SC_TEMPERATURE, max_tokens=2000
        )
        raw_terms = parse_json_response(response)
        validated = validate_terms_with_source(raw_terms, chunk_text)
        terms = {t["term"].lower() for t in validated}
        all_term_sets.append(terms)
        time.sleep(0.3)

    term_counts: Counter = Counter()
    for term_set in all_term_sets:
        term_counts.update(term_set)

    min_count = 3  # 3 out of 5 = 60%
    sc_terms = {term for term, count in term_counts.items() if count >= min_count}

    # Step 3: Combine and verify
    combined_terms = set(t.lower() for t in multi_pass_terms) | sc_terms

    # Final verification
    verified_terms = []
    for term in combined_terms:
        if term.lower() in chunk_text.lower():
            verified_terms.append(term)

    latency = (time.time() - start) * 1000
    return verified_terms, latency


# ============================================================================
# Main Experiment
# ============================================================================

STRATEGIES = {
    "E": ("Structured Basic", strategy_e_structured),
    "F": ("Span Verification", strategy_f_span_verify),
    "G": ("Self-Consistency", strategy_g_self_consistency),
    "H": ("Multi-Pass", strategy_h_multi_pass),
    "I": ("Combined", strategy_i_combined),
}


def run_experiment():
    """Run all experiments."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Load ground truth from POC-1
    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: Ground truth not found at {GROUND_TRUTH_PATH}", flush=True)
        return False

    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    chunks = ground_truth["chunks"]
    print(f"\nLoaded {len(chunks)} chunks from POC-1 ground truth", flush=True)

    # Verify LLM provider works
    print("\nVerifying LLM provider...", flush=True)
    try:
        test_response = call_llm(
            "Say 'OK' if you can read this.", model="claude-haiku", max_tokens=10
        )
        if test_response:
            print(f"  LLM Provider: OK", flush=True)
        else:
            print(f"  LLM Provider: FAILED (empty response)", flush=True)
            return False
    except Exception as e:
        print(f"  LLM Provider: FAILED - {e}", flush=True)
        return False

    # Run experiments
    results = []
    total_conditions = len(chunks) * len(MODELS) * len(STRATEGIES) * TRIALS
    completed = 0

    print(f"\nRunning {total_conditions} extraction conditions...", flush=True)
    start_time = datetime.now(timezone.utc)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        gt_terms = chunk["terms"]

        print(f"\n  Chunk: {chunk_id}", flush=True)

        for model in MODELS:
            for strategy_id, (strategy_name, strategy_fn) in STRATEGIES.items():
                for trial in range(1, TRIALS + 1):
                    completed += 1
                    print(
                        f"    [{completed}/{total_conditions}] {model} {strategy_id} T{trial}...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        terms, latency = strategy_fn(chunk_text, model)
                        metrics = calculate_metrics(terms, gt_terms)

                        result = {
                            "chunk_id": chunk_id,
                            "model": model,
                            "strategy": strategy_id,
                            "strategy_name": strategy_name,
                            "trial": trial,
                            "extracted_terms": terms,
                            "metrics": {
                                "precision": metrics.precision,
                                "recall": metrics.recall,
                                "hallucination_rate": metrics.hallucination_rate,
                                "exact_matches": metrics.exact_matches,
                                "partial_matches": metrics.partial_matches,
                                "fuzzy_matches": metrics.fuzzy_matches,
                                "false_positives": metrics.false_positives,
                                "false_negatives": metrics.false_negatives,
                            },
                            "latency_ms": latency,
                            "num_extracted": len(terms),
                            "num_ground_truth": len(gt_terms),
                        }
                        results.append(result)

                        print(
                            f"P={metrics.precision:.1%} R={metrics.recall:.1%} H={metrics.hallucination_rate:.1%}",
                            flush=True,
                        )

                    except Exception as e:
                        print(f"ERROR: {e}", flush=True)
                        results.append(
                            {
                                "chunk_id": chunk_id,
                                "model": model,
                                "strategy": strategy_id,
                                "strategy_name": strategy_name,
                                "trial": trial,
                                "error": str(e),
                            }
                        )

                    # Rate limiting
                    time.sleep(0.5)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # Save raw results
    raw_results = {
        "experiment_id": "poc-1b-main",
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": duration,
        "total_conditions": total_conditions,
        "completed_conditions": len([r for r in results if "error" not in r]),
        "failed_conditions": len([r for r in results if "error" in r]),
        "configuration": {
            "models": MODELS,
            "strategies": list(STRATEGIES.keys()),
            "trials": TRIALS,
            "sc_num_samples": SC_NUM_SAMPLES,
            "sc_agreement_threshold": SC_AGREEMENT_THRESHOLD,
            "sc_temperature": SC_TEMPERATURE,
        },
        "results": results,
    }

    results_path = ARTIFACTS_DIR / "phase-3-raw-results.json"
    with open(results_path, "w") as f:
        json.dump(raw_results, f, indent=2)

    print(f"\n\nExperiment complete!", flush=True)
    print(f"  Duration: {duration / 60:.1f} minutes", flush=True)
    print(f"  Results saved: {results_path}", flush=True)

    # Quick summary
    print("\n" + "=" * 50, flush=True)
    print("QUICK SUMMARY BY STRATEGY", flush=True)
    print("=" * 50, flush=True)

    for strategy_id in STRATEGIES:
        strategy_results = [
            r for r in results if r.get("strategy") == strategy_id and "metrics" in r
        ]
        if strategy_results:
            avg_p = sum(r["metrics"]["precision"] for r in strategy_results) / len(
                strategy_results
            )
            avg_r = sum(r["metrics"]["recall"] for r in strategy_results) / len(
                strategy_results
            )
            avg_h = sum(
                r["metrics"]["hallucination_rate"] for r in strategy_results
            ) / len(strategy_results)
            print(
                f"  {strategy_id} ({STRATEGIES[strategy_id][0]}): P={avg_p:.1%} R={avg_r:.1%} H={avg_h:.1%}",
                flush=True,
            )

    return True


def main():
    success = run_experiment()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
