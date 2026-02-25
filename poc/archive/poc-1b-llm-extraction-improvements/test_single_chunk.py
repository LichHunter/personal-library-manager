#!/usr/bin/env python3
"""Test POC-1b strategies on a single chunk."""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator
from rapidfuzz import fuzz

# Add POC-1 utils to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: Single Chunk Test", flush=True)
print("=" * 50, flush=True)

# Paths
POC1_DIR = Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"
GROUND_TRUTH_PATH = POC1_DIR / "artifacts" / "phase-2-ground-truth.json"

# Config
SC_NUM_SAMPLES = 5  # Reduced for testing
SC_AGREEMENT_THRESHOLD = 0.6
SC_TEMPERATURE = 0.8


# ============================================================================
# Pydantic Models
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
# Prompts
# ============================================================================

PROMPT_STRUCTURED = """Extract Kubernetes domain-specific terms from this documentation chunk.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide an EXACT text span (10-50 words) where it appears
3. Assign confidence: HIGH (K8s-specific), MEDIUM (technical), LOW (generic)
4. Maximum 15 terms per chunk

TEXT:
---
{chunk_text}
---

OUTPUT FORMAT (JSON only, no markdown):
{{"terms": [{{"term": "example", "span": "exact quote containing example", "confidence": "HIGH"}}]}}"""

PROMPT_MULTI_PASS_1 = """Extract ALL Kubernetes terms from this text. Be EXHAUSTIVE.

TEXT:
---
{chunk_text}
---

OUTPUT (JSON): {{"terms": [{{"term": "...", "span": "...", "confidence": "HIGH|MEDIUM|LOW"}}]}}"""

PROMPT_MULTI_PASS_2 = """Previously extracted: {extracted_terms}

What Kubernetes terms did I MISS? Focus on abbreviations, multi-word terms, API objects.

TEXT:
---
{chunk_text}
---

Additional terms only (JSON): {{"terms": [...]}}"""


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
    }


# ============================================================================
# Strategies
# ============================================================================


def strategy_e(chunk_text: str, model: str) -> list[str]:
    """E: Basic structured output."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)
    raw = parse_json_response(response)
    # No source validation for baseline
    return [t.get("term", "") for t in raw if t.get("term")]


def strategy_f(chunk_text: str, model: str) -> list[str]:
    """F: Structured + span verification."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=2000)
    raw = parse_json_response(response)
    validated = validate_terms_with_source(raw, chunk_text)
    return [t["term"] for t in validated]


def strategy_g(chunk_text: str, model: str) -> list[str]:
    """G: Self-consistency voting."""
    prompt = PROMPT_STRUCTURED.format(chunk_text=chunk_text[:3000])
    all_terms: list[set[str]] = []

    for i in range(SC_NUM_SAMPLES):
        print(f"      SC sample {i + 1}/{SC_NUM_SAMPLES}...", end=" ", flush=True)
        response = call_llm(
            prompt, model=model, temperature=SC_TEMPERATURE, max_tokens=2000
        )
        raw = parse_json_response(response)
        validated = validate_terms_with_source(raw, chunk_text)
        terms = {t["term"].lower() for t in validated}
        all_terms.append(terms)
        print(f"{len(terms)} terms", flush=True)
        time.sleep(0.3)

    counts = Counter()
    for ts in all_terms:
        counts.update(ts)

    min_count = int(SC_NUM_SAMPLES * SC_AGREEMENT_THRESHOLD)
    return [t for t, c in counts.items() if c >= min_count]


def strategy_h(chunk_text: str, model: str) -> list[str]:
    """H: Multi-pass extraction."""
    all_terms: set[str] = set()

    # Pass 1
    print("      Pass 1...", end=" ", flush=True)
    p1 = PROMPT_MULTI_PASS_1.format(chunk_text=chunk_text[:3000])
    r1 = call_llm(p1, model=model, temperature=0.3, max_tokens=2000)
    raw1 = parse_json_response(r1)
    v1 = validate_terms_with_source(raw1, chunk_text)
    all_terms.update(t["term"] for t in v1)
    print(f"{len(v1)} terms", flush=True)

    # Pass 2
    print("      Pass 2...", end=" ", flush=True)
    p2 = PROMPT_MULTI_PASS_2.format(
        extracted_terms=list(all_terms), chunk_text=chunk_text[:3000]
    )
    r2 = call_llm(p2, model=model, temperature=0.5, max_tokens=2000)
    raw2 = parse_json_response(r2)
    v2 = validate_terms_with_source(raw2, chunk_text)
    all_terms.update(t["term"] for t in v2)
    print(f"+{len(v2)} terms", flush=True)

    return list(all_terms)


# ============================================================================
# Main
# ============================================================================


def main():
    # Load first chunk
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    chunk = gt["chunks"][0]
    chunk_id = chunk["chunk_id"]
    chunk_text = chunk["text"]
    gt_terms = chunk["terms"]

    print(f"\nChunk: {chunk_id}", flush=True)
    print(f"Ground truth: {len(gt_terms)} terms", flush=True)
    print(f"Text length: {len(chunk_text)} chars", flush=True)
    print(f"\nGT terms: {[t['term'] for t in gt_terms]}", flush=True)

    # Test each strategy
    strategies = {
        "E": ("Basic Structured", strategy_e),
        "F": ("Span Verify", strategy_f),
        "G": ("Self-Consistency", strategy_g),
        "H": ("Multi-Pass", strategy_h),
    }

    model = "claude-haiku"
    print(f"\n{'=' * 50}", flush=True)
    print(f"Testing with {model}", flush=True)
    print(f"{'=' * 50}", flush=True)

    results = {}

    for sid, (name, fn) in strategies.items():
        print(f"\n  Strategy {sid} ({name}):", flush=True)
        try:
            terms = fn(chunk_text, model)
            metrics = calculate_metrics(terms, gt_terms)
            results[sid] = {"terms": terms, "metrics": metrics}

            print(f"    Extracted: {len(terms)} terms", flush=True)
            print(f"    Terms: {terms}", flush=True)
            print(f"    Precision: {metrics['precision']:.1%}", flush=True)
            print(f"    Recall: {metrics['recall']:.1%}", flush=True)
            print(f"    Hallucination: {metrics['hallucination']:.1%}", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            results[sid] = {"error": str(e)}

    # Summary
    print(f"\n{'=' * 50}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"{'Strategy':<20} {'P':>8} {'R':>8} {'H':>8} {'#Ext':>6}", flush=True)
    print("-" * 50, flush=True)

    for sid, (name, _) in strategies.items():
        if "metrics" in results.get(sid, {}):
            m = results[sid]["metrics"]
            print(f"{sid} ({name})"[:20].ljust(20), end="", flush=True)
            print(
                f" {m['precision']:>7.1%} {m['recall']:>7.1%} {m['hallucination']:>7.1%} {m['extracted']:>6}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
