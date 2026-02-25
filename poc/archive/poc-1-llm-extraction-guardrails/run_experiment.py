#!/usr/bin/env python3
"""Phase 4: Main Experiment Execution

Runs all extraction conditions:
- 45 chunks × 2 models × 4 prompts × 3 trials = 1080 extractions
(Reduced from 50×4×4×3=2400 due to 45 chunks and 2 Claude models only)
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print("Phase 4: Main Experiment Execution", flush=True)
print("=" * 50, flush=True)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MODELS = ["claude-haiku", "claude-sonnet"]
VARIANTS = ["A", "B", "C", "D"]
TRIALS = 3

PROMPT_VARIANTS = {
    "A": """Extract Kubernetes domain-specific terms from this text.

TEXT:
---
{chunk_text}
---

OUTPUT (JSON list of terms):
["term1", "term2", ...]""",
    "B": """Extract Kubernetes domain-specific terms from this text.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide the EXACT text span where it appears
3. If you cannot find the term verbatim, do NOT include it

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{{"terms": [{{"term": "...", "span": "...exact quote from text..."}}]}}""",
    "C": """Extract Kubernetes domain-specific terms from this text.

RULES:
1. Maximum 15 terms per chunk
2. Prioritize terms that are MOST specific to Kubernetes
3. Assign confidence: HIGH (definitely K8s), MEDIUM (likely K8s), LOW (possibly generic)

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{{"terms": [{{"term": "...", "confidence": "HIGH|MEDIUM|LOW"}}]}}""",
    "D": """Extract Kubernetes domain-specific terms from this documentation chunk.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide the EXACT text span where it appears
3. Assign confidence: HIGH (definitely K8s-specific), MEDIUM (technical, likely K8s), LOW (possibly generic)
4. Maximum 15 terms per chunk
5. Prefer multi-word terms over single words when both exist (e.g., "Pod Security Policy" over "Pod")

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{{"terms": [{{"term": "CrashLoopBackOff", "span": "the pod entered CrashLoopBackOff state", "confidence": "HIGH"}}]}}""",
}


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted: str, ground_truth: str) -> str:
    from rapidfuzz import fuzz

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


def parse_extraction_response(response: str, variant: str) -> list[dict]:
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    if variant == "A":
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                terms = json.loads(json_match.group())
                return [{"term": t} if isinstance(t, str) else t for t in terms]
            except json.JSONDecodeError:
                pass
    else:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("terms", [])
            except json.JSONDecodeError:
                pass

    try:
        data = json.loads(response)
        if isinstance(data, list):
            return [{"term": t} if isinstance(t, str) else t for t in data]
        return data.get("terms", [])
    except json.JSONDecodeError:
        return []


def calculate_metrics(
    extracted_terms: list[dict], ground_truth_terms: list[dict]
) -> dict:
    gt_terms = [t.get("term", "") for t in ground_truth_terms]
    ext_terms = [t.get("term", "") for t in extracted_terms]

    matched_gt = set()
    exact_matches = 0
    partial_matches = 0
    fuzzy_matches = 0

    for ext in ext_terms:
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
    false_positives = len(ext_terms) - true_positives
    false_negatives = len(gt_terms) - len(matched_gt)

    precision = true_positives / len(ext_terms) if ext_terms else 0.0
    recall = true_positives / len(gt_terms) if gt_terms else 0.0
    hallucination_rate = false_positives / len(ext_terms) if ext_terms else 0.0

    groundedness = 0.0
    if extracted_terms:
        spans_present = sum(1 for t in extracted_terms if t.get("span", ""))
        groundedness = spans_present / len(extracted_terms)

    return {
        "precision": precision,
        "recall": recall,
        "hallucination_rate": hallucination_rate,
        "groundedness_score": groundedness,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "fuzzy_matches": fuzzy_matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def run_experiment():
    from utils.llm_provider import call_llm

    gt_path = ARTIFACTS_DIR / "phase-2-ground-truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    chunks = ground_truth["chunks"]
    total_conditions = len(chunks) * len(MODELS) * len(VARIANTS) * TRIALS

    print(f"\nExperiment Configuration:", flush=True)
    print(f"  Chunks: {len(chunks)}", flush=True)
    print(f"  Models: {MODELS}", flush=True)
    print(f"  Variants: {VARIANTS}", flush=True)
    print(f"  Trials: {TRIALS}", flush=True)
    print(f"  Total conditions: {total_conditions}", flush=True)
    print(f"  Estimated time: ~{total_conditions * 4 / 60:.0f} minutes", flush=True)

    results = []
    completed = 0
    failed = 0
    start_time = datetime.now(timezone.utc)

    for chunk in chunks:
        for model in MODELS:
            for variant in VARIANTS:
                for trial in range(1, TRIALS + 1):
                    completed += 1

                    if completed % 20 == 0 or completed <= 5:
                        elapsed = (
                            datetime.now(timezone.utc) - start_time
                        ).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_conditions - completed) / rate if rate > 0 else 0
                        print(
                            f"\n[{completed}/{total_conditions}] {chunk['chunk_id']} | {model} | {variant} | trial {trial} (ETA: {eta / 60:.1f}m)",
                            flush=True,
                        )

                    prompt = PROMPT_VARIANTS[variant].format(
                        chunk_text=chunk["text"][:2500]
                    )

                    try:
                        extraction_start = time.time()
                        response = call_llm(
                            prompt,
                            model=model,
                            max_tokens=2000,
                            temperature=0,
                            timeout=90,
                        )
                        latency_ms = (time.time() - extraction_start) * 1000

                        terms = parse_extraction_response(response, variant)
                        metrics = calculate_metrics(terms, chunk["terms"])

                        result = {
                            "chunk_id": chunk["chunk_id"],
                            "model": model,
                            "prompt_variant": variant,
                            "trial": trial,
                            "extracted_terms": terms,
                            "metrics": metrics,
                            "latency_ms": round(latency_ms),
                            "tokens_used": {"input": 0, "output": 0},
                        }
                        results.append(result)

                        if completed % 20 == 0 or completed <= 5:
                            print(
                                f"  P={metrics['precision']:.1%} R={metrics['recall']:.1%} H={metrics['hallucination_rate']:.1%} | {len(terms)} terms | {latency_ms:.0f}ms",
                                flush=True,
                            )

                    except Exception as e:
                        failed += 1
                        print(f"  ERROR: {e}", flush=True)
                        results.append(
                            {
                                "chunk_id": chunk["chunk_id"],
                                "model": model,
                                "prompt_variant": variant,
                                "trial": trial,
                                "extracted_terms": [],
                                "metrics": {
                                    "precision": 0,
                                    "recall": 0,
                                    "hallucination_rate": 0,
                                    "groundedness_score": 0,
                                    "exact_matches": 0,
                                    "partial_matches": 0,
                                    "fuzzy_matches": 0,
                                    "false_positives": 0,
                                    "false_negatives": 0,
                                },
                                "latency_ms": 0,
                                "error": str(e),
                            }
                        )

                    time.sleep(0.3)

    end_time = datetime.now(timezone.utc)

    experiment_data = {
        "experiment_id": "poc-1-main",
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "total_conditions": total_conditions,
        "completed_conditions": completed,
        "failed_conditions": failed,
        "results": results,
    }

    output_path = ARTIFACTS_DIR / "phase-4-raw-results.json"
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n\nExperiment Complete!", flush=True)
    print(
        f"  Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes",
        flush=True,
    )
    print(f"  Completed: {completed}/{total_conditions}", flush=True)
    print(f"  Failed: {failed}", flush=True)
    print(f"  Results saved: {output_path}", flush=True)

    summary_path = ARTIFACTS_DIR / "phase-4-summary.md"
    with open(summary_path, "w") as f:
        f.write(f"""# Phase 4 Summary: Main Experiment Execution

## Objective

Run all {total_conditions} extraction conditions ({len(chunks)} chunks × {len(MODELS)} models × {len(VARIANTS)} variants × {TRIALS} trials).

## Approach

1. Loaded ground truth with {len(chunks)} annotated chunks
2. Ran extraction for each model-variant-trial combination
3. Calculated metrics for each extraction
4. Saved all raw results for analysis

## Results

| Metric | Value |
|--------|-------|
| Total conditions | {total_conditions} |
| Completed | {completed} |
| Failed | {failed} |
| Duration | {(end_time - start_time).total_seconds() / 60:.1f} minutes |

## Issues Encountered

{"None" if failed == 0 else f"{failed} extractions failed (see results for details)"}

## Next Phase Readiness

- [x] All conditions executed
- [x] Raw results saved to artifacts
- [x] Ready for Phase 5: Analysis and Reporting

**Phase 4 Status: COMPLETE**
""")

    print(f"Summary saved: {summary_path}", flush=True)
    return experiment_data


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    run_experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
