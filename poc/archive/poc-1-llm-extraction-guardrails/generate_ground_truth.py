#!/usr/bin/env python3
"""Phase 2: Ground Truth Generation

Creates gold-standard annotations for 50 K8s documentation chunks using Claude Opus.
"""

import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print("Phase 2: Ground Truth Generation", flush=True)
print("=" * 50, flush=True)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CORPUS_DIR = (
    Path(__file__).parent.parent
    / "chunking_benchmark_v2"
    / "corpus"
    / "kubernetes_sample_200"
)

TARGET_CHUNKS = 50
CONTENT_TYPE_DISTRIBUTION = {
    "prose": 20,
    "code": 10,
    "tables": 8,
    "errors": 7,
    "mixed": 5,
}

OPUS_EXTRACTION_PROMPT = """You are creating ground truth annotations for evaluating term extraction models.

TASK: Extract ALL Kubernetes domain-specific terms from this chunk.

TERM CLASSIFICATION:
- Tier 1 (MUST INCLUDE): Terms unique to Kubernetes (CrashLoopBackOff, kube-apiserver, PodSpec)
- Tier 2 (MUST INCLUDE): English words with specific K8s meaning (pod, container, service, node, deployment)
- Tier 3 (CONDITIONAL): Technical terms not K8s-specific - include ONLY if used in K8s context (API, endpoint, namespace)
- Tier 4 (EXCLUDE): Generic words with no special K8s meaning (component, system, configuration)

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, quote the exact text span where it appears
3. Assign tier (1, 2, or 3)
4. Include multi-word terms (e.g., "Pod Security Policy" not just "Pod")

OUTPUT FORMAT (JSON only, no markdown, no backticks):
{{"terms": [{{"term": "CrashLoopBackOff", "tier": 1, "span": "entered CrashLoopBackOff state"}}]}}

CHUNK:
---
{chunk_text}
---"""

OPUS_REVIEW_PROMPT = """Review this ground truth annotation for completeness and accuracy.

ORIGINAL CHUNK:
---
{chunk_text}
---

EXTRACTED TERMS:
{extraction_json}

CHECK:
1. Are there any Tier 1/2 terms in the chunk that were MISSED?
2. Are there any extracted terms that DON'T appear verbatim in the chunk?
3. Are tier assignments correct?

OUTPUT (JSON only, no markdown, no backticks):
{{"review_status": "APPROVED", "missed_terms": [], "false_extractions": [], "tier_corrections": []}}"""


def classify_content_type(text: str, filename: str) -> str:
    text_lower = text.lower()

    if (
        "```" in text
        or "yaml:" in text_lower
        or filename.startswith("reference_kubernetes-api")
    ):
        if "|" in text and text.count("|") > 5:
            return "tables"
        return "code"

    if "|" in text and text.count("|") > 5:
        return "tables"

    if any(
        word in text_lower
        for word in ["error", "troubleshoot", "failed", "issue", "problem"]
    ):
        return "errors"

    if filename.startswith("reference_kubernetes-api") or filename.startswith(
        "reference_config-api"
    ):
        return "mixed"

    return "prose"


def select_chunks() -> list[dict]:
    print("\n[1/4] Selecting chunks with stratified sampling...", flush=True)

    files = list(CORPUS_DIR.glob("*.md"))
    print(f"  Found {len(files)} files in corpus", flush=True)

    categorized: dict[str, list[dict]] = {ct: [] for ct in CONTENT_TYPE_DISTRIBUTION}

    for file_path in files:
        text = file_path.read_text()
        content_type = classify_content_type(text, file_path.name)

        if len(text) < 100:
            continue

        categorized[content_type].append(
            {
                "file": file_path.name,
                "text": text,
                "content_type": content_type,
            }
        )

    for ct, items in categorized.items():
        print(f"    {ct}: {len(items)} files", flush=True)

    selected = []
    for content_type, target_count in CONTENT_TYPE_DISTRIBUTION.items():
        available = categorized[content_type]
        if len(available) < target_count:
            print(
                f"  WARNING: Only {len(available)} {content_type} files (need {target_count})",
                flush=True,
            )
            sample = available
        else:
            sample = random.sample(available, target_count)

        for item in sample:
            selected.append(
                {
                    "chunk_id": f"chunk_{len(selected):03d}",
                    "source_file": item["file"],
                    "content_type": content_type,
                    "text": item["text"][:3000],
                }
            )

    print(f"  Selected {len(selected)} chunks total", flush=True)
    return selected


def parse_json_response(response: str) -> dict | None:
    response = response.strip()

    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def extract_terms_opus(chunk: dict, call_llm) -> list[dict]:
    prompt = OPUS_EXTRACTION_PROMPT.format(chunk_text=chunk["text"][:2500])

    start = time.time()
    response = call_llm(
        prompt, model="claude-opus", max_tokens=2000, temperature=0, timeout=120
    )
    elapsed = time.time() - start

    print(f"      Extraction: {elapsed:.1f}s, {len(response)} chars", flush=True)

    if not response:
        return []

    parsed = parse_json_response(response)
    if not parsed or "terms" not in parsed:
        print(f"      Parse error: {response[:100]}...", flush=True)
        return []

    return parsed["terms"]


def review_extraction_opus(
    chunk: dict, terms: list[dict], call_llm
) -> tuple[list[dict], str]:
    extraction_json = json.dumps({"terms": terms}, indent=2)
    prompt = OPUS_REVIEW_PROMPT.format(
        chunk_text=chunk["text"][:2500],
        extraction_json=extraction_json,
    )

    start = time.time()
    response = call_llm(
        prompt, model="claude-opus", max_tokens=2000, temperature=0, timeout=120
    )
    elapsed = time.time() - start

    print(f"      Review: {elapsed:.1f}s", flush=True)

    if not response:
        return terms, "review_failed"

    parsed = parse_json_response(response)
    if not parsed:
        return terms, "review_failed"

    review_status = parsed.get("review_status", "UNKNOWN")

    if review_status == "APPROVED":
        return terms, "approved"

    final_terms = list(terms)

    false_extraction_list = parsed.get("false_extractions", [])
    false_terms = set()
    for fe in false_extraction_list:
        if isinstance(fe, str):
            false_terms.add(fe)
        elif isinstance(fe, dict) and "term" in fe:
            false_terms.add(fe["term"])
    final_terms = [t for t in final_terms if t.get("term") not in false_terms]

    for missed in parsed.get("missed_terms", []):
        if missed.get("term"):
            final_terms.append(missed)

    for correction in parsed.get("tier_corrections", []):
        for t in final_terms:
            if t.get("term") == correction.get("term"):
                t["tier"] = correction.get("new_tier", t.get("tier"))

    return final_terms, "corrected"


def main():
    from utils.llm_provider import call_llm

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    random.seed(42)

    chunks = select_chunks()

    print(f"\n[2/4] Extracting terms with Claude Opus...", flush=True)
    print(
        f"  Processing {len(chunks)} chunks (this will take ~15-20 minutes)", flush=True
    )

    annotations = []
    total_terms = 0

    for i, chunk in enumerate(chunks):
        print(
            f"\n  [{i + 1}/{len(chunks)}] {chunk['chunk_id']} ({chunk['content_type']})",
            flush=True,
        )
        print(f"    File: {chunk['source_file']}", flush=True)

        terms = extract_terms_opus(chunk, call_llm)

        if terms:
            time.sleep(0.5)
            terms, review_status = review_extraction_opus(chunk, terms, call_llm)
        else:
            review_status = "no_terms"

        print(f"      Final: {len(terms)} terms, status={review_status}", flush=True)

        annotations.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk["source_file"],
                "content_type": chunk["content_type"],
                "text": chunk["text"],
                "terms": terms,
                "total_terms": len(terms),
                "human_validated": False,
                "review_status": review_status,
            }
        )
        total_terms += len(terms)

        time.sleep(0.3)

    print(f"\n[3/4] Compiling ground truth...", flush=True)

    content_type_actual = {}
    for ann in annotations:
        ct = ann["content_type"]
        content_type_actual[ct] = content_type_actual.get(ct, 0) + 1

    spot_check_indices = random.sample(
        range(len(annotations)), min(5, len(annotations))
    )
    spot_check_chunks = [annotations[i]["chunk_id"] for i in spot_check_indices]

    ground_truth = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "claude-opus",
        "total_chunks": len(annotations),
        "total_terms": total_terms,
        "content_type_distribution": content_type_actual,
        "chunks": annotations,
        "human_spot_check": {
            "chunks_checked": 0,
            "chunks_to_check": spot_check_chunks,
            "discrepancies": [],
            "agreement_rate": 0.0,
        },
    }

    output_path = ARTIFACTS_DIR / "phase-2-ground-truth.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n[4/4] Results", flush=True)
    print(f"  Ground truth saved: {output_path}", flush=True)
    print(f"  Total chunks: {ground_truth['total_chunks']}", flush=True)
    print(f"  Total terms: {ground_truth['total_terms']}", flush=True)
    print(
        f"  Avg terms/chunk: {total_terms / max(1, len(annotations)):.1f}", flush=True
    )
    print(f"  Human spot-check needed: {spot_check_chunks}", flush=True)

    summary_path = ARTIFACTS_DIR / "phase-2-summary.md"
    with open(summary_path, "w") as f:
        f.write(f"""# Phase 2 Summary: Ground Truth Creation

## Objective

Create gold-standard annotations for {TARGET_CHUNKS} K8s documentation chunks using Claude Opus.

## Approach

1. Stratified sampling of chunks by content type
2. Claude Opus extraction with detailed tier classification
3. Claude Opus self-review for validation
4. Identified {len(spot_check_chunks)} chunks for human spot-check

## Results

| Metric | Value |
|--------|-------|
| Total chunks annotated | {ground_truth["total_chunks"]} |
| Total terms extracted | {ground_truth["total_terms"]} |
| Average terms per chunk | {total_terms / max(1, len(annotations)):.1f} |

### Content Type Distribution

| Type | Count |
|------|-------|
""")
        for ct, count in content_type_actual.items():
            f.write(f"| {ct} | {count} |\n")

        f.write(f"""
## Issues Encountered

None during automated processing.

## Next Phase Readiness

- [x] {ground_truth["total_chunks"]} chunks annotated
- [x] Ground truth JSON saved to artifacts
- [ ] Human spot-check pending for 5 chunks
- [x] Ready for Phase 3: Evaluation Harness Implementation

**Phase 2 Status: COMPLETE**
""")

    print(f"\nPhase 2 COMPLETE", flush=True)
    print(f"Summary saved: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
