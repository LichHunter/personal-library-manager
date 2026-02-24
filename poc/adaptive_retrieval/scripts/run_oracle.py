#!/usr/bin/env python3
"""
Oracle Performance & Precision@10 Script for Adaptive Retrieval POC.

Computes:
1. Oracle performance from existing per-query results across all approaches
2. Document-level retrieval + LLM judge for missing granularity
3. Precision@10 for baseline

Usage:
    # Full run (oracle + document-level + precision)
    python run_oracle.py

    # Skip document-level LLM calls (oracle from existing data only)
    python run_oracle.py --skip-documents

Outputs:
    - results/oracle_performance.md
    - results/oracle_performance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add src to path for PLM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from plm.search.retriever import HybridRetriever

try:
    from plm.shared.llm.base import call_llm
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    call_llm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

POC_ROOT = Path(__file__).parent.parent
TEST_QUERIES_PATH = POC_ROOT / "benchmarks" / "datasets" / "test_queries.json"
RESULTS_DIR = POC_ROOT / "results"

PLM_DB_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db"
PLM_BM25_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data"

JUDGE_MODEL = "haiku"
JUDGE_MAX_RETRIES = 3
SUCCESS_THRESHOLD = 6

# Approach files and their granularity labels
APPROACH_FILES = {
    "baseline": ("00_baseline.json", "chunk"),
    "reranking": ("01_reranking.json", "reranked_chunk"),
    "auto_merging": ("02_auto_merging.json", "merged"),
    "adaptive_classifier": ("03_adaptive_classifier.json", "adaptive"),
    "parent_child": ("15_parent_child.json", "heading"),
}

# Same judge prompt as run_baseline.py
JUDGE_PROMPT = """You are evaluating whether retrieved document chunks can answer a user's question.

USER QUESTION: {question}

EXPECTED ANSWER (for reference): {expected_answer}

RETRIEVED CHUNKS:
{chunks_text}

SCORING GUIDE:
- 10: PERFECT - Chunks contain the complete answer with all necessary details
- 8-9: EXCELLENT - Chunks contain the core answer, minor details may be missing
- 6-7: GOOD - Chunks contain most relevant information, can answer the question
- 4-5: PARTIAL - Some relevant info but missing key details, incomplete answer
- 2-3: POOR - Tangentially related but doesn't address the question directly
- 1: IRRELEVANT - No useful information for answering the question

Rate how well the retrieved chunks can answer the user's question.

Respond with ONLY a JSON object:
{{"grade": <integer 1-10>, "reasoning": "<brief explanation of what key facts are present or missing>"}}"""


def grade_to_category(grade: int) -> str:
    if grade >= 9:
        return "Correct"
    elif grade >= 6:
        return "Partially Correct"
    elif grade >= 3:
        return "Incorrect"
    else:
        return "Cannot Answer"


# =============================================================================
# PART 1: LOAD EXISTING PER-QUERY DATA
# =============================================================================

def load_approach_results() -> dict[str, dict[str, int]]:
    """Load per-query judge_grade from all approach JSON files.

    Returns:
        {approach_name: {query_id: grade}}
    """
    all_grades: dict[str, dict[str, int]] = {}

    for approach_name, (filename, _granularity) in APPROACH_FILES.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            logger.warning(f"Missing results file: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        grades: dict[str, int] = {}
        for result in data.get("per_query_results", []):
            qid = result.get("query_id", "")
            grade = result.get("judge_grade")
            if qid and grade is not None:
                grades[qid] = int(grade)

        all_grades[approach_name] = grades
        logger.info(f"Loaded {len(grades)} grades from {approach_name}")

    return all_grades


# =============================================================================
# PART 2: DOCUMENT-LEVEL RETRIEVAL + JUDGE
# =============================================================================

def judge_document_context(
    query: str,
    expected_answer: str,
    doc_chunks: list[dict],
) -> tuple[Optional[int], str, float]:
    """Judge document-level context quality."""
    if not HAS_LLM or call_llm is None:
        return (None, "LLM not available", 0.0)

    if not doc_chunks:
        return (1, "No document content", 0.0)

    # Format chunks (limit to avoid token overflow)
    lines = []
    total_chars = 0
    for i, chunk in enumerate(doc_chunks, 1):
        content = chunk.get("content", "")[:500]
        heading = chunk.get("heading", "")
        lines.append(f"[Section {i}] {heading}\n{content}")
        total_chars += len(content)
        if total_chars > 4000:
            break

    chunks_text = "\n\n".join(lines)

    prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        chunks_text=chunks_text,
    )

    start_time = time.perf_counter()

    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = call_llm(
                prompt=prompt,
                model=JUDGE_MODEL,
                max_tokens=200,
                temperature=0,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            if not response:
                continue

            response_text = response.strip()
            try:
                if response_text.startswith("{"):
                    data = json.loads(response_text)
                else:
                    match = re.search(r'\{[^}]+\}', response_text)
                    if match:
                        data = json.loads(match.group())
                    else:
                        raise ValueError("No JSON found")

                grade = max(1, min(10, int(data.get("grade", 5))))
                reasoning = data.get("reasoning", "")
                return (grade, reasoning, latency_ms)

            except (json.JSONDecodeError, ValueError):
                match = re.search(r'grade["\s:]+(\d+)', response_text, re.IGNORECASE)
                if match:
                    grade = max(1, min(10, int(match.group(1))))
                    return (grade, f"Extracted: {response_text[:100]}", latency_ms)

        except Exception as e:
            logger.warning(f"Judge error (attempt {attempt+1}): {e}")
            if attempt < JUDGE_MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))

    latency_ms = (time.perf_counter() - start_time) * 1000
    return (None, "Judge failed after retries", latency_ms)


def run_document_level(
    queries: list[dict],
    retriever: HybridRetriever,
) -> dict[str, int]:
    """Run document-level retrieval + judge for all queries.

    Returns:
        {query_id: grade}
    """
    doc_grades: dict[str, int] = {}

    for q in tqdm(queries, desc="Document-level retrieval"):
        qid = q.get("id", "")
        query_text = q.get("query", "")
        expected_answer = q.get("expected_answer", "")

        try:
            doc_results = retriever.retrieve_documents(query_text, k=3)
        except Exception as e:
            logger.error(f"Document retrieval failed for {qid}: {e}")
            doc_grades[qid] = 1
            continue

        # Get full content for top documents
        all_chunks: list[dict] = []
        for doc in doc_results[:3]:
            doc_id = doc.get("doc_id", "")
            if doc_id:
                try:
                    chunks = retriever.storage.get_chunks_by_doc(doc_id)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to get chunks for doc {doc_id}: {e}")

        grade, reasoning, latency = judge_document_context(
            query_text, expected_answer, all_chunks
        )

        if grade is not None:
            doc_grades[qid] = grade
        else:
            doc_grades[qid] = 1

    return doc_grades


# =============================================================================
# PART 3: PRECISION@10
# =============================================================================

def compute_precision_at_10(baseline_path: Path) -> float:
    """Compute Precision@10 from baseline results for labeled queries."""
    with open(baseline_path) as f:
        data = json.load(f)

    precisions = []
    for result in data.get("per_query_results", []):
        relevant_doc_ids = result.get("relevant_doc_ids", [])
        if not relevant_doc_ids:
            continue

        retrieved_doc_ids = result.get("retrieved_doc_ids", [])
        # Count how many retrieved docs match relevant docs
        relevant_count = 0
        for ret_id in retrieved_doc_ids[:10]:
            if any(rel_id in ret_id for rel_id in relevant_doc_ids):
                relevant_count += 1

        precisions.append(relevant_count / 10.0)

    return sum(precisions) / len(precisions) if precisions else 0.0


# =============================================================================
# PART 4: ORACLE COMPUTATION
# =============================================================================

def compute_oracle(
    all_grades: dict[str, dict[str, int]],
    queries: list[dict],
) -> dict:
    """Compute oracle performance from all approach grades.

    Returns dict with oracle metrics.
    """
    per_query_oracle = []
    oracle_can_answer = 0
    total = 0

    # Per-type tracking
    by_type: dict[str, dict] = {}

    for q in queries:
        qid = q.get("id", "")
        qtype = q.get("query_type", "unknown")
        total += 1

        # Collect grades from all approaches
        approach_grades: dict[str, int] = {}
        for approach_name, grades in all_grades.items():
            if qid in grades:
                approach_grades[approach_name] = grades[qid]

        if not approach_grades:
            per_query_oracle.append({
                "query_id": qid,
                "query_type": qtype,
                "oracle_grade": 0,
                "best_approach": "none",
                "all_grades": {},
                "can_answer": False,
            })
            continue

        # Oracle = max grade across all approaches
        best_approach = max(approach_grades, key=approach_grades.get)  # type: ignore[arg-type]
        oracle_grade = approach_grades[best_approach]
        can_answer = oracle_grade >= SUCCESS_THRESHOLD

        if can_answer:
            oracle_can_answer += 1

        # Map best approach to granularity
        granularity_map = {name: gran for name, (_, gran) in APPROACH_FILES.items()}
        granularity_map["document"] = "document"
        best_granularity = granularity_map.get(best_approach, "unknown")

        # Get baseline grade for comparison
        baseline_grade = all_grades.get("baseline", {}).get(qid, 0)

        per_query_oracle.append({
            "query_id": qid,
            "query_type": qtype,
            "oracle_grade": oracle_grade,
            "baseline_grade": baseline_grade,
            "best_approach": best_approach,
            "best_granularity": best_granularity,
            "all_grades": approach_grades,
            "can_answer": can_answer,
            "oracle_helps": can_answer and baseline_grade < SUCCESS_THRESHOLD,
        })

        # Track per-type
        if qtype not in by_type:
            by_type[qtype] = {"total": 0, "oracle_can": 0, "baseline_can": 0}
        by_type[qtype]["total"] += 1
        if can_answer:
            by_type[qtype]["oracle_can"] += 1
        if baseline_grade >= SUCCESS_THRESHOLD:
            by_type[qtype]["baseline_can"] += 1

    # Compute aggregates
    oracle_asr = (oracle_can_answer / total * 100) if total > 0 else 0.0
    baseline_asr = sum(1 for p in per_query_oracle if p.get("baseline_grade", 0) >= SUCCESS_THRESHOLD) / total * 100 if total > 0 else 0.0

    # Granularity distribution
    granularity_dist: dict[str, int] = {}
    for p in per_query_oracle:
        if p["can_answer"]:
            g = p.get("best_granularity", "unknown")
            granularity_dist[g] = granularity_dist.get(g, 0) + 1

    # Per-type oracle ASR
    oracle_by_type = {}
    for qtype, counts in by_type.items():
        t = counts["total"]
        oracle_by_type[qtype] = {
            "total": t,
            "oracle_asr": (counts["oracle_can"] / t * 100) if t > 0 else 0.0,
            "baseline_asr": (counts["baseline_can"] / t * 100) if t > 0 else 0.0,
            "gap": ((counts["oracle_can"] - counts["baseline_can"]) / t * 100) if t > 0 else 0.0,
        }

    queries_where_oracle_helps = sum(1 for p in per_query_oracle if p.get("oracle_helps", False))

    return {
        "oracle_overall": {
            "oracle_asr": round(oracle_asr, 1),
            "baseline_asr": round(baseline_asr, 1),
            "gap": round(oracle_asr - baseline_asr, 1),
            "total_queries": total,
            "oracle_can_answer": oracle_can_answer,
            "queries_where_oracle_helps": queries_where_oracle_helps,
        },
        "oracle_by_type": oracle_by_type,
        "granularity_distribution": granularity_dist,
        "per_query_oracle": per_query_oracle,
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    oracle_data: dict,
    precision_at_10: float,
    doc_level_run: bool,
) -> str:
    overall = oracle_data["oracle_overall"]
    by_type = oracle_data["oracle_by_type"]
    gran_dist = oracle_data["granularity_distribution"]
    per_query = oracle_data["per_query_oracle"]

    lines = [
        "# Oracle Performance & Precision@10",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall['total_queries']}",
        f"**Document-level retrieval:** {'Yes' if doc_level_run else 'No (existing data only)'}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"The oracle (best possible granularity selection per query) achieves **{overall['oracle_asr']}% ASR**, "
        f"compared to the baseline's **{overall['baseline_asr']}% ASR** — a gap of **{overall['gap']}%**.",
        "",
        f"The oracle improves {overall['queries_where_oracle_helps']} queries over baseline "
        f"({overall['oracle_can_answer']}/{overall['total_queries']} can answer vs "
        f"{overall['total_queries'] - (overall['total_queries'] - overall['oracle_can_answer'])} baseline).",
        "",
        "---",
        "",
        "## Oracle vs Baseline by Query Type",
        "",
        "| Type | Count | Oracle ASR | Baseline ASR | Gap |",
        "|------|-------|------------|-------------|-----|",
    ]

    for qtype in sorted(by_type.keys()):
        t = by_type[qtype]
        lines.append(
            f"| {qtype} | {t['total']} | {t['oracle_asr']:.1f}% | {t['baseline_asr']:.1f}% | +{t['gap']:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Optimal Granularity Distribution",
        "",
        "For queries where the oracle can answer, which granularity provided the best result:",
        "",
        "| Granularity | Count | % |",
        "|-------------|-------|---|",
    ])

    total_answerable = sum(gran_dist.values())
    for gran, count in sorted(gran_dist.items(), key=lambda x: -x[1]):
        pct = (count / total_answerable * 100) if total_answerable > 0 else 0
        lines.append(f"| {gran} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        f"## Precision@10 (Baseline)",
        "",
        f"**Precision@10:** {precision_at_10:.3f}",
        "",
        "Computed on labeled queries with ground-truth relevant_doc_ids.",
        "Precision@10 = (relevant documents in top 10 results) / 10, averaged across labeled queries.",
        "",
        "---",
        "",
        "## Queries Where Oracle Helps (Baseline Fails, Oracle Succeeds)",
        "",
    ])

    helped = [p for p in per_query if p.get("oracle_helps", False)]
    if helped:
        lines.append("| Query ID | Type | Baseline Grade | Oracle Grade | Best Approach |")
        lines.append("|----------|------|---------------|-------------|---------------|")
        for p in sorted(helped, key=lambda x: x["baseline_grade"]):
            lines.append(
                f"| {p['query_id']} | {p['query_type']} | {p['baseline_grade']}/10 | "
                f"{p['oracle_grade']}/10 | {p['best_approach']} |"
            )
    else:
        lines.append("No queries where oracle helps — baseline captures all answerable queries.")

    lines.extend([
        "",
        "---",
        "",
        "## Queries That Fail Across ALL Approaches",
        "",
    ])

    always_fail = [p for p in per_query if not p["can_answer"] and p.get("all_grades")]
    if always_fail:
        lines.append("| Query ID | Type | Max Grade | Notes |")
        lines.append("|----------|------|-----------|-------|")
        for p in sorted(always_fail, key=lambda x: x["query_id"]):
            max_g = p["oracle_grade"]
            lines.append(
                f"| {p['query_id']} | {p['query_type']} | {max_g}/10 | Corpus gap |"
            )
        lines.append("")
        lines.append(f"**{len(always_fail)} queries fail across ALL approaches** — these represent corpus coverage gaps, not retrieval deficiencies.")
    else:
        lines.append("All queries can be answered by at least one approach.")

    lines.extend([
        "",
        "---",
        "",
        f"*Generated by run_oracle.py on {datetime.now().isoformat()}*",
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Oracle performance & Precision@10")
    parser.add_argument("--skip-documents", action="store_true",
                        help="Skip document-level retrieval (use existing data only)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ORACLE PERFORMANCE & PRECISION@10")
    print("=" * 80)
    print()

    # Step 1: Load test queries
    print("[1/5] Loading test queries...")
    with open(TEST_QUERIES_PATH) as f:
        queries = json.load(f)["queries"]
    print(f"  Loaded {len(queries)} queries")
    print()

    # Step 2: Load existing per-query results
    print("[2/5] Loading existing approach results...")
    all_grades = load_approach_results()
    print(f"  Loaded grades from {len(all_grades)} approaches")
    print()

    # Step 3: Document-level retrieval (optional)
    doc_level_run = False
    if not args.skip_documents:
        print("[3/5] Running document-level retrieval...")
        if not Path(PLM_DB_PATH).exists():
            print(f"  WARNING: PLM database not found at {PLM_DB_PATH}, skipping")
        elif not HAS_LLM:
            print("  WARNING: LLM not available, skipping")
        else:
            retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
            doc_grades = run_document_level(queries, retriever)
            all_grades["document"] = doc_grades
            doc_level_run = True
            print(f"  Completed document-level for {len(doc_grades)} queries")
    else:
        print("[3/5] Skipping document-level retrieval (--skip-documents)")
    print()

    # Step 4: Compute oracle
    print("[4/5] Computing oracle performance...")
    oracle_data = compute_oracle(all_grades, queries)
    overall = oracle_data["oracle_overall"]
    print(f"  Oracle ASR: {overall['oracle_asr']}%")
    print(f"  Baseline ASR: {overall['baseline_asr']}%")
    print(f"  Gap: {overall['gap']}%")
    print(f"  Oracle helps {overall['queries_where_oracle_helps']} queries")
    print()

    # Step 5: Precision@10
    print("[5/5] Computing Precision@10...")
    baseline_path = RESULTS_DIR / "00_baseline.json"
    precision_at_10 = compute_precision_at_10(baseline_path)
    print(f"  Precision@10: {precision_at_10:.3f}")
    print()

    # Save results
    print("Saving results...")

    # JSON
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "approaches_used": list(all_grades.keys()),
            "document_level_run": doc_level_run,
        },
        **oracle_data,
        "precision_at_10": precision_at_10,
    }

    json_path = RESULTS_DIR / "oracle_performance.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Markdown
    report = generate_markdown_report(oracle_data, precision_at_10, doc_level_run)
    md_path = RESULTS_DIR / "oracle_performance.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"  Report: {md_path}")

    print()
    print("=" * 80)
    print("ORACLE PERFORMANCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
