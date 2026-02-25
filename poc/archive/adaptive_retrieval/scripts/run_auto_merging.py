#!/usr/bin/env python3
"""
Auto-Merging Retrieval Benchmark Script for Adaptive Retrieval POC (TODO 05).

Implements auto-merging retrieval: retrieve chunks, group by heading, merge to parent
if enough siblings are retrieved (merge_threshold).

Algorithm:
  1. Retrieve top-10 chunks via hybrid search
  2. Build chunk_id -> heading_id lookup from DB
  3. Group retrieved chunks by heading_id
  4. For each heading: count total sibling chunks via get_chunks_by_heading()
  5. If (retrieved_count / total_siblings) >= merge_threshold AND total_siblings > 1:
     Replace with full heading content
  6. Otherwise: keep original individual chunks
  7. Return mixed result set

Usage:
    # Full run
    python run_auto_merging.py

    # Test on limited queries
    python run_auto_merging.py --limit 20

    # Skip LLM judge
    python run_auto_merging.py --skip-judge --limit 10

    # Adjust merge threshold
    python run_auto_merging.py --merge-threshold 0.6

Outputs:
    - results/02_auto_merging.md (human-readable report)
    - results/02_auto_merging.json (machine-readable results)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from tqdm import tqdm

# Add src to path for PLM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from plm.search.retriever import HybridRetriever

# Try to import LLM provider
try:
    from plm.shared.llm.base import call_llm
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    call_llm = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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

# Auto-merging configuration
RETRIEVE_K = 10         # Initial chunk retrieval
MERGE_THRESHOLD = 0.5   # Merge if retrieved_count/total_siblings >= this

QueryType = Literal["factoid", "procedural", "explanatory", "comparison", "troubleshooting"]
JudgeCategory = Literal["Correct", "Partially Correct", "Incorrect", "Cannot Answer"]

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: str
    query_type: QueryType
    query_text: str
    expected_answer: str
    optimal_granularity: str

    # Retrieval results
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    num_chunks_retrieved: int = 0
    num_headings_merged: int = 0
    num_chunks_kept: int = 0
    retrieval_latency_ms: float = 0.0
    merge_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_retrieved: int = 0
    merge_decisions: list[dict] = field(default_factory=list)  # Debug info

    # Ground truth
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None  # Rank of first relevant result
    hit_at_5: bool = False
    hit_at_10: bool = False

    # LLM judge results
    judge_grade: Optional[int] = None
    judge_category: Optional[JudgeCategory] = None
    judge_reasoning: str = ""
    judge_latency_ms: float = 0.0
    judge_failed: bool = False

    @property
    def can_answer(self) -> bool:
        return self.judge_grade is not None and self.judge_grade >= SUCCESS_THRESHOLD


@dataclass
class AggregateMetrics:
    total_queries: int = 0
    labeled_queries: int = 0
    judge_failures: int = 0

    mrr_at_10: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    recall_at_10: float = 0.0

    answer_success_rate: float = 0.0
    avg_grade: float = 0.0
    grade_distribution: dict[int, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    merge_latency_p50_ms: float = 0.0
    merge_latency_p95_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0

    # Auto-merging specific
    merge_rate: float = 0.0  # Percentage of queries where at least one merge happened
    avg_merges_per_query: float = 0.0
    avg_chunks_kept_per_query: float = 0.0


# =============================================================================
# METRICS (reused)
# =============================================================================

def calculate_mrr(ranks: list[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r if r is not None else 0.0 for r in ranks) / len(ranks)


def calculate_hit_at_k(ranks: list[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return (hits / len(ranks)) * 100


def calculate_latency_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return (0.0, 0.0, 0.0)
    arr = np.array(latencies)
    return (float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)))


def count_tokens(text: str) -> int:
    return len(text.split())


def grade_to_category(grade: int) -> JudgeCategory:
    if grade >= 9:
        return "Correct"
    elif grade >= 6:
        return "Partially Correct"
    elif grade >= 3:
        return "Incorrect"
    else:
        return "Cannot Answer"


def calculate_aggregate_metrics(results: list[QueryResult]) -> AggregateMetrics:
    metrics = AggregateMetrics()
    metrics.total_queries = len(results)

    labeled = [r for r in results if r.relevant_doc_ids]
    metrics.labeled_queries = len(labeled)

    judged = [r for r in results if not r.judge_failed and r.judge_grade is not None]
    metrics.judge_failures = sum(1 for r in results if r.judge_failed)

    if labeled:
        ranks = [r.rank for r in labeled]
        metrics.mrr_at_10 = calculate_mrr(ranks)
        metrics.hit_at_5 = calculate_hit_at_k(ranks, 5)
        metrics.hit_at_10 = calculate_hit_at_k(ranks, 10)
        metrics.recall_at_10 = metrics.hit_at_10

    if judged:
        grades = [r.judge_grade for r in judged if r.judge_grade is not None]
        success = sum(1 for r in judged if r.can_answer)
        metrics.answer_success_rate = (success / len(judged)) * 100
        metrics.avg_grade = sum(grades) / len(grades) if grades else 0.0

        for g in grades:
            metrics.grade_distribution[g] = metrics.grade_distribution.get(g, 0) + 1
        for r in judged:
            if r.judge_category:
                metrics.category_distribution[r.judge_category] = \
                    metrics.category_distribution.get(r.judge_category, 0) + 1

    total_latencies = [r.total_latency_ms for r in results]
    metrics.latency_p50_ms, metrics.latency_p95_ms, metrics.latency_p99_ms = \
        calculate_latency_percentiles(total_latencies)

    merge_latencies = [r.merge_latency_ms for r in results]
    metrics.merge_latency_p50_ms, metrics.merge_latency_p95_ms, _ = \
        calculate_latency_percentiles(merge_latencies)

    tokens = [r.tokens_retrieved for r in results]
    metrics.avg_tokens_retrieved = sum(tokens) / len(tokens) if tokens else 0.0

    # Auto-merging specific
    queries_with_merges = sum(1 for r in results if r.num_headings_merged > 0)
    metrics.merge_rate = (queries_with_merges / len(results)) * 100 if results else 0.0

    total_merges = sum(r.num_headings_merged for r in results)
    metrics.avg_merges_per_query = total_merges / len(results) if results else 0.0

    total_kept = sum(r.num_chunks_kept for r in results)
    metrics.avg_chunks_kept_per_query = total_kept / len(results) if results else 0.0

    return metrics


def calculate_metrics_by_query_type(results: list[QueryResult]) -> dict[QueryType, AggregateMetrics]:
    by_type: dict[QueryType, list[QueryResult]] = {}
    for r in results:
        by_type.setdefault(r.query_type, []).append(r)
    return {qt: calculate_aggregate_metrics(res) for qt, res in by_type.items()}


# =============================================================================
# LLM-AS-JUDGE (reused)
# =============================================================================

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


def format_results_for_judge(results: list[dict]) -> str:
    """Format mixed chunk/heading results for judge.

    Each result is either a chunk or a merged heading.
    """
    lines = []
    for i, r in enumerate(results, 1):
        result_type = r.get("type", "chunk")
        heading = r.get("heading", "")
        content = r.get("content", "")[:1500]  # Cap content length
        
        if result_type == "merged_heading":
            lines.append(f"[Merged Section {i}] {heading}\n{content}")
        else:
            lines.append(f"[Chunk {i}] {heading}\n{content}")
    return "\n\n".join(lines)


def judge_answer_quality(
    query: str,
    expected_answer: str,
    results: list[dict],
    quiet: bool = False,
) -> tuple[Optional[int], JudgeCategory | None, str, float]:
    if not HAS_LLM or call_llm is None:
        return (None, None, "LLM not available", 0.0)

    if not results:
        return (1, "Cannot Answer", "No content retrieved", 0.0)

    formatted_prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        chunks_text=format_results_for_judge(results),
    )

    start_time = time.perf_counter()

    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = call_llm(
                prompt=formatted_prompt,
                model=JUDGE_MODEL,
                max_tokens=200,
                temperature=0,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if not response:
                if not quiet:
                    logger.warning(f"Empty LLM response (attempt {attempt+1})")
                continue

            response_text = response.strip()
            try:
                if response_text.startswith("{"):
                    data = json.loads(response_text)
                else:
                    import re
                    match = re.search(r'\{[^}]+\}', response_text)
                    if match:
                        data = json.loads(match.group())
                    else:
                        raise ValueError("No JSON found")

                grade = max(1, min(10, int(data.get("grade", 5))))
                reasoning = data.get("reasoning", "")
                return (grade, grade_to_category(grade), reasoning, latency_ms)

            except (json.JSONDecodeError, ValueError) as e:
                import re
                match = re.search(r'grade["\s:]+(\d+)', response_text, re.IGNORECASE)
                if match:
                    grade = max(1, min(10, int(match.group(1))))
                    return (grade, grade_to_category(grade), f"Extracted: {response_text[:100]}", latency_ms)
                if not quiet:
                    logger.warning(f"Parse failure (attempt {attempt+1}): {e}")

        except Exception as e:
            if not quiet:
                logger.warning(f"Judge API error (attempt {attempt+1}): {e}")
            if attempt < JUDGE_MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))

    latency_ms = (time.perf_counter() - start_time) * 1000
    return (None, None, "Judge failed after retries", latency_ms)


# =============================================================================
# AUTO-MERGING RETRIEVAL
# =============================================================================

def build_heading_id_lookup(db_path: str) -> dict[str, str]:
    """Build chunk_id -> heading_id lookup from DB.

    Returns a dict mapping chunk ID to its heading_id.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT id, heading_id FROM chunks WHERE heading_id IS NOT NULL")
    lookup = {row[0]: row[1] for row in cursor}
    conn.close()
    return lookup


def auto_merge_retrieve(
    retriever: HybridRetriever,
    query: str,
    retrieve_k: int = RETRIEVE_K,
    merge_threshold: float = MERGE_THRESHOLD,
    chunk_to_heading: dict[str, str] | None = None,
) -> tuple[list[dict], float, float, list[dict]]:
    """Retrieve using auto-merging strategy.

    1. Retrieve top-k chunks
    2. Group by heading_id
    3. For each heading: if (retrieved_count / total_siblings) >= threshold AND total_siblings > 1,
       replace with full heading content
    4. Otherwise keep original chunks
    5. Return mixed result set

    Args:
        retriever: PLM HybridRetriever
        query: Query text
        retrieve_k: How many chunks to retrieve
        merge_threshold: Merge if retrieved_count/total_siblings >= this
        chunk_to_heading: Pre-built chunk_id -> heading_id lookup

    Returns:
        (results, retrieval_latency_ms, merge_latency_ms, merge_decisions)
    """
    # Step 1: Retrieve chunks
    retrieval_start = time.perf_counter()
    try:
        chunks = retriever.retrieve(query, k=retrieve_k, use_rewrite=False)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        chunks = []
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    if not chunks:
        return ([], retrieval_ms, 0.0, [])

    # Step 2: Group chunks by heading_id
    merge_start = time.perf_counter()

    heading_groups: dict[str, list[dict]] = {}  # heading_id -> chunks
    orphan_chunks: list[dict] = []  # Chunks without heading_id

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        heading_id = chunk_to_heading.get(chunk_id) if chunk_to_heading else None

        if heading_id is None:
            orphan_chunks.append(chunk)
        else:
            if heading_id not in heading_groups:
                heading_groups[heading_id] = []
            heading_groups[heading_id].append(chunk)

    # Step 3: Decide merge vs keep for each heading
    results = []
    merge_decisions = []

    for heading_id, retrieved_chunks in heading_groups.items():
        # Get all sibling chunks for this heading
        all_siblings = retriever.storage.get_chunks_by_heading(heading_id)
        total_siblings = len(all_siblings)
        retrieved_count = len(retrieved_chunks)

        # Merge decision
        merge_ratio = retrieved_count / total_siblings if total_siblings > 0 else 0.0
        should_merge = (merge_ratio >= merge_threshold) and (total_siblings > 1)

        decision = {
            "heading_id": heading_id,
            "retrieved_count": retrieved_count,
            "total_siblings": total_siblings,
            "merge_ratio": merge_ratio,
            "merged": should_merge,
        }
        merge_decisions.append(decision)

        if should_merge:
            # Merge: return full heading content
            heading_text = all_siblings[0].get("heading", "") if all_siblings else ""
            full_content = "\n".join(c.get("content", "") for c in all_siblings)
            doc_id = all_siblings[0].get("doc_id", "") if all_siblings else ""

            results.append({
                "type": "merged_heading",
                "heading_id": heading_id,
                "heading": heading_text,
                "content": full_content,
                "doc_id": doc_id,
                "num_chunks": total_siblings,
            })
        else:
            # Keep: return original chunks
            for chunk in retrieved_chunks:
                results.append({
                    "type": "chunk",
                    "chunk_id": chunk.get("chunk_id", ""),
                    "heading": chunk.get("heading", ""),
                    "content": chunk.get("content", ""),
                    "doc_id": chunk.get("doc_id", ""),
                })

    # Add orphan chunks (no heading_id)
    for chunk in orphan_chunks:
        results.append({
            "type": "chunk",
            "chunk_id": chunk.get("chunk_id", ""),
            "heading": chunk.get("heading", ""),
            "content": chunk.get("content", ""),
            "doc_id": chunk.get("doc_id", ""),
        })

    merge_ms = (time.perf_counter() - merge_start) * 1000

    return (results, retrieval_ms, merge_ms, merge_decisions)


# =============================================================================
# CORE LOOP
# =============================================================================

def run_auto_merging_measurement(
    queries: list[dict],
    retriever: HybridRetriever,
    chunk_to_heading: dict[str, str],
    retrieve_k: int = RETRIEVE_K,
    merge_threshold: float = MERGE_THRESHOLD,
    quiet: bool = False,
    skip_judge: bool = False,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    iterator = tqdm(queries, desc="Running auto-merging") if not quiet else queries

    for q in iterator:
        query_id = q.get("id", "unknown")
        query_type = q.get("query_type", "unknown")
        query_text = q.get("query", "")
        expected_answer = q.get("expected_answer", "")
        optimal_granularity = q.get("optimal_granularity", "chunk")
        relevant_doc_ids = q.get("relevant_doc_ids", [])

        result = QueryResult(
            query_id=query_id,
            query_type=query_type,
            query_text=query_text,
            expected_answer=expected_answer,
            optimal_granularity=optimal_granularity,
            relevant_doc_ids=relevant_doc_ids,
        )

        # Retrieve with auto-merging
        merged_results, retrieval_ms, merge_ms, merge_decisions = auto_merge_retrieve(
            retriever, query_text, retrieve_k=retrieve_k,
            merge_threshold=merge_threshold, chunk_to_heading=chunk_to_heading,
        )

        result.retrieval_latency_ms = retrieval_ms
        result.merge_latency_ms = merge_ms
        result.total_latency_ms = retrieval_ms + merge_ms
        result.num_chunks_retrieved = retrieve_k
        result.merge_decisions = merge_decisions

        # Count merges and kept chunks
        for r in merged_results:
            if r.get("type") == "merged_heading":
                result.num_headings_merged += 1
            else:
                result.num_chunks_kept += 1

        # Extract IDs and tokens
        for r in merged_results:
            doc_id = r.get("doc_id", "")
            content = r.get("content", "")

            if r.get("type") == "chunk":
                chunk_id = r.get("chunk_id", "")
                result.retrieved_chunk_ids.append(chunk_id)

            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += count_tokens(content)

        # Calculate rank
        if relevant_doc_ids:
            for i, r in enumerate(merged_results, 1):
                doc_id = r.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break

        # LLM judge
        if not skip_judge:
            grade, category, reasoning, judge_latency = judge_answer_quality(
                query_text, expected_answer, merged_results, quiet=quiet
            )
            result.judge_grade = grade
            result.judge_category = category
            result.judge_reasoning = reasoning
            result.judge_latency_ms = judge_latency
            result.judge_failed = (grade is None)

        if not quiet and isinstance(iterator, tqdm):
            grade_str = f"{result.judge_grade}/10" if result.judge_grade else "N/A"
            rank_str = str(result.rank) if result.rank else "-"
            iterator.set_postfix_str(
                f"{query_id} | Grade: {grade_str} | Rank: {rank_str} | Merged: {result.num_headings_merged}"
            )

        results.append(result)

    return results


# =============================================================================
# BASELINE LOADING
# =============================================================================

def load_baseline_results() -> tuple[Optional[dict], Optional[dict[str, dict]]]:
    baseline_json = RESULTS_DIR / "00_baseline.json"
    if not baseline_json.exists():
        return (None, None)

    with open(baseline_json) as f:
        data = json.load(f)

    overall = data.get("overall_metrics")
    by_type = data.get("by_query_type", {})
    per_query = {}
    for r in data.get("per_query_results", []):
        qid = r.get("query_id", "")
        if qid:
            per_query[qid] = r

    return ({"overall": overall, "by_type": by_type}, per_query)


# =============================================================================
# REPORT
# =============================================================================

def generate_markdown_report(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    config: dict,
    baseline: Optional[dict] = None,
) -> str:
    bl_overall = baseline["overall"] if baseline else None
    bl_by_type = baseline["by_type"] if baseline else None

    def delta(current: float, bl_val: Optional[float], fmt: str = ".1f") -> str:
        if bl_val is None:
            return ""
        diff = current - bl_val
        sign = "+" if diff >= 0 else ""
        return f" ({sign}{diff:{fmt}})"

    lines = [
        "# Auto-Merging Retrieval Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall_metrics.total_queries}",
        f"**Labeled queries:** {overall_metrics.labeled_queries}",
        f"**Configuration:** retrieve_k={config['retrieve_k']}, merge_threshold={config['merge_threshold']}, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Approach Summary",
        "",
        "Auto-merging retrieval: retrieve chunks, merge to parent heading if threshold met:",
        f"1. Retrieve top-{config['retrieve_k']} chunks via hybrid search",
        "2. Group chunks by heading_id",
        "3. For each heading: count total sibling chunks",
        f"4. If (retrieved_count / total_siblings) >= {config['merge_threshold']} AND total_siblings > 1:",
        "   - Replace with full heading content (all sibling chunks concatenated)",
        "5. Otherwise: keep original individual chunks",
        "6. Return mixed result set (merged headings + individual chunks)",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Auto-Merging | Baseline | Delta |",
        "|--------|-------------|----------|-------|",
    ]

    if bl_overall:
        bl = bl_overall
        lines.extend([
            f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} | {bl.get('mrr_at_10', 0):.3f} | {delta(overall_metrics.mrr_at_10, bl.get('mrr_at_10'))} |",
            f"| Hit@5 (labeled) | {overall_metrics.hit_at_5:.1f}% | {bl.get('hit_at_5', 0):.1f}% | {delta(overall_metrics.hit_at_5, bl.get('hit_at_5'))} |",
            f"| **Answer Success Rate** | **{overall_metrics.answer_success_rate:.1f}%** | **{bl.get('answer_success_rate', 0):.1f}%** | **{delta(overall_metrics.answer_success_rate, bl.get('answer_success_rate'))}** |",
            f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 | {bl.get('avg_grade', 0):.2f}/10 | {delta(overall_metrics.avg_grade, bl.get('avg_grade'), '.2f')} |",
            f"| Latency p50 (total) | {overall_metrics.latency_p50_ms:.0f}ms | {bl.get('latency_p50_ms', 0):.0f}ms | {delta(overall_metrics.latency_p50_ms, bl.get('latency_p50_ms'), '.0f')}ms |",
            f"| Latency p95 (total) | {overall_metrics.latency_p95_ms:.0f}ms | {bl.get('latency_p95_ms', 0):.0f}ms | {delta(overall_metrics.latency_p95_ms, bl.get('latency_p95_ms'), '.0f')}ms |",
            f"| Merge latency p50 | {overall_metrics.merge_latency_p50_ms:.0f}ms | — | — |",
            f"| Merge latency p95 | {overall_metrics.merge_latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | {bl.get('avg_tokens_retrieved', 0):.0f} | {delta(overall_metrics.avg_tokens_retrieved, bl.get('avg_tokens_retrieved'), '.0f')} |",
            f"| Merge Rate | {overall_metrics.merge_rate:.1f}% | — | — |",
            f"| Avg Merges per Query | {overall_metrics.avg_merges_per_query:.2f} | — | — |",
            f"| Avg Chunks Kept per Query | {overall_metrics.avg_chunks_kept_per_query:.2f} | — | — |",
        ])
    else:
        lines.extend([
            f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} | — | — |",
            f"| **Answer Success Rate** | **{overall_metrics.answer_success_rate:.1f}%** | — | — |",
            f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 | — | — |",
            f"| Latency p95 (total) | {overall_metrics.latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | — | — |",
            f"| Merge Rate | {overall_metrics.merge_rate:.1f}% | — | — |",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## Performance by Query Type",
        "",
        "| Type | Count | MRR@10 | Hit@5 | Success Rate | Avg Grade | Baseline SR | Delta |",
        "|------|-------|--------|-------|--------------|-----------|-------------|-------|",
    ])

    sorted_types = sorted(by_type_metrics.items(), key=lambda x: x[1].answer_success_rate)
    for qt, m in sorted_types:
        bl_sr = "—"
        dt = "—"
        if bl_by_type and qt in bl_by_type:
            bl_sr_val = bl_by_type[qt].get("answer_success_rate", 0)
            bl_sr = f"{bl_sr_val:.1f}%"
            diff = m.answer_success_rate - bl_sr_val
            dt = f"{'+' if diff >= 0 else ''}{diff:.1f}%"
        lines.append(
            f"| {qt} | {m.total_queries} | {m.mrr_at_10:.3f} | {m.hit_at_5:.1f}% | "
            f"{m.answer_success_rate:.1f}% | {m.avg_grade:.2f} | {bl_sr} | {dt} |"
        )

    # Success criteria
    lines.extend(["", "---", "", "## Success Criteria Evaluation", ""])
    if bl_overall:
        bl_asr = bl_overall.get("answer_success_rate", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        bl_p95 = bl_overall.get("latency_p95_ms", 0)
        lat_delta = overall_metrics.latency_p95_ms - bl_p95

        lines.extend([
            "| Criterion | Threshold | Actual | Pass? |",
            "|-----------|-----------|--------|-------|",
            f"| Answer Success Rate ≥+10% | +10.0% | {asr_delta:+.1f}% | {'✅' if asr_delta >= 10 else '❌'} |",
            f"| Latency increase ≤500ms (p95) | ≤500ms | {lat_delta:+.0f}ms | {'✅' if lat_delta <= 500 else '❌'} |",
        ])

        no_regress = True
        if bl_by_type:
            for qt, m in by_type_metrics.items():
                if qt in bl_by_type:
                    qt_delta = m.answer_success_rate - bl_by_type[qt].get("answer_success_rate", 0)
                    if qt_delta < -5:
                        no_regress = False
                        lines.append(f"| No type regresses >5% | >-5% | {qt}: {qt_delta:+.1f}% | ❌ |")
        if no_regress:
            lines.append("| No type regresses >5% | >-5% | All OK | ✅ |")

    # Category distribution
    lines.extend(["", "---", "", "## Category Distribution", ""])
    for cat, count in sorted(overall_metrics.category_distribution.items(), key=lambda x: -x[1]):
        pct = (count / overall_metrics.total_queries) * 100
        lines.append(f"- {cat}: {count} ({pct:.1f}%)")

    # Failed queries
    failed = [r for r in results if r.judge_failed or (r.judge_grade is not None and r.judge_grade < SUCCESS_THRESHOLD)]
    if failed:
        lines.extend(["", "## Sample Failed Queries (Top 10)", ""])
        for r in failed[:10]:
            grade_str = f"{r.judge_grade}/10" if r.judge_grade else "JUDGE_FAILED"
            lines.append(f"- **{r.query_id}** ({r.query_type}): {grade_str}")
            lines.append(f"  - Query: \"{r.query_text[:80]}...\"")
            if r.judge_reasoning:
                lines.append(f"  - Reasoning: {r.judge_reasoning[:120]}...")
            lines.append("")

    # Decision
    lines.extend(["", "---", "", "## Decision", ""])
    if bl_overall:
        bl_asr = bl_overall.get("answer_success_rate", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        lat_delta = overall_metrics.latency_p95_ms - bl_overall.get("latency_p95_ms", 0)

        if asr_delta >= 10 and lat_delta <= 500:
            lines.append("**RECOMMEND** — Meets all success criteria.")
        elif asr_delta >= 5:
            lines.append("**NEEDS MODIFICATION** — Shows promise but doesn't meet ≥10% ASR threshold.")
        elif asr_delta >= 0:
            lines.append("**REJECT** — Marginal improvement doesn't justify changes.")
        else:
            lines.append("**REJECT** — Regression from baseline.")

        lines.append("")
        lines.append(f"Answer Success Rate delta: {asr_delta:+.1f}% (threshold: +10%)")
        lines.append(f"Latency increase (p95): {lat_delta:+.0f}ms (threshold: ≤500ms)")
        lines.append(f"Merge rate: {overall_metrics.merge_rate:.1f}% of queries")

    lines.extend(["", "---", "", f"*Generated by run_auto_merging.py on {datetime.now().isoformat()}*"])
    return "\n".join(lines)


def save_json_results(results, overall_metrics, by_type_metrics, config, output_path):
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "approach": "auto_merging",
            "config": config,
            "query_counts": {
                "total": overall_metrics.total_queries,
                "labeled": overall_metrics.labeled_queries,
                "judge_failures": overall_metrics.judge_failures,
            },
        },
        "overall_metrics": asdict(overall_metrics),
        "by_query_type": {qt: asdict(m) for qt, m in by_type_metrics.items()},
        "per_query_results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run auto-merging retrieval benchmark")
    parser.add_argument("--merge-threshold", type=float, default=MERGE_THRESHOLD, help=f"Merge threshold (default: {MERGE_THRESHOLD})")
    parser.add_argument("--retrieve-k", type=int, default=RETRIEVE_K, help=f"Chunks to retrieve (default: {RETRIEVE_K})")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries for testing")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-judge", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PLM AUTO-MERGING RETRIEVAL BENCHMARK")
    print("=" * 80)
    print()

    # Step 1: Load test queries
    print("[1/7] Loading test queries...")
    if not TEST_QUERIES_PATH.exists():
        print(f"ERROR: Test queries not found at {TEST_QUERIES_PATH}")
        sys.exit(1)

    with open(TEST_QUERIES_PATH) as f:
        data = json.load(f)
    queries = data["queries"]
    if args.limit:
        queries = queries[:args.limit]
        print(f"  Limited to {args.limit} queries")

    print(f"  Loaded {len(queries)} queries")
    type_counts: dict[str, int] = {}
    for q in queries:
        type_counts[q.get("query_type", "unknown")] = type_counts.get(q.get("query_type", "unknown"), 0) + 1
    for qt, count in sorted(type_counts.items()):
        print(f"    - {qt}: {count}")
    print()

    # Step 2: Load baseline
    print("[2/7] Loading baseline results...")
    baseline_data, _ = load_baseline_results()
    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        print(f"  Baseline: ASR={bl_asr:.1f}%")
    else:
        print("  WARNING: No baseline found")
    print()

    # Step 3: Initialize retriever
    print("[3/7] Initializing PLM HybridRetriever...")
    if not Path(PLM_DB_PATH).exists():
        print(f"ERROR: DB not found at {PLM_DB_PATH}")
        sys.exit(1)
    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    print("  Retriever initialized")
    print()

    # Step 4: Build heading lookup
    print("[4/7] Building chunk->heading lookup...")
    chunk_to_heading = build_heading_id_lookup(PLM_DB_PATH)
    print(f"  Mapped {len(chunk_to_heading)} chunks to headings")
    print()

    # Step 5: Run benchmark
    print(f"[5/7] Running auto-merging retrieval (retrieve_k={args.retrieve_k}, merge_threshold={args.merge_threshold})...")
    if args.skip_judge:
        print("  NOTE: Skipping LLM-as-judge")

    results = run_auto_merging_measurement(
        queries, retriever, chunk_to_heading,
        retrieve_k=args.retrieve_k, merge_threshold=args.merge_threshold,
        quiet=args.quiet, skip_judge=args.skip_judge,
    )

    judge_success = sum(1 for r in results if not r.judge_failed)
    print(f"\n  Completed {len(results)} queries")
    if not args.skip_judge:
        print(f"  Judge success: {judge_success}/{len(results)} ({100*judge_success/len(results):.1f}%)")
    print()

    # Step 6: Calculate metrics
    print("[6/7] Calculating metrics...")
    overall = calculate_aggregate_metrics(results)
    by_type = calculate_metrics_by_query_type(results)
    print(f"  MRR@10: {overall.mrr_at_10:.3f}")
    print(f"  ASR: {overall.answer_success_rate:.1f}%")
    print(f"  Latency p95: {overall.latency_p95_ms:.0f}ms")
    print(f"  Merge rate: {overall.merge_rate:.1f}%")
    print(f"  Avg merges per query: {overall.avg_merges_per_query:.2f}")
    print()

    # Step 7: Generate reports
    config = {
        "retrieve_k": args.retrieve_k,
        "merge_threshold": args.merge_threshold,
        "judge_model": JUDGE_MODEL,
        "skip_judge": args.skip_judge,
    }

    print("[7/7] Generating reports...")
    report = generate_markdown_report(results, overall, by_type, config, baseline_data)
    report_path = output_dir / "02_auto_merging.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    json_path = output_dir / "02_auto_merging.json"
    save_json_results(results, overall, by_type, config, json_path)
    print(f"  JSON: {json_path}")
    print()

    # Summary
    print("=" * 80)
    print("AUTO-MERGING BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  MRR@10: {overall.mrr_at_10:.3f}")
    print(f"  ASR: {overall.answer_success_rate:.1f}%")
    print(f"  Latency p95: {overall.latency_p95_ms:.0f}ms")
    print(f"  Merge rate: {overall.merge_rate:.1f}%")
    print(f"  Avg merges per query: {overall.avg_merges_per_query:.2f}")

    if baseline_data:
        bl = baseline_data["overall"]
        asr_d = overall.answer_success_rate - bl.get("answer_success_rate", 0)
        lat_d = overall.latency_p95_ms - bl.get("latency_p95_ms", 0)
        print()
        print("vs Baseline:")
        print(f"  ASR: {asr_d:+.1f}% (target: +10%)")
        print(f"  Latency: {lat_d:+.0f}ms (limit: 500ms)")

    print()
    print(f"See {report_path} for full results.")


if __name__ == "__main__":
    main()
