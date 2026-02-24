#!/usr/bin/env python3
"""
Parent-Child Retrieval Benchmark Script for Adaptive Retrieval POC (TODO 04).

Implements parent-child retrieval: search on chunks (children), return heading-level
content (parents). Uses PLM's existing heading_id and get_chunks_by_heading() to
reconstruct parent context.

Algorithm:
  1. Retrieve top-20 chunks via hybrid search (children)
  2. Extract unique heading_ids (preserving match order)
  3. For each heading_id, fetch all sibling chunks and concatenate
  4. Return top-5 heading-level results (more content per result)

Usage:
    # Full run
    python run_parent_child.py

    # Test on limited queries
    python run_parent_child.py --limit 20

    # Skip LLM judge
    python run_parent_child.py --skip-judge --limit 10

Outputs:
    - results/15_parent_child.md (human-readable report)
    - results/15_parent_child.json (machine-readable results)
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

# Parent-child configuration
SEARCH_K = 20       # Over-retrieve children
RETURN_K = 5        # Return top-N parent headings (each is a full heading)
RETURN_LEVEL = "heading"  # heading or document

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
    retrieved_heading_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    num_child_chunks_matched: int = 0
    num_parent_headings_returned: int = 0
    retrieval_latency_ms: float = 0.0
    parent_fetch_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_retrieved: int = 0
    chunks_per_heading: list[int] = field(default_factory=list)

    # Ground truth
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None  # Rank of first relevant heading
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
    parent_fetch_latency_p50_ms: float = 0.0
    parent_fetch_latency_p95_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0
    avg_chunks_per_heading: float = 0.0


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

    parent_latencies = [r.parent_fetch_latency_ms for r in results]
    metrics.parent_fetch_latency_p50_ms, metrics.parent_fetch_latency_p95_ms, _ = \
        calculate_latency_percentiles(parent_latencies)

    tokens = [r.tokens_retrieved for r in results]
    metrics.avg_tokens_retrieved = sum(tokens) / len(tokens) if tokens else 0.0

    all_cph = [c for r in results for c in r.chunks_per_heading]
    metrics.avg_chunks_per_heading = sum(all_cph) / len(all_cph) if all_cph else 0.0

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


def format_headings_for_judge(heading_contents: list[dict]) -> str:
    """Format heading-level content for judge.

    Each heading_content is a dict with 'heading_id', 'heading_text', 'content' (concatenated chunks).
    """
    lines = []
    for i, h in enumerate(heading_contents, 1):
        heading_text = h.get("heading_text", "")
        content = h.get("content", "")[:1500]  # More content per heading, but cap it
        lines.append(f"[Section {i}] {heading_text}\n{content}")
    return "\n\n".join(lines)


def judge_answer_quality(
    query: str,
    expected_answer: str,
    heading_contents: list[dict],
    quiet: bool = False,
) -> tuple[Optional[int], JudgeCategory | None, str, float]:
    if not HAS_LLM or call_llm is None:
        return (None, None, "LLM not available", 0.0)

    if not heading_contents:
        return (1, "Cannot Answer", "No content retrieved", 0.0)

    formatted_prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        chunks_text=format_headings_for_judge(heading_contents),
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
# PARENT-CHILD RETRIEVAL
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


def parent_child_retrieve(
    retriever: HybridRetriever,
    query: str,
    search_k: int = SEARCH_K,
    return_k: int = RETURN_K,
    chunk_to_heading: dict[str, str] | None = None,
) -> tuple[list[dict], float, float, int]:
    """Retrieve using parent-child strategy.

    1. Search children (chunks) normally
    2. Extract unique heading_ids (order preserved)
    3. Fetch all chunks for each heading, concatenate as parent content
    4. Return top return_k headings

    Args:
        retriever: PLM HybridRetriever
        query: Query text
        search_k: How many children to retrieve
        return_k: How many parents to return
        chunk_to_heading: Pre-built chunk_id -> heading_id lookup

    Returns:
        (heading_contents, retrieval_latency_ms, parent_fetch_latency_ms, num_children)
    """
    # Step 1: Retrieve children
    retrieval_start = time.perf_counter()
    try:
        children = retriever.retrieve(query, k=search_k, use_rewrite=False)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        children = []
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    if not children:
        return ([], retrieval_ms, 0.0, 0)

    # Step 2: Get unique heading_ids preserving match order
    parent_start = time.perf_counter()

    seen_headings: dict[str, list[dict]] = {}  # heading_id -> matched children
    heading_order: list[str] = []

    for child in children:
        chunk_id = child.get("chunk_id", "")
        heading_id = chunk_to_heading.get(chunk_id) if chunk_to_heading else None

        if heading_id is None:
            continue

        if heading_id not in seen_headings:
            seen_headings[heading_id] = []
            heading_order.append(heading_id)
        seen_headings[heading_id].append(child)

    # Step 3: Fetch parent content for top return_k headings
    heading_contents = []
    for hid in heading_order[:return_k]:
        sibling_chunks = retriever.storage.get_chunks_by_heading(hid)
        if not sibling_chunks:
            continue

        # Get heading text from first chunk's heading field
        heading_text = sibling_chunks[0].get("heading", "") if sibling_chunks else ""

        # Concatenate all sibling chunks for the full heading content
        full_content = "\n".join(c.get("content", "") for c in sibling_chunks)

        heading_contents.append({
            "heading_id": hid,
            "heading_text": heading_text,
            "content": full_content,
            "doc_id": sibling_chunks[0].get("doc_id", "") if sibling_chunks else "",
            "num_chunks": len(sibling_chunks),
            "matched_children": len(seen_headings.get(hid, [])),
        })

    parent_fetch_ms = (time.perf_counter() - parent_start) * 1000

    return (heading_contents, retrieval_ms, parent_fetch_ms, len(children))


# =============================================================================
# CORE LOOP
# =============================================================================

def run_parent_child_measurement(
    queries: list[dict],
    retriever: HybridRetriever,
    chunk_to_heading: dict[str, str],
    search_k: int = SEARCH_K,
    return_k: int = RETURN_K,
    quiet: bool = False,
    skip_judge: bool = False,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    iterator = tqdm(queries, desc="Running parent-child") if not quiet else queries

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

        # Retrieve parent content
        heading_contents, retrieval_ms, parent_ms, num_children = parent_child_retrieve(
            retriever, query_text, search_k=search_k, return_k=return_k,
            chunk_to_heading=chunk_to_heading,
        )

        result.retrieval_latency_ms = retrieval_ms
        result.parent_fetch_latency_ms = parent_ms
        result.total_latency_ms = retrieval_ms + parent_ms
        result.num_child_chunks_matched = num_children
        result.num_parent_headings_returned = len(heading_contents)

        # Extract heading IDs and tokens
        for h in heading_contents:
            hid = h.get("heading_id", "")
            doc_id = h.get("doc_id", "")
            content = h.get("content", "")

            result.retrieved_heading_ids.append(hid)
            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += count_tokens(content)
            result.chunks_per_heading.append(h.get("num_chunks", 0))

        # Calculate rank by heading (which heading contains relevant doc?)
        if relevant_doc_ids:
            for i, h in enumerate(heading_contents, 1):
                doc_id = h.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break

        # LLM judge
        if not skip_judge:
            grade, category, reasoning, judge_latency = judge_answer_quality(
                query_text, expected_answer, heading_contents, quiet=quiet
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
                f"{query_id} | Grade: {grade_str} | Rank: {rank_str} | Headings: {len(heading_contents)}"
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
        "# Parent-Child Retrieval Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall_metrics.total_queries}",
        f"**Labeled queries:** {overall_metrics.labeled_queries}",
        f"**Configuration:** search_k={config['search_k']}, return_k={config['return_k']}, return_level={config['return_level']}, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Approach Summary",
        "",
        "Parent-child retrieval: search on chunks (children), return heading-level content (parents):",
        f"1. Retrieve top-{config['search_k']} chunks via hybrid search",
        "2. Extract unique heading_ids (preserving match order)",
        f"3. For each heading, fetch all sibling chunks and concatenate",
        f"4. Return top-{config['return_k']} heading-level results",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Parent-Child | Baseline | Delta |",
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
            f"| Parent fetch p50 | {overall_metrics.parent_fetch_latency_p50_ms:.0f}ms | — | — |",
            f"| Parent fetch p95 | {overall_metrics.parent_fetch_latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | {bl.get('avg_tokens_retrieved', 0):.0f} | {delta(overall_metrics.avg_tokens_retrieved, bl.get('avg_tokens_retrieved'), '.0f')} |",
            f"| Avg Chunks per Heading | {overall_metrics.avg_chunks_per_heading:.1f} | — | — |",
        ])
    else:
        lines.extend([
            f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} | — | — |",
            f"| **Answer Success Rate** | **{overall_metrics.answer_success_rate:.1f}%** | — | — |",
            f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 | — | — |",
            f"| Latency p95 (total) | {overall_metrics.latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | — | — |",
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

    lines.extend(["", "---", "", f"*Generated by run_parent_child.py on {datetime.now().isoformat()}*"])
    return "\n".join(lines)


def save_json_results(results, overall_metrics, by_type_metrics, config, output_path):
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "approach": "parent_child",
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
    parser = argparse.ArgumentParser(description="Run parent-child retrieval benchmark")
    parser.add_argument("--search-k", type=int, default=SEARCH_K, help=f"Children to retrieve (default: {SEARCH_K})")
    parser.add_argument("--return-k", type=int, default=RETURN_K, help=f"Parents to return (default: {RETURN_K})")
    parser.add_argument("--limit", type=int, default=None, help="Limit queries for testing")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-judge", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PLM PARENT-CHILD RETRIEVAL BENCHMARK")
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
    print(f"[5/7] Running parent-child retrieval (search_k={args.search_k}, return_k={args.return_k})...")
    if args.skip_judge:
        print("  NOTE: Skipping LLM-as-judge")

    results = run_parent_child_measurement(
        queries, retriever, chunk_to_heading,
        search_k=args.search_k, return_k=args.return_k,
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
    print(f"  Avg tokens: {overall.avg_tokens_retrieved:.0f}")
    print()

    # Step 7: Generate reports
    config = {
        "search_k": args.search_k,
        "return_k": args.return_k,
        "return_level": RETURN_LEVEL,
        "judge_model": JUDGE_MODEL,
        "skip_judge": args.skip_judge,
    }

    print("[7/7] Generating reports...")
    report = generate_markdown_report(results, overall, by_type, config, baseline_data)
    report_path = output_dir / "15_parent_child.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    json_path = output_dir / "15_parent_child.json"
    save_json_results(results, overall, by_type, config, json_path)
    print(f"  JSON: {json_path}")
    print()

    # Summary
    print("=" * 80)
    print("PARENT-CHILD BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  MRR@10: {overall.mrr_at_10:.3f}")
    print(f"  ASR: {overall.answer_success_rate:.1f}%")
    print(f"  Latency p95: {overall.latency_p95_ms:.0f}ms")
    print(f"  Avg tokens: {overall.avg_tokens_retrieved:.0f}")

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
