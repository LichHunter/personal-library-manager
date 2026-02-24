#!/usr/bin/env python3
"""
Reranking Benchmark Script for Adaptive Retrieval POC (TODO 03).

Implements cross-encoder reranking: retrieve top-50 candidates via hybrid search,
re-score with a cross-encoder model, return top-10. Compares two models:
  - BAAI/bge-reranker-v2-m3 (560MB, high quality, ~3s on CPU)
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB, fast, ~175ms on CPU)

Usage:
    # Full run with BGE reranker (default)
    python run_reranking.py

    # Use fast MiniLM model
    python run_reranking.py --model minilm

    # Test on limited queries
    python run_reranking.py --limit 20

    # Skip LLM judge (quick latency test)
    python run_reranking.py --skip-judge --limit 10

Outputs:
    - results/01_reranking.md (human-readable report)
    - results/01_reranking.json (machine-readable results)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

# Paths
POC_ROOT = Path(__file__).parent.parent
TEST_QUERIES_PATH = POC_ROOT / "benchmarks" / "datasets" / "test_queries.json"
RESULTS_DIR = POC_ROOT / "results"

# PLM database paths (production index)
PLM_DB_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db"
PLM_BM25_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data"

# LLM configuration
JUDGE_MODEL = "haiku"
JUDGE_TIMEOUT = 30
JUDGE_MAX_RETRIES = 3
SUCCESS_THRESHOLD = 6  # Grade >= 6 is "can answer"

# Reranking configuration
RERANKER_MODELS = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
DEFAULT_RERANKER = "bge"
CANDIDATES_K = 50  # Retrieve top-50 before reranking
FINAL_K = 10       # Return top-10 after reranking

# Types
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
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_retrieved: int = 0

    # Reranking details
    reranker_scores: list[float] = field(default_factory=list)

    # Ground truth (if available)
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None
    hit_at_5: bool = False
    hit_at_10: bool = False

    # Baseline comparison (rank from baseline)
    baseline_rank: Optional[int] = None

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
    """Aggregate metrics for a set of query results."""
    total_queries: int = 0
    labeled_queries: int = 0
    judge_failures: int = 0

    # Retrieval quality
    mrr_at_10: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    recall_at_10: float = 0.0

    # Context sufficiency
    answer_success_rate: float = 0.0
    avg_grade: float = 0.0
    grade_distribution: dict[int, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)

    # Performance
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    reranking_latency_p50_ms: float = 0.0
    reranking_latency_p95_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0


# =============================================================================
# METRICS CALCULATIONS (reused from baseline)
# =============================================================================

def calculate_mrr(ranks: list[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    reciprocals = [1.0 / r if r is not None else 0.0 for r in ranks]
    return sum(reciprocals) / len(reciprocals)


def calculate_hit_at_k(ranks: list[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return (hits / len(ranks)) * 100


def calculate_latency_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return (0.0, 0.0, 0.0)
    arr = np.array(latencies)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


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

    labeled_results = [r for r in results if r.relevant_doc_ids]
    metrics.labeled_queries = len(labeled_results)

    judged_results = [r for r in results if not r.judge_failed and r.judge_grade is not None]
    metrics.judge_failures = sum(1 for r in results if r.judge_failed)

    # Retrieval quality (labeled only)
    if labeled_results:
        ranks = [r.rank for r in labeled_results]
        metrics.mrr_at_10 = calculate_mrr(ranks)
        metrics.hit_at_5 = calculate_hit_at_k(ranks, 5)
        metrics.hit_at_10 = calculate_hit_at_k(ranks, 10)
        metrics.recall_at_10 = metrics.hit_at_10

    # Context sufficiency (all judged)
    if judged_results:
        grades = [r.judge_grade for r in judged_results if r.judge_grade is not None]
        success_count = sum(1 for r in judged_results if r.can_answer)

        metrics.answer_success_rate = (success_count / len(judged_results)) * 100
        metrics.avg_grade = sum(grades) / len(grades) if grades else 0.0

        for grade in grades:
            metrics.grade_distribution[grade] = metrics.grade_distribution.get(grade, 0) + 1

        for r in judged_results:
            if r.judge_category:
                metrics.category_distribution[r.judge_category] = \
                    metrics.category_distribution.get(r.judge_category, 0) + 1

    # Performance (total latency = retrieval + reranking)
    total_latencies = [r.total_latency_ms for r in results]
    metrics.latency_p50_ms, metrics.latency_p95_ms, metrics.latency_p99_ms = \
        calculate_latency_percentiles(total_latencies)

    reranking_latencies = [r.reranking_latency_ms for r in results]
    metrics.reranking_latency_p50_ms, metrics.reranking_latency_p95_ms, _ = \
        calculate_latency_percentiles(reranking_latencies)

    tokens = [r.tokens_retrieved for r in results]
    metrics.avg_tokens_retrieved = sum(tokens) / len(tokens) if tokens else 0.0

    return metrics


def calculate_metrics_by_query_type(results: list[QueryResult]) -> dict[QueryType, AggregateMetrics]:
    by_type: dict[QueryType, list[QueryResult]] = {}
    for r in results:
        qt = r.query_type
        if qt not in by_type:
            by_type[qt] = []
        by_type[qt].append(r)
    return {qt: calculate_aggregate_metrics(res) for qt, res in by_type.items()}


# =============================================================================
# LLM-AS-JUDGE (reused from baseline)
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


def format_chunks_for_judge(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")[:500]
        heading = chunk.get("heading", "")
        lines.append(f"[Chunk {i}] {heading}\n{content}")
    return "\n\n".join(lines)


def judge_answer_quality(
    query: str,
    expected_answer: str,
    chunks: list[dict],
    quiet: bool = False,
) -> tuple[Optional[int], JudgeCategory | None, str, float]:
    if not HAS_LLM or call_llm is None:
        return (None, None, "LLM not available", 0.0)

    if not chunks:
        return (1, "Cannot Answer", "No chunks retrieved", 0.0)

    formatted_prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        chunks_text=format_chunks_for_judge(chunks),
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
                        raise ValueError("No JSON found in response")

                grade = int(data.get("grade", 5))
                grade = max(1, min(10, grade))
                reasoning = data.get("reasoning", "")
                category = grade_to_category(grade)

                return (grade, category, reasoning, latency_ms)

            except (json.JSONDecodeError, ValueError) as e:
                import re
                match = re.search(r'grade["\s:]+(\d+)', response_text, re.IGNORECASE)
                if match:
                    grade = int(match.group(1))
                    grade = max(1, min(10, grade))
                    return (grade, grade_to_category(grade), f"Extracted from: {response_text[:100]}", latency_ms)

                if not quiet:
                    logger.warning(f"Failed to parse judge response (attempt {attempt+1}): {e}")

        except Exception as e:
            if not quiet:
                logger.warning(f"Judge API error (attempt {attempt+1}): {e}")

            if attempt < JUDGE_MAX_RETRIES - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)

    latency_ms = (time.perf_counter() - start_time) * 1000
    return (None, None, "Judge failed after retries", latency_ms)


# =============================================================================
# RERANKING
# =============================================================================

def load_reranker(model_key: str) -> "CrossEncoder":
    """Load cross-encoder reranking model.

    Args:
        model_key: 'bge' or 'minilm'

    Returns:
        CrossEncoder model instance
    """
    from sentence_transformers import CrossEncoder

    model_name = RERANKER_MODELS[model_key]
    logger.info(f"Loading reranker: {model_name}")
    start = time.time()
    model = CrossEncoder(model_name)
    elapsed = time.time() - start
    logger.info(f"Reranker loaded in {elapsed:.1f}s")
    return model


def rerank_results(
    reranker: "CrossEncoder",
    query: str,
    candidates: list[dict],
    top_k: int = FINAL_K,
) -> tuple[list[dict], list[float], float]:
    """Rerank candidates using cross-encoder.

    Args:
        reranker: CrossEncoder model
        query: Query text
        candidates: List of retrieval result dicts (must have 'content' key)
        top_k: Number of results to return after reranking

    Returns:
        (reranked_results, scores, reranking_latency_ms)
    """
    if not candidates:
        return ([], [], 0.0)

    # Build query-document pairs using content field
    pairs = [[query, c.get("content", "")] for c in candidates]

    start = time.perf_counter()
    scores = reranker.predict(pairs)
    reranking_latency_ms = (time.perf_counter() - start) * 1000

    # Sort by score descending
    scored_indices = np.argsort(scores)[::-1][:top_k]

    reranked = [candidates[i] for i in scored_indices]
    reranked_scores = [float(scores[i]) for i in scored_indices]

    return (reranked, reranked_scores, reranking_latency_ms)


# =============================================================================
# CORE RETRIEVAL + RERANKING LOOP
# =============================================================================

def run_reranking_measurement(
    queries: list[dict],
    retriever: HybridRetriever,
    reranker: "CrossEncoder",
    candidates_k: int = CANDIDATES_K,
    final_k: int = FINAL_K,
    quiet: bool = False,
    skip_judge: bool = False,
    baseline_results: Optional[dict[str, dict]] = None,
) -> list[QueryResult]:
    """Run reranking measurement on all queries.

    Args:
        queries: List of query dicts from test_queries.json
        retriever: PLM HybridRetriever instance
        reranker: CrossEncoder model for reranking
        candidates_k: Number of candidates to retrieve before reranking
        final_k: Number of results to return after reranking
        quiet: Suppress verbose output
        skip_judge: Skip LLM-as-judge
        baseline_results: Optional dict of query_id -> baseline result for comparison

    Returns:
        List of QueryResult objects
    """
    results: list[QueryResult] = []

    iterator = tqdm(queries, desc="Running reranking") if not quiet else queries

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

        # Step 1: Retrieve top-K candidates (over-retrieve)
        retrieval_start = time.perf_counter()
        try:
            candidates = retriever.retrieve(query_text, k=candidates_k, use_rewrite=False)
        except Exception as e:
            logger.error(f"Retrieval failed for {query_id}: {e}")
            candidates = []
        result.retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 2: Rerank and take top final_k
        reranked, scores, reranking_ms = rerank_results(reranker, query_text, candidates, top_k=final_k)
        result.reranking_latency_ms = reranking_ms
        result.total_latency_ms = result.retrieval_latency_ms + result.reranking_latency_ms
        result.reranker_scores = scores

        # Extract IDs and count tokens from reranked results
        for sr in reranked:
            chunk_id = sr.get("chunk_id", "")
            doc_id = sr.get("doc_id", "")
            content = sr.get("content", "")

            result.retrieved_chunk_ids.append(chunk_id)
            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += count_tokens(content)

        # Calculate rank (if ground truth available)
        if relevant_doc_ids:
            for i, sr in enumerate(reranked, 1):
                doc_id = sr.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break

        # Store baseline rank for comparison
        if baseline_results and query_id in baseline_results:
            result.baseline_rank = baseline_results[query_id].get("rank")

        # Run LLM judge
        if not skip_judge:
            grade, category, reasoning, judge_latency = judge_answer_quality(
                query_text, expected_answer, reranked, quiet=quiet
            )
            result.judge_grade = grade
            result.judge_category = category
            result.judge_reasoning = reasoning
            result.judge_latency_ms = judge_latency
            result.judge_failed = (grade is None)

        # Verbose output
        if not quiet and isinstance(iterator, tqdm):
            grade_str = f"{result.judge_grade}/10" if result.judge_grade else "N/A"
            cat_str = result.judge_category or "N/A"
            rank_str = str(result.rank) if result.rank else "-"
            iterator.set_postfix_str(
                f"{query_id} | Grade: {grade_str} ({cat_str}) | Rank: {rank_str} | Rerank: {result.reranking_latency_ms:.0f}ms"
            )

        results.append(result)

    return results


# =============================================================================
# BASELINE LOADING
# =============================================================================

def load_baseline_results() -> tuple[Optional[dict], Optional[dict[str, dict]]]:
    """Load baseline results for comparison.

    Returns:
        (baseline_overall_metrics, per_query_results_dict)
    """
    baseline_json = RESULTS_DIR / "00_baseline.json"
    if not baseline_json.exists():
        logger.warning(f"Baseline results not found at {baseline_json}")
        return (None, None)

    with open(baseline_json) as f:
        data = json.load(f)

    overall = data.get("overall_metrics")
    by_type = data.get("by_query_type", {})

    # Build per-query lookup
    per_query = {}
    for r in data.get("per_query_results", []):
        qid = r.get("query_id", "")
        if qid:
            per_query[qid] = r

    return ({"overall": overall, "by_type": by_type}, per_query)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    config: dict,
    baseline: Optional[dict] = None,
) -> str:
    """Generate markdown report with baseline comparison."""

    bl_overall = baseline["overall"] if baseline else None
    bl_by_type = baseline["by_type"] if baseline else None

    def delta_str(current: float, baseline_val: Optional[float], fmt: str = ".1f", is_pct: bool = False) -> str:
        if baseline_val is None:
            return ""
        diff = current - baseline_val
        sign = "+" if diff >= 0 else ""
        suffix = "%" if is_pct else ""
        return f" ({sign}{diff:{fmt}}{suffix})"

    lines = [
        "# Reranking Approach Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall_metrics.total_queries}",
        f"**Labeled queries:** {overall_metrics.labeled_queries}",
        f"**Reranker model:** {config['reranker_model']}",
        f"**Configuration:** candidates_k={config['candidates_k']}, final_k={config['final_k']}, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Approach Summary",
        "",
        "Cross-encoder reranking as post-processing step:",
        f"1. Retrieve top-{config['candidates_k']} candidates via hybrid search",
        f"2. Re-score each (query, candidate) pair with cross-encoder ({config['reranker_model']})",
        f"3. Return top-{config['final_k']} by cross-encoder score",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Reranking | Baseline | Delta |",
        "|--------|-----------|----------|-------|",
    ]

    if bl_overall:
        bl_mrr = bl_overall.get("mrr_at_10", 0)
        bl_hit5 = bl_overall.get("hit_at_5", 0)
        bl_hit10 = bl_overall.get("hit_at_10", 0)
        bl_asr = bl_overall.get("answer_success_rate", 0)
        bl_grade = bl_overall.get("avg_grade", 0)
        bl_p50 = bl_overall.get("latency_p50_ms", 0)
        bl_p95 = bl_overall.get("latency_p95_ms", 0)
        bl_p99 = bl_overall.get("latency_p99_ms", 0)
        bl_tokens = bl_overall.get("avg_tokens_retrieved", 0)

        lines.extend([
            f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} | {bl_mrr:.3f} | {delta_str(overall_metrics.mrr_at_10, bl_mrr, '.3f')} |",
            f"| Hit@5 (labeled) | {overall_metrics.hit_at_5:.1f}% | {bl_hit5:.1f}% | {delta_str(overall_metrics.hit_at_5, bl_hit5)} |",
            f"| Hit@10 (labeled) | {overall_metrics.hit_at_10:.1f}% | {bl_hit10:.1f}% | {delta_str(overall_metrics.hit_at_10, bl_hit10)} |",
            f"| **Answer Success Rate** | **{overall_metrics.answer_success_rate:.1f}%** | **{bl_asr:.1f}%** | **{delta_str(overall_metrics.answer_success_rate, bl_asr)}** |",
            f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 | {bl_grade:.2f}/10 | {delta_str(overall_metrics.avg_grade, bl_grade, '.2f')} |",
            f"| Judge Failures | {overall_metrics.judge_failures}/{overall_metrics.total_queries} | — | — |",
            f"| Latency p50 (total) | {overall_metrics.latency_p50_ms:.0f}ms | {bl_p50:.0f}ms | {delta_str(overall_metrics.latency_p50_ms, bl_p50, '.0f')}ms |",
            f"| Latency p95 (total) | {overall_metrics.latency_p95_ms:.0f}ms | {bl_p95:.0f}ms | {delta_str(overall_metrics.latency_p95_ms, bl_p95, '.0f')}ms |",
            f"| Latency p99 (total) | {overall_metrics.latency_p99_ms:.0f}ms | {bl_p99:.0f}ms | {delta_str(overall_metrics.latency_p99_ms, bl_p99, '.0f')}ms |",
            f"| Reranking latency p50 | {overall_metrics.reranking_latency_p50_ms:.0f}ms | — | — |",
            f"| Reranking latency p95 | {overall_metrics.reranking_latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | {bl_tokens:.0f} | {delta_str(overall_metrics.avg_tokens_retrieved, bl_tokens, '.0f')} |",
        ])
    else:
        lines.extend([
            f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} | — | — |",
            f"| Hit@5 (labeled) | {overall_metrics.hit_at_5:.1f}% | — | — |",
            f"| Hit@10 (labeled) | {overall_metrics.hit_at_10:.1f}% | — | — |",
            f"| **Answer Success Rate** | **{overall_metrics.answer_success_rate:.1f}%** | — | — |",
            f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 | — | — |",
            f"| Latency p50 (total) | {overall_metrics.latency_p50_ms:.0f}ms | — | — |",
            f"| Latency p95 (total) | {overall_metrics.latency_p95_ms:.0f}ms | — | — |",
            f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} | — | — |",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## Performance by Query Type",
        "",
        "| Type | Count | MRR@10 | Hit@10 | Success Rate | Avg Grade | Baseline SR | Delta |",
        "|------|-------|--------|--------|--------------|-----------|-------------|-------|",
    ])

    sorted_types = sorted(
        by_type_metrics.items(),
        key=lambda x: x[1].answer_success_rate
    )

    for qt, metrics in sorted_types:
        bl_sr = "—"
        delta = "—"
        if bl_by_type and qt in bl_by_type:
            bl_sr_val = bl_by_type[qt].get("answer_success_rate", 0)
            bl_sr = f"{bl_sr_val:.1f}%"
            diff = metrics.answer_success_rate - bl_sr_val
            sign = "+" if diff >= 0 else ""
            delta = f"{sign}{diff:.1f}%"

        lines.append(
            f"| {qt} | {metrics.total_queries} | "
            f"{metrics.mrr_at_10:.3f} | {metrics.hit_at_10:.1f}% | "
            f"{metrics.answer_success_rate:.1f}% | {metrics.avg_grade:.2f} | "
            f"{bl_sr} | {delta} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Success Criteria Evaluation",
        "",
    ])

    # Check success criteria
    if bl_overall:
        bl_asr = bl_overall.get("answer_success_rate", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        bl_p95 = bl_overall.get("latency_p95_ms", 0)
        lat_delta = overall_metrics.latency_p95_ms - bl_p95

        asr_pass = asr_delta >= 10
        lat_pass = lat_delta <= 500
        no_regress = True

        lines.extend([
            f"| Criterion | Threshold | Actual | Pass? |",
            f"|-----------|-----------|--------|-------|",
            f"| Answer Success Rate ≥+10% | +10.0% | {asr_delta:+.1f}% | {'✅' if asr_pass else '❌'} |",
            f"| Latency increase ≤500ms (p95) | ≤500ms | {lat_delta:+.0f}ms | {'✅' if lat_pass else '❌'} |",
        ])

        # Check per-type regression
        if bl_by_type:
            for qt, metrics in by_type_metrics.items():
                if qt in bl_by_type:
                    bl_qt_asr = bl_by_type[qt].get("answer_success_rate", 0)
                    qt_delta = metrics.answer_success_rate - bl_qt_asr
                    if qt_delta < -5:
                        no_regress = False
                        lines.append(f"| No type regresses >5% | >-5% | {qt}: {qt_delta:+.1f}% | ❌ |")

        if no_regress:
            lines.append(f"| No type regresses >5% | >-5% | All OK | ✅ |")

    lines.extend([
        "",
        "---",
        "",
        "## Category Distribution",
        "",
    ])

    for cat, count in sorted(overall_metrics.category_distribution.items(), key=lambda x: -x[1]):
        pct = (count / overall_metrics.total_queries) * 100
        lines.append(f"- {cat}: {count} ({pct:.1f}%)")

    # Failed queries
    failed_queries = [r for r in results if r.judge_failed or (r.judge_grade is not None and r.judge_grade < SUCCESS_THRESHOLD)]
    if failed_queries:
        lines.extend([
            "",
            "## Sample Failed Queries (Top 10)",
            "",
        ])
        for r in failed_queries[:10]:
            grade_str = f"{r.judge_grade}/10" if r.judge_grade else "JUDGE_FAILED"
            bl_grade_str = ""
            lines.append(f"- **{r.query_id}** ({r.query_type}): {grade_str}{bl_grade_str}")
            lines.append(f"  - Query: \"{r.query_text[:80]}...\"")
            if r.judge_reasoning:
                lines.append(f"  - Reasoning: {r.judge_reasoning[:120]}...")
            lines.append("")

    # Decision
    lines.extend([
        "",
        "---",
        "",
        "## Decision",
        "",
    ])

    if bl_overall:
        bl_asr = bl_overall.get("answer_success_rate", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        lat_delta = overall_metrics.latency_p95_ms - bl_overall.get("latency_p95_ms", 0)

        if asr_delta >= 10 and lat_delta <= 500:
            lines.append("**RECOMMEND** — Meets all success criteria.")
        elif asr_delta >= 5:
            lines.append("**NEEDS MODIFICATION** — Shows promise but doesn't meet the ≥10% ASR threshold.")
        elif asr_delta >= 0:
            lines.append("**REJECT** — Marginal improvement doesn't justify the added complexity and latency.")
        else:
            lines.append("**REJECT** — Regression from baseline.")

        lines.append("")
        lines.append(f"Answer Success Rate delta: {asr_delta:+.1f}% (threshold: +10%)")
        lines.append(f"Latency increase (p95): {lat_delta:+.0f}ms (threshold: ≤500ms)")
    else:
        lines.append("**PENDING** — No baseline available for comparison.")

    lines.extend([
        "",
        "---",
        "",
        f"*Generated by run_reranking.py on {datetime.now().isoformat()}*",
    ])

    return "\n".join(lines)


def save_json_results(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    config: dict,
    output_path: Path,
) -> None:
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "approach": "reranking",
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
    parser = argparse.ArgumentParser(
        description="Run reranking benchmark for adaptive retrieval POC"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_RERANKER,
        choices=list(RERANKER_MODELS.keys()),
        help=f"Reranker model: {', '.join(f'{k}={v}' for k, v in RERANKER_MODELS.items())} (default: {DEFAULT_RERANKER})"
    )
    parser.add_argument(
        "--candidates-k", type=int, default=CANDIDATES_K,
        help=f"Number of candidates to retrieve before reranking (default: {CANDIDATES_K})"
    )
    parser.add_argument(
        "--final-k", type=int, default=FINAL_K,
        help=f"Number of results to return after reranking (default: {FINAL_K})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of queries for testing"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output (no progress bar)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-judge evaluation"
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    reranker_model_name = RERANKER_MODELS[args.model]

    print("=" * 80)
    print("PLM RERANKING BENCHMARK")
    print("=" * 80)
    print()

    # Step 1: Load test queries
    print("[1/8] Loading test queries...")
    if not TEST_QUERIES_PATH.exists():
        print(f"ERROR: Test queries not found at {TEST_QUERIES_PATH}")
        sys.exit(1)

    with open(TEST_QUERIES_PATH) as f:
        data = json.load(f)

    queries = data["queries"]

    if args.limit:
        queries = queries[:args.limit]
        print(f"  Limited to {args.limit} queries for testing")

    print(f"  Loaded {len(queries)} queries")

    type_counts: dict[str, int] = {}
    for q in queries:
        qt = q.get("query_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    print("  Query type distribution:")
    for qt, count in sorted(type_counts.items()):
        print(f"    - {qt}: {count}")
    print()

    # Step 2: Load baseline results
    print("[2/8] Loading baseline results for comparison...")
    baseline_data, baseline_per_query = load_baseline_results()
    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        print(f"  Baseline loaded: ASR={bl_asr:.1f}%, MRR={baseline_data['overall'].get('mrr_at_10', 0):.3f}")
    else:
        print("  WARNING: No baseline found — will skip comparison")
    print()

    # Step 3: Initialize retriever
    print("[3/8] Initializing PLM HybridRetriever...")
    print(f"  DB path: {PLM_DB_PATH}")

    if not Path(PLM_DB_PATH).exists():
        print(f"ERROR: PLM database not found at {PLM_DB_PATH}")
        sys.exit(1)

    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    print("  Retriever initialized")
    print()

    # Step 4: Load reranker
    print(f"[4/8] Loading reranker model ({args.model}: {reranker_model_name})...")
    reranker = load_reranker(args.model)
    print()

    # Step 5: Run reranking benchmark
    print(f"[5/8] Running reranking (candidates_k={args.candidates_k}, final_k={args.final_k})...")
    if args.skip_judge:
        print("  NOTE: Skipping LLM-as-judge evaluation")

    results = run_reranking_measurement(
        queries,
        retriever,
        reranker,
        candidates_k=args.candidates_k,
        final_k=args.final_k,
        quiet=args.quiet,
        skip_judge=args.skip_judge,
        baseline_results=baseline_per_query,
    )

    judge_success = sum(1 for r in results if not r.judge_failed)
    print(f"\n  Completed {len(results)} queries")
    if not args.skip_judge:
        print(f"  Judge success rate: {judge_success}/{len(results)} ({100*judge_success/len(results):.1f}%)")
    print()

    # Step 6: Calculate metrics
    print("[6/8] Calculating aggregate metrics...")
    overall_metrics = calculate_aggregate_metrics(results)
    by_type_metrics = calculate_metrics_by_query_type(results)

    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f}")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  Latency p95 (total): {overall_metrics.latency_p95_ms:.0f}ms")
    print(f"  Reranking latency p95: {overall_metrics.reranking_latency_p95_ms:.0f}ms")
    print()

    # Step 7: Generate markdown report
    print("[7/8] Generating markdown report...")
    config = {
        "reranker_model": reranker_model_name,
        "reranker_key": args.model,
        "candidates_k": args.candidates_k,
        "final_k": args.final_k,
        "judge_model": JUDGE_MODEL,
        "skip_judge": args.skip_judge,
    }

    report = generate_markdown_report(results, overall_metrics, by_type_metrics, config, baseline_data)
    report_path = output_dir / "01_reranking.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    print()

    # Step 8: Save JSON results
    print("[8/8] Saving JSON results...")
    json_path = output_dir / "01_reranking.json"
    save_json_results(results, overall_metrics, by_type_metrics, config, json_path)
    print(f"  JSON saved to {json_path}")
    print()

    # Summary
    print("=" * 80)
    print("RERANKING BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f}")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  Latency p95 (total): {overall_metrics.latency_p95_ms:.0f}ms")
    print(f"  Reranking latency p95: {overall_metrics.reranking_latency_p95_ms:.0f}ms")
    print(f"  Judge failures: {overall_metrics.judge_failures}/{overall_metrics.total_queries}")

    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        bl_p95 = baseline_data["overall"].get("latency_p95_ms", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        lat_delta = overall_metrics.latency_p95_ms - bl_p95
        print()
        print("Comparison to Baseline:")
        print(f"  Answer Success Rate: {asr_delta:+.1f}% (target: +10%)")
        print(f"  Latency increase (p95): {lat_delta:+.0f}ms (limit: 500ms)")

    print()
    print(f"See {report_path} for full results.")


if __name__ == "__main__":
    main()
