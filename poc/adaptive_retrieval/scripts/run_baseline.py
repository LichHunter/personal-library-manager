#!/usr/bin/env python3
"""
Baseline Measurement Script for Adaptive Retrieval POC.

Runs all 229 test queries against PLM HybridRetriever and measures:
- Retrieval quality (MRR@10, Hit@k, Recall@10)
- Context sufficiency (LLM-as-judge Answer Success Rate)
- Performance (latency p50/p95/p99, tokens retrieved)
- Per-query-type breakdown (factoid, procedural, explanatory, comparison, troubleshooting)

Usage:
    # Full run (229 queries)
    python run_baseline.py
    
    # Test on limited queries
    python run_baseline.py --limit 20
    
    # Quiet mode (minimal output)
    python run_baseline.py --quiet
    
Outputs:
    - results/00_baseline.md (human-readable report)
    - results/00_baseline.json (machine-readable results)
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
JUDGE_TIMEOUT = 30  # seconds
JUDGE_MAX_RETRIES = 3

# Judge thresholds
SUCCESS_THRESHOLD = 6  # Grade >= 6 is "can answer"

# Query types
QueryType = Literal["factoid", "procedural", "explanatory", "comparison", "troubleshooting"]
JudgeCategory = Literal["Correct", "Partially Correct", "Incorrect", "Cannot Answer"]

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    # Query info
    query_id: str
    query_type: QueryType
    query_text: str
    expected_answer: str
    optimal_granularity: str
    
    # Retrieval results
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    tokens_retrieved: int = 0
    
    # Ground truth (if available)
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None  # Rank of first relevant doc (1-indexed)
    hit_at_5: bool = False
    hit_at_10: bool = False
    
    # LLM judge results
    judge_grade: Optional[int] = None  # 1-10
    judge_category: Optional[JudgeCategory] = None
    judge_reasoning: str = ""
    judge_latency_ms: float = 0.0
    judge_failed: bool = False
    
    @property
    def can_answer(self) -> bool:
        """Whether the retrieved context can answer the query (grade >= 6)."""
        return self.judge_grade is not None and self.judge_grade >= SUCCESS_THRESHOLD


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a set of query results."""
    # Counts
    total_queries: int = 0
    labeled_queries: int = 0
    judge_failures: int = 0
    
    # Retrieval quality (calculated on labeled queries only)
    mrr_at_10: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    recall_at_10: float = 0.0
    
    # Context sufficiency (calculated on all successfully judged queries)
    answer_success_rate: float = 0.0
    avg_grade: float = 0.0
    grade_distribution: dict[int, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    
    # Performance
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_mrr(ranks: list[Optional[int]]) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not ranks:
        return 0.0
    reciprocals = []
    for rank in ranks:
        if rank is not None:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals)


def calculate_hit_at_k(ranks: list[Optional[int]], k: int) -> float:
    """Calculate Hit@k (percentage of queries where target found in top k).
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        k: Number of top results to check
        
    Returns:
        Hit@k as percentage (0-100)
    """
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return (hits / len(ranks)) * 100


def calculate_latency_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    """Calculate latency percentiles.
    
    Returns:
        (p50, p95, p99) in milliseconds
    """
    if not latencies:
        return (0.0, 0.0, 0.0)
    arr = np.array(latencies)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def count_tokens(text: str) -> int:
    """Simple word-based token approximation."""
    return len(text.split())


def grade_to_category(grade: int) -> JudgeCategory:
    """Map 1-10 grade to 4-point category."""
    if grade >= 9:
        return "Correct"
    elif grade >= 6:
        return "Partially Correct"
    elif grade >= 3:
        return "Incorrect"
    else:
        return "Cannot Answer"


def calculate_aggregate_metrics(results: list[QueryResult]) -> AggregateMetrics:
    """Calculate aggregate metrics from query results."""
    metrics = AggregateMetrics()
    metrics.total_queries = len(results)
    
    # Filter for labeled queries (have relevant_doc_ids)
    labeled_results = [r for r in results if r.relevant_doc_ids]
    metrics.labeled_queries = len(labeled_results)
    
    # Filter for successfully judged queries
    judged_results = [r for r in results if not r.judge_failed and r.judge_grade is not None]
    metrics.judge_failures = sum(1 for r in results if r.judge_failed)
    
    # Retrieval quality metrics (labeled only)
    if labeled_results:
        ranks = [r.rank for r in labeled_results]
        metrics.mrr_at_10 = calculate_mrr(ranks)
        metrics.hit_at_5 = calculate_hit_at_k(ranks, 5)
        metrics.hit_at_10 = calculate_hit_at_k(ranks, 10)
        # Recall@10 = same as Hit@10 for single-doc ground truth
        metrics.recall_at_10 = metrics.hit_at_10
    
    # Context sufficiency metrics (all judged)
    if judged_results:
        grades = [r.judge_grade for r in judged_results if r.judge_grade is not None]
        success_count = sum(1 for r in judged_results if r.can_answer)
        
        metrics.answer_success_rate = (success_count / len(judged_results)) * 100
        metrics.avg_grade = sum(grades) / len(grades) if grades else 0.0
        
        # Grade distribution
        for grade in grades:
            metrics.grade_distribution[grade] = metrics.grade_distribution.get(grade, 0) + 1
        
        # Category distribution
        for r in judged_results:
            if r.judge_category:
                metrics.category_distribution[r.judge_category] = \
                    metrics.category_distribution.get(r.judge_category, 0) + 1
    
    # Performance metrics
    retrieval_latencies = [r.retrieval_latency_ms for r in results]
    metrics.latency_p50_ms, metrics.latency_p95_ms, metrics.latency_p99_ms = \
        calculate_latency_percentiles(retrieval_latencies)
    
    tokens = [r.tokens_retrieved for r in results]
    metrics.avg_tokens_retrieved = sum(tokens) / len(tokens) if tokens else 0.0
    
    return metrics


def calculate_metrics_by_query_type(results: list[QueryResult]) -> dict[QueryType, AggregateMetrics]:
    """Calculate metrics grouped by query type."""
    by_type: dict[QueryType, list[QueryResult]] = {}
    
    for r in results:
        qt = r.query_type
        if qt not in by_type:
            by_type[qt] = []
        by_type[qt].append(r)
    
    return {qt: calculate_aggregate_metrics(res) for qt, res in by_type.items()}


# =============================================================================
# LLM-AS-JUDGE
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
    """Format retrieved chunks for the judge prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")[:500]  # Truncate for prompt
        heading = chunk.get("heading", "")
        lines.append(f"[Chunk {i}] {heading}\n{content}")
    return "\n\n".join(lines)


def judge_answer_quality(
    query: str,
    expected_answer: str,
    chunks: list[dict],
    quiet: bool = False,
) -> tuple[Optional[int], JudgeCategory | None, str, float]:
    """Judge whether retrieved chunks can answer the query.
    
    Args:
        query: The user's question
        expected_answer: Expected answer for reference
        chunks: Retrieved chunks from PLM
        quiet: Suppress verbose logging
        
    Returns:
        (grade, category, reasoning, latency_ms)
    """
    if not HAS_LLM or call_llm is None:
        # No LLM available - return placeholder
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
            
            # Parse JSON response
            response_text = response.strip()
            try:
                if response_text.startswith("{"):
                    data = json.loads(response_text)
                else:
                    # Try to extract JSON from text
                    import re
                    match = re.search(r'\{[^}]+\}', response_text)
                    if match:
                        data = json.loads(match.group())
                    else:
                        raise ValueError("No JSON found in response")
                
                grade = int(data.get("grade", 5))
                grade = max(1, min(10, grade))  # Clamp to 1-10
                reasoning = data.get("reasoning", "")
                category = grade_to_category(grade)
                
                return (grade, category, reasoning, latency_ms)
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: try to extract grade from text
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
            
            # Exponential backoff
            if attempt < JUDGE_MAX_RETRIES - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
    
    # All retries failed
    latency_ms = (time.perf_counter() - start_time) * 1000
    return (None, None, "Judge failed after retries", latency_ms)


# =============================================================================
# CORE RETRIEVAL LOOP
# =============================================================================

def run_baseline_measurement(
    queries: list[dict],
    retriever: HybridRetriever,
    k: int = 10,
    quiet: bool = False,
    skip_judge: bool = False,
) -> list[QueryResult]:
    """Run baseline measurement on all queries.
    
    Args:
        queries: List of query dicts from test_queries.json
        retriever: PLM HybridRetriever instance
        k: Number of results to retrieve
        quiet: Suppress verbose output
        skip_judge: Skip LLM-as-judge (for testing)
        
    Returns:
        List of QueryResult objects
    """
    results: list[QueryResult] = []
    
    iterator = tqdm(queries, desc="Running baseline") if not quiet else queries
    
    for q in iterator:
        query_id = q.get("id", "unknown")
        query_type = q.get("query_type", "unknown")
        query_text = q.get("query", "")
        expected_answer = q.get("expected_answer", "")
        optimal_granularity = q.get("optimal_granularity", "chunk")
        relevant_doc_ids = q.get("relevant_doc_ids", [])
        
        # Create result object
        result = QueryResult(
            query_id=query_id,
            query_type=query_type,
            query_text=query_text,
            expected_answer=expected_answer,
            optimal_granularity=optimal_granularity,
            relevant_doc_ids=relevant_doc_ids,
        )
        
        # Run retrieval
        start_time = time.perf_counter()
        try:
            search_results = retriever.retrieve(query_text, k=k, use_rewrite=False)
        except Exception as e:
            logger.error(f"Retrieval failed for {query_id}: {e}")
            search_results = []
        result.retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract IDs and count tokens
        for sr in search_results:
            chunk_id = sr.get("chunk_id", "")
            doc_id = sr.get("doc_id", "")
            content = sr.get("content", "")
            
            result.retrieved_chunk_ids.append(chunk_id)
            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += count_tokens(content)
        
        # Calculate rank (if ground truth available)
        if relevant_doc_ids:
            for i, sr in enumerate(search_results, 1):
                doc_id = sr.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break
        
        # Run LLM judge
        if not skip_judge:
            grade, category, reasoning, judge_latency = judge_answer_quality(
                query_text, expected_answer, search_results, quiet=quiet
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
                f"{query_id} | Grade: {grade_str} ({cat_str}) | Rank: {rank_str}"
            )
        
        results.append(result)
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    config: dict,
) -> str:
    """Generate markdown report."""
    
    lines = [
        "# Baseline Measurement Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall_metrics.total_queries}",
        f"**Labeled queries:** {overall_metrics.labeled_queries}",
        f"**Configuration:** k={config['k']}, use_rewrite=False, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MRR@10 (labeled) | {overall_metrics.mrr_at_10:.3f} |",
        f"| Hit@5 (labeled) | {overall_metrics.hit_at_5:.1f}% |",
        f"| Hit@10 (labeled) | {overall_metrics.hit_at_10:.1f}% |",
        f"| Answer Success Rate | {overall_metrics.answer_success_rate:.1f}% |",
        f"| Avg Grade | {overall_metrics.avg_grade:.2f}/10 |",
        f"| Judge Failures | {overall_metrics.judge_failures}/{overall_metrics.total_queries} |",
        f"| Latency p50 | {overall_metrics.latency_p50_ms:.0f}ms |",
        f"| Latency p95 | {overall_metrics.latency_p95_ms:.0f}ms |",
        f"| Latency p99 | {overall_metrics.latency_p99_ms:.0f}ms |",
        f"| Avg Tokens Retrieved | {overall_metrics.avg_tokens_retrieved:.0f} |",
        "",
        "---",
        "",
        "## Performance by Query Type",
        "",
        "| Type | Count | MRR@10 | Hit@10 | Success Rate | Avg Grade |",
        "|------|-------|--------|--------|--------------|-----------|",
    ]
    
    # Sort by success rate (lowest first to highlight weak points)
    sorted_types = sorted(
        by_type_metrics.items(),
        key=lambda x: x[1].answer_success_rate
    )
    
    for qt, metrics in sorted_types:
        lines.append(
            f"| {qt} | {metrics.total_queries} | "
            f"{metrics.mrr_at_10:.3f} | {metrics.hit_at_10:.1f}% | "
            f"{metrics.answer_success_rate:.1f}% | {metrics.avg_grade:.2f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Weak Points Identified",
        "",
        "### Lowest Success Rates:",
        "",
    ])
    
    # Top 3 lowest success rates
    for qt, metrics in sorted_types[:3]:
        lines.append(f"- **{qt}**: {metrics.answer_success_rate:.1f}% success rate")
    
    lines.extend([
        "",
        "### Category Distribution:",
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
            "### Sample Failed Queries (Top 10):",
            "",
        ])
        for r in failed_queries[:10]:
            grade_str = f"{r.judge_grade}/10" if r.judge_grade else "JUDGE_FAILED"
            lines.append(f"- **{r.query_id}** ({r.query_type}): {grade_str}")
            lines.append(f"  - Query: \"{r.query_text[:80]}...\"")
            if r.judge_reasoning:
                lines.append(f"  - Reasoning: {r.judge_reasoning[:100]}...")
            lines.append("")
    
    lines.extend([
        "",
        "---",
        "",
        "## Latency Distribution",
        "",
        "| Percentile | Retrieval (ms) |",
        "|------------|----------------|",
        f"| p50 | {overall_metrics.latency_p50_ms:.0f} |",
        f"| p95 | {overall_metrics.latency_p95_ms:.0f} |",
        f"| p99 | {overall_metrics.latency_p99_ms:.0f} |",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Test P0 approaches: reranking, parent-child, auto-merging",
        f"2. Focus on improving **{sorted_types[0][0]}** queries (lowest success rate)",
        "3. Investigate common failure patterns in judge reasoning",
        "",
        "---",
        "",
        f"*Generated by run_baseline.py on {datetime.now().isoformat()}*",
    ])
    
    return "\n".join(lines)


def save_json_results(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    config: dict,
    output_path: Path,
) -> None:
    """Save detailed results to JSON."""
    
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
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
        description="Run baseline measurement for adaptive retrieval POC"
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
        "--k", type=int, default=10,
        help="Top-k retrieval (default: 10)"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-judge evaluation (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PLM BASELINE MEASUREMENT")
    print("=" * 80)
    print()
    
    # Step 1: Load test queries
    print("[1/6] Loading test queries...")
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
    
    # Show distribution
    type_counts: dict[str, int] = {}
    for q in queries:
        qt = q.get("query_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    
    print("  Query type distribution:")
    for qt, count in sorted(type_counts.items()):
        print(f"    - {qt}: {count}")
    print()
    
    # Step 2: Initialize retriever
    print("[2/6] Initializing PLM HybridRetriever...")
    print(f"  DB path: {PLM_DB_PATH}")
    print(f"  BM25 path: {PLM_BM25_PATH}")
    
    if not Path(PLM_DB_PATH).exists():
        print(f"ERROR: PLM database not found at {PLM_DB_PATH}")
        sys.exit(1)
    
    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    print("  Retriever initialized")
    print()
    
    # Step 3: Run baseline retrieval
    print(f"[3/6] Running baseline retrieval (k={args.k}, use_rewrite=False)...")
    if args.skip_judge:
        print("  NOTE: Skipping LLM-as-judge evaluation")
    
    results = run_baseline_measurement(
        queries, retriever, k=args.k, quiet=args.quiet, skip_judge=args.skip_judge
    )
    
    judge_success = sum(1 for r in results if not r.judge_failed)
    print(f"\n  Completed {len(results)} queries")
    if not args.skip_judge:
        print(f"  Judge success rate: {judge_success}/{len(results)} ({100*judge_success/len(results):.1f}%)")
    print()
    
    # Step 4: Calculate metrics
    print("[4/6] Calculating aggregate metrics...")
    overall_metrics = calculate_aggregate_metrics(results)
    by_type_metrics = calculate_metrics_by_query_type(results)
    
    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f} ({overall_metrics.labeled_queries} queries)")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  Latency p95: {overall_metrics.latency_p95_ms:.0f}ms")
    print()
    
    # Step 5: Generate markdown report
    print("[5/6] Generating markdown report...")
    config = {
        "k": args.k,
        "use_rewrite": False,
        "judge_model": JUDGE_MODEL,
        "skip_judge": args.skip_judge,
    }
    
    report = generate_markdown_report(results, overall_metrics, by_type_metrics, config)
    report_path = output_dir / "00_baseline.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    print()
    
    # Step 6: Save JSON results
    print("[6/6] Saving JSON results...")
    json_path = output_dir / "00_baseline.json"
    save_json_results(results, overall_metrics, by_type_metrics, config, json_path)
    print(f"  JSON saved to {json_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("BASELINE MEASUREMENT COMPLETE")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f}")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  Latency p95: {overall_metrics.latency_p95_ms:.0f}ms")
    print(f"  Judge failures: {overall_metrics.judge_failures}/{overall_metrics.total_queries}")
    print()
    print(f"See {report_path} for full results.")


if __name__ == "__main__":
    main()
