#!/usr/bin/env python3
"""
Adaptive Classifier Benchmark Script for Adaptive Retrieval POC (TODO 07).

Implements rule-based query classifier that routes queries to different retrieval
strategies based on predicted query type:
  - SIMPLE (factoid, procedural): Standard chunk retrieval (k=10)
  - COMPLEX (explanatory, troubleshooting): Chunk retrieval + cross-encoder reranking
  - MULTI (comparison): Standard chunk retrieval (k=10)

The classifier uses keyword/pattern heuristics to predict query type without
relying on ground truth labels. This tests whether we can selectively apply
expensive reranking only where it helps most (explanatory/troubleshooting
queries where P0 showed the biggest gains).

Usage:
    # Full run (all 229 queries)
    python run_adaptive_classifier.py

    # Test on limited queries
    python run_adaptive_classifier.py --limit 20

    # Skip LLM judge
    python run_adaptive_classifier.py --skip-judge --limit 10

    # Adjust reranking parameters
    python run_adaptive_classifier.py --candidates-k 30 --final-k 10

Outputs:
    - results/03_adaptive_classifier.md (human-readable report)
    - results/03_adaptive_classifier.json (machine-readable results)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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

# Retrieval configuration
BASELINE_K = 10         # k for simple queries
CANDIDATES_K = 50       # k for reranking over-retrieval
FINAL_K = 10            # k for reranking final results

# Classifier complexity levels
COMPLEXITY_SIMPLE = "simple"      # factoid, procedural → chunk only
COMPLEXITY_COMPLEX = "complex"    # explanatory, troubleshooting → rerank
COMPLEXITY_MULTI = "multi"        # comparison → chunk only (already 100%)

QueryType = Literal["factoid", "procedural", "explanatory", "comparison", "troubleshooting"]
JudgeCategory = Literal["Correct", "Partially Correct", "Incorrect", "Cannot Answer"]
ComplexityLevel = Literal["simple", "complex", "multi"]


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

def classify_query(query: str) -> ComplexityLevel:
    """Rule-based query complexity classifier.

    Routes queries to retrieval strategies based on heuristics:
    - SIMPLE: factoid/procedural queries → standard chunk retrieval
    - COMPLEX: explanatory/troubleshooting → reranking for better ranking
    - MULTI: comparison queries → standard chunk retrieval

    Returns:
        Complexity level: 'simple', 'complex', or 'multi'
    """
    q = query.lower().strip()
    words = q.split()

    # ---- COMPARISON patterns (check first — distinctive keywords) ----
    comparison_patterns = [
        r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b', r'\bcomparison\b',
        r'\bdifference\s+between\b', r'\bdiffer(?:s|ent|ence)?\b',
        r'\bhow\s+(?:does|do|is|are)\s+\w+\s+(?:different|compare)',
        r'\bwhat\s+(?:is|are)\s+the\s+difference',
        r'\bwhich\s+(?:is|should|one)\s+(?:better|preferred|recommended)',
        r'\bpros\s+and\s+cons\b',
        r'\btrade-?offs?\b',
        r'\bchoose\s+between\b',
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, q):
            return COMPLEXITY_MULTI

    # ---- EXPLANATORY patterns (complex — benefits from reranking) ----
    explanatory_patterns = [
        r'^what\s+is\s+the\s+purpose\s+of\b',
        r'^what\s+(?:is|are)\s+(?:a\s+)?(?:the\s+)?(?:role|function|purpose|concept|idea|mechanism)\b',
        r'^explain\b', r'^describe\b',
        r'^how\s+does\b', r'^how\s+do\b',
        r'^why\s+(?:does|do|is|are|should|would|can)\b',
        r'^what\s+happens\s+when\b',
        r'^what\s+(?:is|are)\s+the\s+(?:benefit|advantage|disadvantage|implication|consequence)',
        r'\bhow\s+(?:does|do)\s+\w+\s+work\b',
        r'\barchitecture\s+of\b',
        r'\bexplain\s+(?:the|how|why)\b',
        r'\bdescribe\s+(?:the|how|why)\b',
        r'\bwhat\s+is\s+the\s+(?:kubelet|kube-proxy|etcd|api[\s-]?server|scheduler|controller)',
        r'\bwhat\s+(?:is|are)\s+(?:the\s+purpose|the\s+role)\b',
        # "What is X?" pattern for concepts (not specific facts)
        r'^what\s+(?:is|are)\s+(?:a\s+|the\s+)?(?:namespace|pod|service|deployment|statefulset|daemonset|job|cronjob|configmap|secret|ingress|networkpolic|rbac|admission\s+controller|init\s+container|sidecar|operator|crd|custom\s+resource|persistent\s+volume|storage\s+class|node\s+selector|taint|toleration|affinity|finalizer|annotation|label|replica\s*set|pod\s*security)',
    ]
    for pattern in explanatory_patterns:
        if re.search(pattern, q):
            return COMPLEXITY_COMPLEX

    # ---- TROUBLESHOOTING patterns (complex — benefits from reranking) ----
    troubleshooting_patterns = [
        r'\btroubleshoot\b', r'\bdebug\b', r'\bdiagnos\b',
        r'\bfix\b', r'\bresolv\b', r'\bsolv\b',
        r'\bwhat\s*\'?s?\s+wrong\b',
        r'\bnot\s+working\b', r'\bfail(?:s|ed|ing|ure)?\b',
        r'\berror\b', r'\bcrash\b', r'\bcrashl[o]*p\b',
        r'\boomkill\b', r'\boom\b',
        r'\bpending\b.*\bpod\b', r'\bpod\b.*\bpending\b',
        r'\bstuck\b', r'\bhang(?:s|ing)?\b',
        r'\bwhy\s+(?:is|are|does|do)\s+my\b',
        r'\bhow\s+(?:do\s+i|to)\s+(?:troubleshoot|debug|diagnose|fix|resolve)',
        r'\bimage\s*pull\s*back\s*off\b',
        r'\bevict\b',
    ]
    for pattern in troubleshooting_patterns:
        if re.search(pattern, q):
            return COMPLEXITY_COMPLEX

    # ---- PROCEDURAL patterns (simple — already 100% baseline) ----
    procedural_patterns = [
        r'^how\s+(?:do\s+i|to|can\s+i)\b',
        r'^steps?\s+to\b', r'^guide\s+to\b',
        r'^create\b', r'^configure\b', r'^set\s*up\b',
        r'^install\b', r'^deploy\b', r'^upgrade\b',
        r'\bstep[\s-]by[\s-]step\b',
        r'\bhow\s+(?:do\s+i|to|can\s+i)\s+(?:create|configure|set\s*up|install|deploy|upgrade|delete|remove|update|scale|rollback|roll\s+back)',
    ]
    for pattern in procedural_patterns:
        if re.search(pattern, q):
            return COMPLEXITY_SIMPLE

    # ---- FACTOID patterns (simple — already 96.3% baseline) ----
    factoid_patterns = [
        r'^what\s+(?:is|are)\s+the\s+(?:default|maximum|minimum|max|min)\b',
        r'^what\s+(?:command|port|flag|option|parameter|field|annotation|label|version)\b',
        r'^which\s+(?:command|port|flag|option|parameter|field|version)\b',
        r'^what\s+(?:is|are)\s+the\s+(?:name|type|value|format|syntax)\b',
        r'\bdefault\s+(?:value|port|size|limit|timeout)\b',
        r'\bwhat\s+(?:flag|command|option)\b',
        r'\bhow\s+many\b', r'\bhow\s+much\b',
    ]
    for pattern in factoid_patterns:
        if re.search(pattern, q):
            return COMPLEXITY_SIMPLE

    # ---- Default: short queries → simple, longer → complex ----
    if len(words) <= 6:
        return COMPLEXITY_SIMPLE

    return COMPLEXITY_COMPLEX


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

    # Classifier result
    predicted_complexity: ComplexityLevel = "simple"
    strategy_used: str = "chunk"  # "chunk" or "rerank"

    # Retrieval results
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    classification_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_retrieved: int = 0

    # Ground truth
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None
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
    reranking_latency_p50_ms: float = 0.0
    reranking_latency_p95_ms: float = 0.0
    avg_tokens_retrieved: float = 0.0

    # Classifier specific
    rerank_rate: float = 0.0  # % of queries routed to reranking
    classification_accuracy: float = 0.0  # vs ground truth query type


# =============================================================================
# METRICS
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

    reranking_latencies = [r.reranking_latency_ms for r in results if r.reranking_latency_ms > 0]
    if reranking_latencies:
        metrics.reranking_latency_p50_ms, metrics.reranking_latency_p95_ms, _ = \
            calculate_latency_percentiles(reranking_latencies)

    tokens = [r.tokens_retrieved for r in results]
    metrics.avg_tokens_retrieved = sum(tokens) / len(tokens) if tokens else 0.0

    # Classifier specific
    reranked_count = sum(1 for r in results if r.strategy_used == "rerank")
    metrics.rerank_rate = (reranked_count / len(results)) * 100 if results else 0.0

    return metrics


def calculate_metrics_by_query_type(results: list[QueryResult]) -> dict[QueryType, AggregateMetrics]:
    by_type: dict[QueryType, list[QueryResult]] = {}
    for r in results:
        by_type.setdefault(r.query_type, []).append(r)
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

def load_reranker() -> "CrossEncoder":
    """Load cross-encoder reranking model (MiniLM — the practical CPU option)."""
    from sentence_transformers import CrossEncoder

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
) -> tuple[list[dict], float]:
    """Rerank candidates using cross-encoder.

    Returns:
        (reranked_results, reranking_latency_ms)
    """
    if not candidates:
        return ([], 0.0)

    pairs = [[query, c.get("content", "")] for c in candidates]

    start = time.perf_counter()
    scores = reranker.predict(pairs)
    reranking_latency_ms = (time.perf_counter() - start) * 1000

    scored_indices = np.argsort(scores)[::-1][:top_k]
    reranked = [candidates[i] for i in scored_indices]

    return (reranked, reranking_latency_ms)


# =============================================================================
# CLASSIFICATION ACCURACY
# =============================================================================

# Ground-truth mapping: query_type → expected complexity
QUERY_TYPE_TO_COMPLEXITY: dict[str, ComplexityLevel] = {
    "factoid": "simple",
    "procedural": "simple",
    "explanatory": "complex",
    "comparison": "multi",
    "troubleshooting": "complex",
}


def calculate_classification_accuracy(results: list[QueryResult]) -> dict:
    """Calculate classifier accuracy vs ground truth query types."""
    total = len(results)
    correct = 0
    confusion: dict[str, dict[str, int]] = {}

    for r in results:
        expected = QUERY_TYPE_TO_COMPLEXITY.get(r.query_type, "simple")
        predicted = r.predicted_complexity

        if expected == predicted:
            correct += 1

        # Build confusion matrix
        if expected not in confusion:
            confusion[expected] = {}
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

    accuracy = (correct / total) * 100 if total else 0.0

    # Per query type accuracy
    per_type: dict[str, dict] = {}
    for r in results:
        qt = r.query_type
        if qt not in per_type:
            per_type[qt] = {"total": 0, "correct": 0, "predictions": {}}
        per_type[qt]["total"] += 1
        expected = QUERY_TYPE_TO_COMPLEXITY.get(qt, "simple")
        if r.predicted_complexity == expected:
            per_type[qt]["correct"] += 1
        pred = r.predicted_complexity
        per_type[qt]["predictions"][pred] = per_type[qt]["predictions"].get(pred, 0) + 1

    for qt, data in per_type.items():
        data["accuracy"] = (data["correct"] / data["total"]) * 100 if data["total"] else 0.0

    return {
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "confusion_matrix": confusion,
        "per_query_type": per_type,
    }


# =============================================================================
# CORE LOOP
# =============================================================================

def run_adaptive_measurement(
    queries: list[dict],
    retriever: HybridRetriever,
    reranker: "CrossEncoder",
    candidates_k: int = CANDIDATES_K,
    final_k: int = FINAL_K,
    baseline_k: int = BASELINE_K,
    quiet: bool = False,
    skip_judge: bool = False,
) -> list[QueryResult]:
    """Run adaptive classifier measurement on all queries."""
    results: list[QueryResult] = []

    iterator = tqdm(queries, desc="Running adaptive classifier") if not quiet else queries

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

        # Step 1: Classify query
        classify_start = time.perf_counter()
        complexity = classify_query(query_text)
        result.classification_latency_ms = (time.perf_counter() - classify_start) * 1000
        result.predicted_complexity = complexity

        # Step 2: Route to appropriate strategy
        if complexity == COMPLEXITY_COMPLEX:
            # Explanatory/troubleshooting → rerank for better ranking
            result.strategy_used = "rerank"

            retrieval_start = time.perf_counter()
            try:
                candidates = retriever.retrieve(query_text, k=candidates_k, use_rewrite=False)
            except Exception as e:
                logger.error(f"Retrieval failed for {query_id}: {e}")
                candidates = []
            result.retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            chunks, reranking_ms = rerank_results(reranker, query_text, candidates, top_k=final_k)
            result.reranking_latency_ms = reranking_ms

        else:
            # Simple/multi → standard chunk retrieval
            result.strategy_used = "chunk"

            retrieval_start = time.perf_counter()
            try:
                chunks = retriever.retrieve(query_text, k=baseline_k, use_rewrite=False)
            except Exception as e:
                logger.error(f"Retrieval failed for {query_id}: {e}")
                chunks = []
            result.retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        result.total_latency_ms = (
            result.classification_latency_ms
            + result.retrieval_latency_ms
            + result.reranking_latency_ms
        )

        # Extract IDs and count tokens
        for sr in chunks:
            chunk_id = sr.get("chunk_id", "")
            doc_id = sr.get("doc_id", "")
            content = sr.get("content", "")

            result.retrieved_chunk_ids.append(chunk_id)
            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += count_tokens(content)

        # Calculate rank
        if relevant_doc_ids:
            for i, sr in enumerate(chunks, 1):
                doc_id = sr.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break

        # Run LLM judge
        if not skip_judge:
            grade, category, reasoning, judge_latency = judge_answer_quality(
                query_text, expected_answer, chunks, quiet=quiet
            )
            result.judge_grade = grade
            result.judge_category = category
            result.judge_reasoning = reasoning
            result.judge_latency_ms = judge_latency
            result.judge_failed = (grade is None)

        if not quiet and isinstance(iterator, tqdm):
            grade_str = f"{result.judge_grade}/10" if result.judge_grade else "N/A"
            cat_str = result.judge_category or "N/A"
            rank_str = str(result.rank) if result.rank else "-"
            iterator.set_postfix_str(
                f"{query_id} | {complexity} | {result.strategy_used} | Grade: {grade_str} ({cat_str}) | Rank: {rank_str}"
            )

        results.append(result)

    return results


# =============================================================================
# BASELINE LOADING
# =============================================================================

def load_baseline_results() -> tuple[Optional[dict], Optional[dict[str, dict]]]:
    """Load baseline results for comparison."""
    baseline_json = RESULTS_DIR / "00_baseline.json"
    if not baseline_json.exists():
        logger.warning(f"Baseline results not found at {baseline_json}")
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


def load_reranking_results() -> Optional[dict]:
    """Load reranking results for comparison (best P0)."""
    reranking_json = RESULTS_DIR / "01_reranking.json"
    if not reranking_json.exists():
        return None

    with open(reranking_json) as f:
        data = json.load(f)

    return {
        "overall": data.get("overall_metrics"),
        "by_type": data.get("by_query_type", {}),
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    classification_stats: dict,
    config: dict,
    baseline: Optional[dict] = None,
    reranking: Optional[dict] = None,
) -> str:
    """Generate markdown report."""

    bl = baseline["overall"] if baseline else None
    bl_by_type = baseline["by_type"] if baseline else None
    rr = reranking["overall"] if reranking else None

    def delta_str(current: float, ref: Optional[float], fmt: str = ".1f") -> str:
        if ref is None:
            return "—"
        diff = current - ref
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:{fmt}}"

    lines = [
        "# Adaptive Classifier Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {overall_metrics.total_queries}",
        f"**Labeled queries:** {overall_metrics.labeled_queries}",
        f"**Configuration:** candidates_k={config['candidates_k']}, final_k={config['final_k']}, "
        f"baseline_k={config['baseline_k']}, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Approach Summary",
        "",
        "Adaptive query classifier routes queries to different retrieval strategies:",
        "1. **Classify** query complexity using rule-based heuristics (~0ms)",
        "2. **Route**:",
        "   - SIMPLE (factoid/procedural) → Standard chunk retrieval (k=10)",
        "   - COMPLEX (explanatory/troubleshooting) → Chunk retrieval (k=50) + cross-encoder reranking → top-10",
        "   - MULTI (comparison) → Standard chunk retrieval (k=10)",
        f"3. **Reranking model:** cross-encoder/ms-marco-MiniLM-L-6-v2",
        "",
        "**Key insight:** Reranking gave +22.6% on explanatory queries but only +0-3% on others.",
        "Selective reranking saves ~60% of reranking latency by skipping queries that don't benefit.",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Adaptive | Baseline | Δ vs Baseline | Reranking (all) | Δ vs Reranking |",
        "|--------|----------|----------|---------------|-----------------|----------------|",
    ]

    bl_asr = bl.get("answer_success_rate", 0) if bl else None
    bl_mrr = bl.get("mrr_at_10", 0) if bl else None
    bl_p50 = bl.get("latency_p50_ms", 0) if bl else None
    bl_p95 = bl.get("latency_p95_ms", 0) if bl else None
    bl_grade = bl.get("avg_grade", 0) if bl else None
    bl_tokens = bl.get("avg_tokens_retrieved", 0) if bl else None
    bl_hit5 = bl.get("hit_at_5", 0) if bl else None

    rr_asr = rr.get("answer_success_rate", 0) if rr else None
    rr_mrr = rr.get("mrr_at_10", 0) if rr else None
    rr_p95 = rr.get("latency_p95_ms", 0) if rr else None
    rr_grade = rr.get("avg_grade", 0) if rr else None
    rr_tokens = rr.get("avg_tokens_retrieved", 0) if rr else None

    m = overall_metrics
    lines.extend([
        f"| **Answer Success Rate** | **{m.answer_success_rate:.1f}%** | {bl_asr:.1f}% | **{delta_str(m.answer_success_rate, bl_asr)}** | {rr_asr:.1f}% | {delta_str(m.answer_success_rate, rr_asr)} |" if bl and rr else
        f"| **Answer Success Rate** | **{m.answer_success_rate:.1f}%** | — | — | — | — |",
        f"| MRR@10 (labeled) | {m.mrr_at_10:.3f} | {bl_mrr:.3f} | {delta_str(m.mrr_at_10, bl_mrr, '.3f')} | {rr_mrr:.3f} | {delta_str(m.mrr_at_10, rr_mrr, '.3f')} |" if bl and rr else
        f"| MRR@10 (labeled) | {m.mrr_at_10:.3f} | — | — | — | — |",
        f"| Hit@5 (labeled) | {m.hit_at_5:.1f}% | {bl_hit5:.1f}% | {delta_str(m.hit_at_5, bl_hit5)} | — | — |" if bl else
        f"| Hit@5 (labeled) | {m.hit_at_5:.1f}% | — | — | — | — |",
        f"| Avg Grade | {m.avg_grade:.2f}/10 | {bl_grade:.2f}/10 | {delta_str(m.avg_grade, bl_grade, '.2f')} | {rr_grade:.2f}/10 | {delta_str(m.avg_grade, rr_grade, '.2f')} |" if bl and rr else
        f"| Avg Grade | {m.avg_grade:.2f}/10 | — | — | — | — |",
        f"| Latency p50 | {m.latency_p50_ms:.0f}ms | {bl_p50:.0f}ms | {delta_str(m.latency_p50_ms, bl_p50, '.0f')}ms | — | — |" if bl else
        f"| Latency p50 | {m.latency_p50_ms:.0f}ms | — | — | — | — |",
        f"| Latency p95 | {m.latency_p95_ms:.0f}ms | {bl_p95:.0f}ms | {delta_str(m.latency_p95_ms, bl_p95, '.0f')}ms | {rr_p95:.0f}ms | {delta_str(m.latency_p95_ms, rr_p95, '.0f')}ms |" if bl and rr else
        f"| Latency p95 | {m.latency_p95_ms:.0f}ms | — | — | — | — |",
        f"| Avg Tokens | {m.avg_tokens_retrieved:.0f} | {bl_tokens:.0f} | {delta_str(m.avg_tokens_retrieved, bl_tokens, '.0f')} | {rr_tokens:.0f} | {delta_str(m.avg_tokens_retrieved, rr_tokens, '.0f')} |" if bl and rr else
        f"| Avg Tokens | {m.avg_tokens_retrieved:.0f} | — | — | — | — |",
        f"| Rerank Rate | {m.rerank_rate:.1f}% | 0% | — | 100% | — |",
    ])

    if m.reranking_latency_p50_ms > 0:
        lines.append(f"| Reranking latency p50 | {m.reranking_latency_p50_ms:.0f}ms | — | — | — | — |")
        lines.append(f"| Reranking latency p95 | {m.reranking_latency_p95_ms:.0f}ms | — | — | — | — |")

    # Per-type breakdown
    lines.extend([
        "",
        "---",
        "",
        "## Performance by Query Type",
        "",
        "| Type | Count | Strategy | MRR@10 | Hit@5 | Success Rate | Avg Grade | Baseline SR | Δ Baseline | Rerank SR | Δ Rerank |",
        "|------|-------|----------|--------|-------|--------------|-----------|-------------|------------|-----------|----------|",
    ])

    sorted_types = sorted(by_type_metrics.items(), key=lambda x: x[1].answer_success_rate)

    for qt, metrics in sorted_types:
        # Determine dominant strategy for this type
        type_results = [r for r in results if r.query_type == qt]
        rerank_count = sum(1 for r in type_results if r.strategy_used == "rerank")
        strategy = "rerank" if rerank_count > len(type_results) / 2 else "chunk"

        bl_sr = bl_by_type[qt].get("answer_success_rate", 0) if bl_by_type and qt in bl_by_type else None
        rr_sr = reranking["by_type"][qt].get("answer_success_rate", 0) if reranking and qt in reranking.get("by_type", {}) else None

        bl_delta = delta_str(metrics.answer_success_rate, bl_sr) if bl_sr is not None else "—"
        rr_delta = delta_str(metrics.answer_success_rate, rr_sr) if rr_sr is not None else "—"
        bl_sr_str = f"{bl_sr:.1f}%" if bl_sr is not None else "—"
        rr_sr_str = f"{rr_sr:.1f}%" if rr_sr is not None else "—"

        lines.append(
            f"| {qt} | {metrics.total_queries} | {strategy} | "
            f"{metrics.mrr_at_10:.3f} | {metrics.hit_at_5:.1f}% | "
            f"{metrics.answer_success_rate:.1f}% | {metrics.avg_grade:.2f} | "
            f"{bl_sr_str} | {bl_delta} | {rr_sr_str} | {rr_delta} |"
        )

    # Classification accuracy section
    lines.extend([
        "",
        "---",
        "",
        "## Classification Accuracy",
        "",
        f"**Overall accuracy:** {classification_stats['overall_accuracy']:.1f}% "
        f"({classification_stats['correct']}/{classification_stats['total']})",
        "",
        "| Query Type | Expected | Total | Correct | Accuracy | Predictions |",
        "|------------|----------|-------|---------|----------|-------------|",
    ])

    for qt, data in sorted(classification_stats["per_query_type"].items()):
        expected = QUERY_TYPE_TO_COMPLEXITY.get(qt, "simple")
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
        lines.append(
            f"| {qt} | {expected} | {data['total']} | {data['correct']} | "
            f"{data['accuracy']:.1f}% | {preds_str} |"
        )

    # Routing summary
    lines.extend([
        "",
        "### Routing Summary",
        "",
    ])

    strategy_counts: dict[str, int] = {}
    for r in results:
        strategy_counts[r.strategy_used] = strategy_counts.get(r.strategy_used, 0) + 1

    for strat, count in sorted(strategy_counts.items()):
        pct = (count / len(results)) * 100
        lines.append(f"- **{strat}**: {count} queries ({pct:.1f}%)")

    # Success criteria
    lines.extend([
        "",
        "---",
        "",
        "## Success Criteria Evaluation",
        "",
        "| Criterion | Threshold | Actual | Pass? |",
        "|-----------|-----------|--------|-------|",
    ])

    if bl:
        bl_asr_val = bl.get("answer_success_rate", 0)
        asr_delta = m.answer_success_rate - bl_asr_val
        bl_p95_val = bl.get("latency_p95_ms", 0)
        lat_delta = m.latency_p95_ms - bl_p95_val

        asr_pass = asr_delta >= 10
        lat_pass = lat_delta <= 500

        lines.extend([
            f"| Answer Success Rate ≥+10% | +10.0% | {asr_delta:+.1f}% | {'✅' if asr_pass else '❌'} |",
            f"| Latency increase ≤500ms (p95) | ≤500ms | {lat_delta:+.0f}ms | {'✅' if lat_pass else '❌'} |",
        ])

        # Check regressions
        no_regress = True
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

        lines.append(f"| Classification accuracy ≥70% | ≥70% | {classification_stats['overall_accuracy']:.1f}% | {'✅' if classification_stats['overall_accuracy'] >= 70 else '❌'} |")

    # Better than best P0?
    if rr:
        rr_asr_val = rr.get("answer_success_rate", 0)
        lines.extend([
            "",
            f"**Better than best P0 (reranking)?** Adaptive ASR: {m.answer_success_rate:.1f}%, "
            f"Reranking ASR: {rr_asr_val:.1f}% → {delta_str(m.answer_success_rate, rr_asr_val)}",
        ])

    # Category distribution
    lines.extend([
        "",
        "---",
        "",
        "## Category Distribution",
        "",
    ])

    for cat, count in sorted(m.category_distribution.items(), key=lambda x: -x[1]):
        pct = (count / m.total_queries) * 100
        lines.append(f"- {cat}: {count} ({pct:.1f}%)")

    # Failed queries
    failed = [r for r in results if r.judge_failed or (r.judge_grade is not None and r.judge_grade < SUCCESS_THRESHOLD)]
    if failed:
        lines.extend([
            "",
            "## Sample Failed Queries (Top 10)",
            "",
        ])
        for r in failed[:10]:
            grade_str = f"{r.judge_grade}/10" if r.judge_grade else "JUDGE_FAILED"
            lines.append(f"- **{r.query_id}** ({r.query_type}, classified={r.predicted_complexity}, strategy={r.strategy_used}): {grade_str}")
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

    if bl:
        bl_asr_val = bl.get("answer_success_rate", 0)
        asr_delta = m.answer_success_rate - bl_asr_val
        bl_p95_val = bl.get("latency_p95_ms", 0)
        lat_delta = m.latency_p95_ms - bl_p95_val

        if asr_delta >= 10 and lat_delta <= 500:
            lines.append("**RECOMMEND** — Meets all success criteria.")
        elif asr_delta >= 5 and lat_delta <= 500:
            lines.append("**NEEDS MODIFICATION** — Good improvement with acceptable latency, but below +10% ASR threshold.")
        elif asr_delta >= 5:
            lines.append("**NEEDS MODIFICATION** — Shows promise but latency exceeds limit.")
        elif asr_delta >= 0:
            lines.append("**REJECT** — Marginal improvement doesn't justify complexity.")
        else:
            lines.append("**REJECT** — Regression from baseline.")

        lines.append("")
        lines.append(f"Answer Success Rate delta: {asr_delta:+.1f}% (threshold: +10%)")
        lines.append(f"Latency increase (p95): {lat_delta:+.0f}ms (threshold: ≤500ms)")
        lines.append(f"Classification accuracy: {classification_stats['overall_accuracy']:.1f}% (threshold: ≥70%)")
        lines.append(f"Rerank rate: {m.rerank_rate:.1f}% of queries")
    else:
        lines.append("**PENDING** — No baseline available for comparison.")

    lines.extend([
        "",
        "---",
        "",
        f"*Generated by run_adaptive_classifier.py on {datetime.now().isoformat()}*",
    ])

    return "\n".join(lines)


def save_json_results(
    results: list[QueryResult],
    overall_metrics: AggregateMetrics,
    by_type_metrics: dict[QueryType, AggregateMetrics],
    classification_stats: dict,
    config: dict,
    output_path: Path,
) -> None:
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "approach": "adaptive_classifier",
            "config": config,
            "query_counts": {
                "total": overall_metrics.total_queries,
                "labeled": overall_metrics.labeled_queries,
                "judge_failures": overall_metrics.judge_failures,
            },
        },
        "overall_metrics": asdict(overall_metrics),
        "by_query_type": {qt: asdict(m) for qt, m in by_type_metrics.items()},
        "classification_stats": classification_stats,
        "per_query_results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive classifier benchmark for adaptive retrieval POC"
    )
    parser.add_argument(
        "--candidates-k", type=int, default=CANDIDATES_K,
        help=f"Number of candidates for reranking (default: {CANDIDATES_K})"
    )
    parser.add_argument(
        "--final-k", type=int, default=FINAL_K,
        help=f"Number of results after reranking (default: {FINAL_K})"
    )
    parser.add_argument(
        "--baseline-k", type=int, default=BASELINE_K,
        help=f"Number of results for simple queries (default: {BASELINE_K})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of queries for testing"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-judge evaluation"
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PLM ADAPTIVE CLASSIFIER BENCHMARK")
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

    # Step 2: Test classifier on all queries (pre-run)
    print("[2/8] Testing classifier predictions...")
    for q in queries:
        predicted = classify_query(q.get("query", ""))
        q["_predicted_complexity"] = predicted

    complexity_counts: dict[str, int] = {}
    for q in queries:
        c = q["_predicted_complexity"]
        complexity_counts[c] = complexity_counts.get(c, 0) + 1

    print("  Predicted complexity distribution:")
    for c, count in sorted(complexity_counts.items()):
        pct = (count / len(queries)) * 100
        print(f"    - {c}: {count} ({pct:.1f}%)")

    # Quick accuracy check using ground truth
    correct = 0
    for q in queries:
        expected = QUERY_TYPE_TO_COMPLEXITY.get(q.get("query_type", ""), "simple")
        if q["_predicted_complexity"] == expected:
            correct += 1
    accuracy = (correct / len(queries)) * 100
    print(f"  Pre-run classification accuracy: {accuracy:.1f}%")
    print()

    # Step 3: Load baseline and reranking results
    print("[3/8] Loading comparison results...")
    baseline_data, _ = load_baseline_results()
    reranking_data = load_reranking_results()

    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        print(f"  Baseline loaded: ASR={bl_asr:.1f}%")
    if reranking_data:
        rr_asr = reranking_data["overall"].get("answer_success_rate", 0)
        print(f"  Reranking loaded: ASR={rr_asr:.1f}%")
    print()

    # Step 4: Initialize retriever
    print("[4/8] Initializing PLM HybridRetriever...")
    if not Path(PLM_DB_PATH).exists():
        print(f"ERROR: PLM database not found at {PLM_DB_PATH}")
        sys.exit(1)

    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    print("  Retriever initialized")
    print()

    # Step 5: Load reranker
    print("[5/8] Loading reranker model (MiniLM)...")
    reranker = load_reranker()
    print()

    # Step 6: Run benchmark
    print(f"[6/8] Running adaptive classifier benchmark...")
    if args.skip_judge:
        print("  NOTE: Skipping LLM-as-judge")

    results = run_adaptive_measurement(
        queries,
        retriever,
        reranker,
        candidates_k=args.candidates_k,
        final_k=args.final_k,
        baseline_k=args.baseline_k,
        quiet=args.quiet,
        skip_judge=args.skip_judge,
    )

    judge_success = sum(1 for r in results if not r.judge_failed)
    reranked = sum(1 for r in results if r.strategy_used == "rerank")
    print(f"\n  Completed {len(results)} queries")
    print(f"  Reranked: {reranked}/{len(results)} ({100*reranked/len(results):.1f}%)")
    if not args.skip_judge:
        print(f"  Judge success rate: {judge_success}/{len(results)}")
    print()

    # Step 7: Calculate metrics
    print("[7/8] Calculating metrics...")
    overall_metrics = calculate_aggregate_metrics(results)
    by_type_metrics = calculate_metrics_by_query_type(results)
    classification_stats = calculate_classification_accuracy(results)

    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f}")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  Latency p95 (total): {overall_metrics.latency_p95_ms:.0f}ms")
    print(f"  Classification accuracy: {classification_stats['overall_accuracy']:.1f}%")
    print()

    # Step 8: Generate reports
    print("[8/8] Generating reports...")
    config = {
        "candidates_k": args.candidates_k,
        "final_k": args.final_k,
        "baseline_k": args.baseline_k,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "classifier_type": "rule_based",
        "judge_model": JUDGE_MODEL,
    }

    report = generate_markdown_report(
        results, overall_metrics, by_type_metrics, classification_stats,
        config, baseline_data, reranking_data
    )
    report_path = RESULTS_DIR / "03_adaptive_classifier.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    json_path = RESULTS_DIR / "03_adaptive_classifier.json"
    save_json_results(results, overall_metrics, by_type_metrics, classification_stats, config, json_path)
    print(f"  JSON saved to {json_path}")
    print()

    # Summary
    print("=" * 80)
    print("ADAPTIVE CLASSIFIER BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  Answer Success Rate: {overall_metrics.answer_success_rate:.1f}%")
    print(f"  MRR@10 (labeled): {overall_metrics.mrr_at_10:.3f}")
    print(f"  Latency p95: {overall_metrics.latency_p95_ms:.0f}ms")
    print(f"  Classification accuracy: {classification_stats['overall_accuracy']:.1f}%")
    print(f"  Rerank rate: {overall_metrics.rerank_rate:.1f}%")

    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        bl_p95 = baseline_data["overall"].get("latency_p95_ms", 0)
        asr_delta = overall_metrics.answer_success_rate - bl_asr
        lat_delta = overall_metrics.latency_p95_ms - bl_p95
        print()
        print("Comparison to Baseline:")
        print(f"  Answer Success Rate: {asr_delta:+.1f}% (target: +10%)")
        print(f"  Latency increase (p95): {lat_delta:+.0f}ms (limit: 500ms)")

    if reranking_data:
        rr_asr = reranking_data["overall"].get("answer_success_rate", 0)
        rr_p95 = reranking_data["overall"].get("latency_p95_ms", 0)
        print()
        print("Comparison to Reranking (all queries):")
        print(f"  ASR delta: {overall_metrics.answer_success_rate - rr_asr:+.1f}%")
        print(f"  Latency saved: {rr_p95 - overall_metrics.latency_p95_ms:+.0f}ms")

    print()
    print(f"See {report_path} for full results.")


if __name__ == "__main__":
    main()
