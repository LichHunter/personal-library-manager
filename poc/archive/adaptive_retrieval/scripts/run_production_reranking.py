#!/usr/bin/env python3
"""
Production Reranking Benchmark — tests the HybridRetriever.retrieve(use_rerank=True)
integrated into the production search service, compares against baseline, and
computes oracle analysis to identify remaining gaps.

Usage:
    python run_production_reranking.py
    python run_production_reranking.py --limit 20 --skip-judge
    python run_production_reranking.py --skip-oracle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import random
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from tqdm import tqdm

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

POC_ROOT = Path(__file__).parent.parent
TEST_QUERIES_PATH = POC_ROOT / "benchmarks" / "datasets" / "test_queries.json"
RESULTS_DIR = POC_ROOT / "results"

PLM_DB_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db"
PLM_BM25_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data"

JUDGE_MODEL = "haiku"
JUDGE_MAX_RETRIES = 3
SUCCESS_THRESHOLD = 6
FINAL_K = 10
CANDIDATES_K = 50

QueryType = Literal["factoid", "procedural", "explanatory", "comparison", "troubleshooting"]
JudgeCategory = Literal["Correct", "Partially Correct", "Incorrect", "Cannot Answer"]

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


@dataclass
class QueryResult:
    query_id: str
    query_type: QueryType
    query_text: str
    expected_answer: str
    optimal_granularity: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_retrieved: int = 0
    relevant_doc_ids: list[str] = field(default_factory=list)
    rank: Optional[int] = None
    hit_at_5: bool = False
    hit_at_10: bool = False
    judge_grade: Optional[int] = None
    judge_category: Optional[JudgeCategory] = None
    judge_reasoning: str = ""
    judge_latency_ms: float = 0.0
    judge_failed: bool = False

    @property
    def can_answer(self) -> bool:
        return self.judge_grade is not None and self.judge_grade >= SUCCESS_THRESHOLD


def format_chunks_for_judge(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")[:500]
        heading = chunk.get("heading", "")
        lines.append(f"[Chunk {i}] {heading}\n{content}")
    return "\n\n".join(lines)


def judge_answer_quality(query, expected_answer, chunks, quiet=False):
    if not HAS_LLM or call_llm is None:
        return (None, None, "LLM not available", 0.0)
    if not chunks:
        return (1, "Cannot Answer", "No chunks retrieved", 0.0)

    prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        chunks_text=format_chunks_for_judge(chunks),
    )

    start_time = time.perf_counter()
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = call_llm(prompt=prompt, model=JUDGE_MODEL, max_tokens=200, temperature=0)
            latency_ms = (time.perf_counter() - start_time) * 1000
            if not response:
                continue

            response_text = response.strip()
            try:
                import re
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
                category = "Correct" if grade >= 9 else "Partially Correct" if grade >= 6 else "Incorrect" if grade >= 3 else "Cannot Answer"
                return (grade, category, reasoning, latency_ms)
            except (json.JSONDecodeError, ValueError):
                import re
                match = re.search(r'grade["\s:]+(\d+)', response_text, re.IGNORECASE)
                if match:
                    grade = max(1, min(10, int(match.group(1))))
                    cat = "Correct" if grade >= 9 else "Partially Correct" if grade >= 6 else "Incorrect" if grade >= 3 else "Cannot Answer"
                    return (grade, cat, f"Extracted: {response_text[:100]}", latency_ms)
        except Exception as e:
            if attempt < JUDGE_MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))

    return (None, None, "Judge failed after retries", (time.perf_counter() - start_time) * 1000)


def calculate_metrics(results: list[QueryResult]) -> dict:
    labeled = [r for r in results if r.relevant_doc_ids]
    judged = [r for r in results if not r.judge_failed and r.judge_grade is not None]

    ranks = [r.rank for r in labeled]
    mrr = sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks) if ranks else 0.0
    hit5 = sum(1 for r in ranks if r is not None and r <= 5) / len(ranks) * 100 if ranks else 0.0
    hit10 = sum(1 for r in ranks if r is not None and r <= 10) / len(ranks) * 100 if ranks else 0.0

    grades = [r.judge_grade for r in judged if r.judge_grade is not None]
    success = sum(1 for r in judged if r.can_answer)
    asr = (success / len(judged)) * 100 if judged else 0.0
    avg_grade = sum(grades) / len(grades) if grades else 0.0

    latencies = np.array([r.total_latency_ms for r in results])
    rerank_latencies = np.array([r.reranking_latency_ms for r in results])
    tokens = [r.tokens_retrieved for r in results]

    cat_dist = {}
    for r in judged:
        if r.judge_category:
            cat_dist[r.judge_category] = cat_dist.get(r.judge_category, 0) + 1

    return {
        "total_queries": len(results),
        "labeled_queries": len(labeled),
        "judge_failures": sum(1 for r in results if r.judge_failed),
        "mrr_at_10": mrr,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "answer_success_rate": asr,
        "avg_grade": avg_grade,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if len(latencies) else 0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0,
        "latency_p99_ms": float(np.percentile(latencies, 99)) if len(latencies) else 0,
        "reranking_latency_p50_ms": float(np.percentile(rerank_latencies, 50)) if len(rerank_latencies) else 0,
        "reranking_latency_p95_ms": float(np.percentile(rerank_latencies, 95)) if len(rerank_latencies) else 0,
        "avg_tokens_retrieved": sum(tokens) / len(tokens) if tokens else 0,
        "category_distribution": cat_dist,
    }


def metrics_by_type(results: list[QueryResult]) -> dict[str, dict]:
    by_type: dict[str, list[QueryResult]] = {}
    for r in results:
        by_type.setdefault(r.query_type, []).append(r)
    return {qt: calculate_metrics(res) for qt, res in by_type.items()}


def run_benchmark(queries, retriever, skip_judge=False, quiet=False):
    results = []
    iterator = tqdm(queries, desc="Production reranking") if not quiet else queries

    for q in iterator:
        qid = q.get("id", "unknown")
        result = QueryResult(
            query_id=qid,
            query_type=q.get("query_type", "unknown"),
            query_text=q.get("query", ""),
            expected_answer=q.get("expected_answer", ""),
            optimal_granularity=q.get("optimal_granularity", "chunk"),
            relevant_doc_ids=q.get("relevant_doc_ids", []),
        )

        start = time.perf_counter()
        try:
            search_results = retriever.retrieve(
                result.query_text, k=FINAL_K, use_rerank=True, candidates_k=CANDIDATES_K
            )
        except Exception as e:
            logger.error(f"Retrieval failed for {qid}: {e}")
            search_results = []
        result.total_latency_ms = (time.perf_counter() - start) * 1000

        for sr in search_results:
            result.retrieved_chunk_ids.append(sr.get("chunk_id", ""))
            doc_id = sr.get("doc_id", "")
            if doc_id and doc_id not in result.retrieved_doc_ids:
                result.retrieved_doc_ids.append(doc_id)
            result.tokens_retrieved += len(sr.get("content", "").split())

        if result.relevant_doc_ids:
            for i, sr in enumerate(search_results, 1):
                doc_id = sr.get("doc_id", "")
                if any(rel_id in doc_id for rel_id in result.relevant_doc_ids):
                    result.rank = i
                    result.hit_at_5 = (i <= 5)
                    result.hit_at_10 = (i <= 10)
                    break

        if not skip_judge:
            grade, cat, reasoning, jlat = judge_answer_quality(
                result.query_text, result.expected_answer, search_results, quiet=quiet
            )
            result.judge_grade = grade
            result.judge_category = cat
            result.judge_reasoning = reasoning
            result.judge_latency_ms = jlat
            result.judge_failed = (grade is None)

        if not quiet and isinstance(iterator, tqdm):
            grade_str = f"{result.judge_grade}/10" if result.judge_grade else "N/A"
            iterator.set_postfix_str(f"{qid} | Grade: {grade_str} | {result.total_latency_ms:.0f}ms")

        results.append(result)

    return results


def run_oracle(queries, retriever, rerank_results, skip_judge=False, quiet=False):
    """Run oracle: test all granularities per query, find best possible result."""
    oracle_results = []
    improved_queries = []

    iterator = tqdm(queries, desc="Oracle analysis") if not quiet else queries
    rerank_lookup = {r.query_id: r for r in rerank_results}

    for q in iterator:
        qid = q.get("id", "unknown")
        query_text = q.get("query", "")
        expected_answer = q.get("expected_answer", "")
        query_type = q.get("query_type", "unknown")

        rerank_result = rerank_lookup.get(qid)
        rerank_grade = rerank_result.judge_grade if rerank_result else None

        best_grade = rerank_grade or 0
        best_approach = "reranked_chunk"
        approaches_tried = {"reranked_chunk": rerank_grade or 0}

        # Only test other granularities if reranking didn't succeed
        if rerank_grade is None or rerank_grade < SUCCESS_THRESHOLD:
            # Document-level retrieval
            try:
                doc_results = retriever.retrieve_documents(query_text, k=3)
                if doc_results:
                    doc_ids = [d["doc_id"] for d in doc_results[:3]]
                    doc_chunks = []
                    for did in doc_ids:
                        chunks = retriever.storage.get_chunks_by_doc(did)
                        doc_chunks.extend(chunks[:20])

                    if doc_chunks and not skip_judge:
                        grade, _, _, _ = judge_answer_quality(
                            query_text, expected_answer, doc_chunks[:10], quiet=True
                        )
                        approaches_tried["document"] = grade or 0
                        if grade and grade > best_grade:
                            best_grade = grade
                            best_approach = "document"
            except Exception as e:
                logger.warning(f"Document retrieval failed for {qid}: {e}")

            # Heading-level retrieval
            try:
                heading_results = retriever.retrieve_headings(query_text, k=5)
                if heading_results:
                    heading_chunks = []
                    for h in heading_results[:3]:
                        hid = h["heading_id"]
                        chunks = retriever.storage.get_chunks_by_heading(hid)
                        heading_chunks.extend(chunks)

                    if heading_chunks and not skip_judge:
                        grade, _, _, _ = judge_answer_quality(
                            query_text, expected_answer, heading_chunks[:10], quiet=True
                        )
                        approaches_tried["heading"] = grade or 0
                        if grade and grade > best_grade:
                            best_grade = grade
                            best_approach = "heading"
            except Exception as e:
                logger.warning(f"Heading retrieval failed for {qid}: {e}")

        can_answer = best_grade >= SUCCESS_THRESHOLD
        oracle_entry = {
            "query_id": qid,
            "query_type": query_type,
            "rerank_grade": rerank_grade,
            "best_grade": best_grade,
            "best_approach": best_approach,
            "can_answer": can_answer,
            "approaches": approaches_tried,
        }
        oracle_results.append(oracle_entry)

        if rerank_grade is not None and rerank_grade < SUCCESS_THRESHOLD and best_grade >= SUCCESS_THRESHOLD:
            improved_queries.append(oracle_entry)

        if not quiet and isinstance(iterator, tqdm):
            iterator.set_postfix_str(f"{qid} | best={best_grade}/10 via {best_approach}")

    return oracle_results, improved_queries


def generate_report(
    rerank_results, rerank_metrics, rerank_by_type,
    oracle_results, improved_queries, baseline_data
):
    bl = baseline_data.get("overall", {}) if baseline_data else {}
    bl_bt = baseline_data.get("by_type", {}) if baseline_data else {}

    def delta(cur, base_key, fmt=".1f"):
        bv = bl.get(base_key, 0) if bl else 0
        diff = cur - bv
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:{fmt}}"

    lines = [
        "# Production Reranking Benchmark Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries:** {rerank_metrics['total_queries']}",
        f"**Labeled queries:** {rerank_metrics['labeled_queries']}",
        f"**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (integrated in HybridRetriever)",
        f"**Configuration:** candidates_k={CANDIDATES_K}, final_k={FINAL_K}, judge={JUDGE_MODEL}",
        "",
        "---",
        "",
        "## Overall Metrics (Production Reranking vs Baseline)",
        "",
        "| Metric | Production Rerank | Baseline | Delta |",
        "|--------|-------------------|----------|-------|",
        f"| MRR@10 (labeled) | {rerank_metrics['mrr_at_10']:.3f} | {bl.get('mrr_at_10', 0):.3f} | {delta(rerank_metrics['mrr_at_10'], 'mrr_at_10', '.3f')} |",
        f"| Hit@5 (labeled) | {rerank_metrics['hit_at_5']:.1f}% | {bl.get('hit_at_5', 0):.1f}% | {delta(rerank_metrics['hit_at_5'], 'hit_at_5')} |",
        f"| Hit@10 (labeled) | {rerank_metrics['hit_at_10']:.1f}% | {bl.get('hit_at_10', 0):.1f}% | {delta(rerank_metrics['hit_at_10'], 'hit_at_10')} |",
        f"| **Answer Success Rate** | **{rerank_metrics['answer_success_rate']:.1f}%** | **{bl.get('answer_success_rate', 0):.1f}%** | **{delta(rerank_metrics['answer_success_rate'], 'answer_success_rate')}** |",
        f"| Avg Grade | {rerank_metrics['avg_grade']:.2f}/10 | {bl.get('avg_grade', 0):.2f}/10 | {delta(rerank_metrics['avg_grade'], 'avg_grade', '.2f')} |",
        f"| Latency p50 | {rerank_metrics['latency_p50_ms']:.0f}ms | {bl.get('latency_p50_ms', 0):.0f}ms | {delta(rerank_metrics['latency_p50_ms'], 'latency_p50_ms', '.0f')}ms |",
        f"| Latency p95 | {rerank_metrics['latency_p95_ms']:.0f}ms | {bl.get('latency_p95_ms', 0):.0f}ms | {delta(rerank_metrics['latency_p95_ms'], 'latency_p95_ms', '.0f')}ms |",
        f"| Avg Tokens | {rerank_metrics['avg_tokens_retrieved']:.0f} | {bl.get('avg_tokens_retrieved', 0):.0f} | {delta(rerank_metrics['avg_tokens_retrieved'], 'avg_tokens_retrieved', '.0f')} |",
        "",
        "---",
        "",
        "## Performance by Query Type",
        "",
        "| Type | Count | MRR@10 | Hit@10 | ASR | Avg Grade | Baseline ASR | Delta |",
        "|------|-------|--------|--------|-----|-----------|-------------|-------|",
    ]

    for qt in sorted(rerank_by_type.keys()):
        m = rerank_by_type[qt]
        bl_asr = bl_bt.get(qt, {}).get("answer_success_rate", 0)
        d = m["answer_success_rate"] - bl_asr
        sign = "+" if d >= 0 else ""
        lines.append(
            f"| {qt} | {m['total_queries']} | {m['mrr_at_10']:.3f} | "
            f"{m['hit_at_10']:.1f}% | {m['answer_success_rate']:.1f}% | "
            f"{m['avg_grade']:.2f} | {bl_asr:.1f}% | {sign}{d:.1f}% |"
        )

    # Oracle section
    if oracle_results:
        oracle_success = sum(1 for o in oracle_results if o["can_answer"])
        oracle_asr = (oracle_success / len(oracle_results)) * 100
        rerank_asr = rerank_metrics["answer_success_rate"]
        gap = oracle_asr - rerank_asr

        lines.extend([
            "",
            "---",
            "",
            "## Oracle Analysis (Remaining Gaps)",
            "",
            f"**Oracle ASR:** {oracle_asr:.1f}% | **Production Rerank ASR:** {rerank_asr:.1f}% | **Remaining gap:** {gap:.1f}%",
            "",
        ])

        # Oracle by type
        oracle_by_type: dict[str, list] = {}
        for o in oracle_results:
            oracle_by_type.setdefault(o["query_type"], []).append(o)

        lines.extend([
            "### Oracle vs Production Reranking by Query Type",
            "",
            "| Type | Rerank ASR | Oracle ASR | Gap | Queries Improvable |",
            "|------|-----------|------------|-----|--------------------|",
        ])

        for qt in sorted(oracle_by_type.keys()):
            oracles = oracle_by_type[qt]
            o_success = sum(1 for o in oracles if o["can_answer"])
            o_asr = (o_success / len(oracles)) * 100
            r_asr = rerank_by_type.get(qt, {}).get("answer_success_rate", 0)
            g = o_asr - r_asr
            improvable = sum(1 for o in oracles if o["rerank_grade"] is not None and o["rerank_grade"] < SUCCESS_THRESHOLD and o["best_grade"] >= SUCCESS_THRESHOLD)
            lines.append(f"| {qt} | {r_asr:.1f}% | {o_asr:.1f}% | {g:+.1f}% | {improvable} |")

        if improved_queries:
            lines.extend([
                "",
                "### Queries Where Oracle Improves Over Reranking",
                "",
                "| Query ID | Type | Rerank Grade | Oracle Grade | Best Approach |",
                "|----------|------|-------------|-------------|---------------|",
            ])
            for iq in improved_queries:
                lines.append(
                    f"| {iq['query_id']} | {iq['query_type']} | "
                    f"{iq['rerank_grade']}/10 | {iq['best_grade']}/10 | "
                    f"{iq['best_approach']} |"
                )

        # Approach distribution
        approach_counts: dict[str, int] = {}
        for o in oracle_results:
            if o["can_answer"]:
                approach_counts[o["best_approach"]] = approach_counts.get(o["best_approach"], 0) + 1

        lines.extend([
            "",
            "### Optimal Approach Distribution (Oracle)",
            "",
            "| Approach | Count | % |",
            "|----------|-------|---|",
        ])
        total_answerable = sum(approach_counts.values())
        for approach, count in sorted(approach_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_answerable * 100) if total_answerable else 0
            lines.append(f"| {approach} | {count} | {pct:.1f}% |")

        # Unanswerable queries
        unanswerable = [o for o in oracle_results if not o["can_answer"]]
        if unanswerable:
            lines.extend([
                "",
                "### Queries That Fail Across ALL Approaches",
                "",
                "| Query ID | Type | Best Grade | Notes |",
                "|----------|------|-----------|-------|",
            ])
            for u in unanswerable:
                lines.append(f"| {u['query_id']} | {u['query_type']} | {u['best_grade']}/10 | Corpus gap |")

    lines.extend([
        "",
        "---",
        "",
        f"*Generated by run_production_reranking.py on {datetime.now().isoformat()}*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Production reranking benchmark with oracle analysis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PRODUCTION RERANKING BENCHMARK")
    print("=" * 80)
    print()

    # Load test queries
    print("[1/6] Loading test queries...")
    with open(TEST_QUERIES_PATH) as f:
        data = json.load(f)
    queries = data["queries"]
    if args.limit:
        queries = queries[:args.limit]
    print(f"  Loaded {len(queries)} queries")

    type_counts = {}
    for q in queries:
        qt = q.get("query_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    for qt, c in sorted(type_counts.items()):
        print(f"    {qt}: {c}")
    print()

    # Load baseline for comparison
    print("[2/6] Loading baseline results...")
    baseline_data = None
    baseline_json = RESULTS_DIR / "00_baseline.json"
    if baseline_json.exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        baseline_data = {"overall": bl.get("overall_metrics", {}), "by_type": bl.get("by_query_type", {})}
        print(f"  Baseline ASR: {baseline_data['overall'].get('answer_success_rate', 0):.1f}%")
    else:
        print("  WARNING: No baseline found")
    print()

    # Initialize retriever
    print("[3/6] Initializing HybridRetriever...")
    if not Path(PLM_DB_PATH).exists():
        print(f"ERROR: PLM database not found at {PLM_DB_PATH}")
        sys.exit(1)
    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    print("  Retriever initialized (reranker will load on first query)")
    print()

    # Run benchmark
    print(f"[4/6] Running production reranking benchmark (candidates_k={CANDIDATES_K}, final_k={FINAL_K})...")
    rerank_results = run_benchmark(queries, retriever, skip_judge=args.skip_judge, quiet=args.quiet)

    judged = sum(1 for r in rerank_results if not r.judge_failed)
    print(f"\n  Completed {len(rerank_results)} queries, {judged} judged")

    rerank_metrics = calculate_metrics(rerank_results)
    rerank_by_type = metrics_by_type(rerank_results)

    print(f"  ASR: {rerank_metrics['answer_success_rate']:.1f}%")
    print(f"  MRR@10: {rerank_metrics['mrr_at_10']:.3f}")
    print(f"  Latency p95: {rerank_metrics['latency_p95_ms']:.0f}ms")
    print()

    # Oracle analysis
    oracle_results = []
    improved_queries = []
    if not args.skip_oracle and not args.skip_judge:
        print("[5/6] Running oracle analysis (testing alternative granularities for failed queries)...")
        oracle_results, improved_queries = run_oracle(
            queries, retriever, rerank_results, skip_judge=args.skip_judge, quiet=args.quiet
        )
        oracle_success = sum(1 for o in oracle_results if o["can_answer"])
        oracle_asr = (oracle_success / len(oracle_results)) * 100
        print(f"\n  Oracle ASR: {oracle_asr:.1f}%")
        print(f"  Queries improvable beyond reranking: {len(improved_queries)}")
    else:
        print("[5/6] Skipping oracle analysis")
    print()

    # Generate report
    print("[6/6] Generating report...")
    report = generate_report(
        rerank_results, rerank_metrics, rerank_by_type,
        oracle_results, improved_queries, baseline_data
    )
    report_path = RESULTS_DIR / "production_reranking.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # Save JSON
    json_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "candidates_k": CANDIDATES_K,
                "final_k": FINAL_K,
                "judge_model": JUDGE_MODEL,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
        },
        "overall_metrics": rerank_metrics,
        "by_query_type": rerank_by_type,
        "per_query_results": [asdict(r) for r in rerank_results],
        "oracle_results": oracle_results,
        "improved_queries": improved_queries,
    }
    json_path = RESULTS_DIR / "production_reranking.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  JSON: {json_path}")
    print()

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"  ASR: {rerank_metrics['answer_success_rate']:.1f}%")
    print(f"  MRR@10: {rerank_metrics['mrr_at_10']:.3f}")
    print(f"  Latency p95: {rerank_metrics['latency_p95_ms']:.0f}ms")
    if baseline_data:
        bl_asr = baseline_data["overall"].get("answer_success_rate", 0)
        print(f"  vs Baseline: {rerank_metrics['answer_success_rate'] - bl_asr:+.1f}% ASR")
    if oracle_results:
        oracle_asr = sum(1 for o in oracle_results if o["can_answer"]) / len(oracle_results) * 100
        print(f"  Oracle ceiling: {oracle_asr:.1f}%")
        print(f"  Remaining gap: {oracle_asr - rerank_metrics['answer_success_rate']:.1f}%")


if __name__ == "__main__":
    main()
