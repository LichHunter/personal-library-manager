from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from plm.benchmark.metrics import BenchmarkMetrics, QueryResult, get_best_questions, get_worst_questions
from plm.shared.logger import get_logger


def generate_report(
    results: list[QueryResult],
    metrics: BenchmarkMetrics,
    config: dict,
    output_dir: Path | None = None,
) -> str:
    log = get_logger()
    timestamp = datetime.now().isoformat()
    
    worst = get_worst_questions(results, n=15)
    best = get_best_questions(results, n=10)
    
    misses_by_dataset: dict[str, int] = {}
    for r in results:
        if r.rank is None:
            misses_by_dataset[r.dataset] = misses_by_dataset.get(r.dataset, 0) + 1
    
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {timestamp}")
    lines.append(f"Service URL: {config.get('service_url', 'N/A')}")
    lines.append(f"k: {config.get('k', 'N/A')}")
    lines.append(f"use_rewrite: {config.get('use_rewrite', False)}")
    lines.append(f"use_rerank: {config.get('use_rerank', False)}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Total queries: {metrics.total_queries}")
    lines.append("")
    lines.append("Hit Rates:")
    lines.append(f"  Hit@1:  {metrics.hit_at_1:.1%}")
    lines.append(f"  Hit@3:  {metrics.hit_at_3:.1%}")
    lines.append(f"  Hit@5:  {metrics.hit_at_5:.1%}")
    lines.append(f"  Hit@10: {metrics.hit_at_10:.1%}")
    lines.append("")
    lines.append(f"MRR (Mean Reciprocal Rank): {metrics.mrr:.4f}")
    lines.append("")
    lines.append("Latency:")
    lines.append(f"  Mean: {metrics.mean_latency_ms:.1f}ms")
    lines.append(f"  P50:  {metrics.p50_latency_ms:.1f}ms")
    lines.append(f"  P95:  {metrics.p95_latency_ms:.1f}ms")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("PER-DATASET BREAKDOWN")
    lines.append("-" * 80)
    for dataset, stats in sorted(metrics.per_dataset.items()):
        lines.append(f"\n{dataset.upper()}:")
        lines.append(f"  Total: {stats['total']}")
        lines.append(f"  Hit@k: {stats['hit_at_k']:.1%}")
        lines.append(f"  MRR:   {stats['mrr']:.4f}")
        if dataset in misses_by_dataset:
            lines.append(f"  Misses: {misses_by_dataset[dataset]}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("WORST PERFORMING QUESTIONS (not found or low rank)")
    lines.append("-" * 80)
    for i, r in enumerate(worst, start=1):
        rank_str = f"rank={r.rank}" if r.rank else "NOT FOUND"
        lines.append(f"\n{i}. [{r.dataset}] {rank_str}")
        lines.append(f"   ID: {r.question_id}")
        lines.append(f"   Q: {r.question}")
        lines.append(f"   Target: {r.target_doc_id}")
        if r.retrieved_doc_ids:
            lines.append(f"   Got: {r.retrieved_doc_ids[:3]}")
        if r.trace and r.request_id:
            lines.append(f"   Trace [{r.request_id[:8]}...]:")
            for entry in r.trace.entries:
                lines.append(f"      [{entry.stage}] {entry.message}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("BEST PERFORMING QUESTIONS (rank=1)")
    lines.append("-" * 80)
    for i, r in enumerate(best, start=1):
        lines.append(f"\n{i}. [{r.dataset}] rank={r.rank}")
        lines.append(f"   ID: {r.question_id}")
        lines.append(f"   Q: {r.question[:80]}...")
        lines.append(f"   Target: {r.target_doc_id}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / "report.txt"
        report_file.write_text(report)
        log.info(f"Report saved to: {report_file}")
        
        results_file = output_dir / "results.json"
        results_data = {
            "timestamp": timestamp,
            "config": config,
            "metrics": {
                "total_queries": metrics.total_queries,
                "hit_at_1": metrics.hit_at_1,
                "hit_at_3": metrics.hit_at_3,
                "hit_at_5": metrics.hit_at_5,
                "hit_at_10": metrics.hit_at_10,
                "mrr": metrics.mrr,
                "mean_latency_ms": metrics.mean_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "per_dataset": metrics.per_dataset,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "target_doc_id": r.target_doc_id,
                    "dataset": r.dataset,
                    "hit": r.hit,
                    "rank": r.rank,
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "latency_ms": r.latency_ms,
                    "request_id": r.request_id,
                    "trace": [
                        {"stage": e.stage, "message": e.message}
                        for e in r.trace.entries
                    ] if r.trace else None,
                }
                for r in results
            ],
        }
        results_file.write_text(json.dumps(results_data, indent=2))
        log.info(f"Results JSON saved to: {results_file}")
    
    return report
