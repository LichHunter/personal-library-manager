#!/usr/bin/env python3
"""SPLADE benchmark evaluation.

Runs SPLADE retrieval on all benchmark queries and compares to BM25 baselines.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python splade_benchmark.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from splade_encoder import SPLADEEncoder
from splade_index import SPLADEIndex
from baseline_bm25 import (
    load_informed_queries,
    load_needle_queries,
    load_realistic_queries,
    QueryResult,
    BenchmarkResults,
    calculate_ndcg,
)


POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


class SPLADEBenchmark:
    """SPLADE benchmark evaluator."""
    
    def __init__(self, index_path: Path = SPLADE_INDEX_PATH):
        """Initialize SPLADE benchmark.
        
        Args:
            index_path: Path to SPLADE index directory
        """
        print("[SPLADEBenchmark] Loading SPLADE encoder...")
        self.encoder = SPLADEEncoder()
        
        print("[SPLADEBenchmark] Loading SPLADE index...")
        self.index = SPLADEIndex.load(str(index_path), encoder=self.encoder)
        
        self.query_encoding_times: list[float] = []
    
    def evaluate_query(
        self,
        query_id: str,
        query: str,
        ground_truth_doc_id: str,
        k: int = 10,
    ) -> QueryResult:
        """Evaluate a single query with SPLADE.
        
        Args:
            query_id: Unique query identifier
            query: Query text
            ground_truth_doc_id: Expected document ID
            k: Number of results to retrieve
            
        Returns:
            QueryResult with rank and metrics
        """
        start_time = time.time()
        
        encode_start = time.time()
        query_vec = self.encoder.encode(query, return_tokens=False)
        assert isinstance(query_vec, dict)
        encode_time = (time.time() - encode_start) * 1000
        self.query_encoding_times.append(encode_time)
        
        results = self.index.search_with_vec(query_vec, k=k)
        
        total_latency = (time.time() - start_time) * 1000
        
        rank = None
        score = None
        for i, r in enumerate(results):
            if ground_truth_doc_id in r["doc_id"]:
                rank = i + 1
                score = r["score"]
                break
        
        top_k_doc_ids = [r["doc_id"] for r in results]
        
        return QueryResult(
            query_id=query_id,
            query=query,
            ground_truth_doc_id=ground_truth_doc_id,
            rank=rank,
            score=score,
            top_k_doc_ids=top_k_doc_ids,
            latency_ms=total_latency,
        )
    
    def run_benchmark(
        self,
        name: str,
        queries: list[dict],
        k: int = 10,
    ) -> BenchmarkResults:
        """Run benchmark on a query set.
        
        Args:
            name: Benchmark name
            queries: List of query dicts
            k: Number of results to retrieve
            
        Returns:
            BenchmarkResults with aggregate metrics
        """
        results = []
        
        for q in tqdm(queries, desc=f"[{name}]"):
            result = self.evaluate_query(
                q["id"],
                q["query"],
                q["ground_truth_doc_id"],
                k=k,
            )
            results.append(result)
        
        ranks = [r.rank for r in results]
        found_ranks = [r for r in ranks if r is not None]
        
        mrr = sum(1.0 / r for r in found_ranks) / len(ranks) if ranks else 0
        recall_at_k = len(found_ranks) / len(ranks) if ranks else 0
        hit_at_1 = sum(1 for r in found_ranks if r <= 1) / len(ranks) if ranks else 0
        hit_at_5 = sum(1 for r in found_ranks if r <= 5) / len(ranks) if ranks else 0
        hit_at_10 = sum(1 for r in found_ranks if r <= 10) / len(ranks) if ranks else 0
        ndcg_at_10 = calculate_ndcg(ranks, k=10)
        avg_latency = float(np.mean([r.latency_ms for r in results]))
        
        return BenchmarkResults(
            name=name,
            total_queries=len(queries),
            k=k,
            mrr=mrr,
            recall_at_k=recall_at_k,
            hit_at_1=hit_at_1 * 100,
            hit_at_5=hit_at_5 * 100,
            hit_at_10=hit_at_10 * 100,
            ndcg_at_10=ndcg_at_10,
            avg_latency_ms=avg_latency,
            found_count=len(found_ranks),
            not_found_count=len(ranks) - len(found_ranks),
            per_query=results,
        )
    
    def get_query_encoding_stats(self) -> dict:
        """Get query encoding latency statistics."""
        if not self.query_encoding_times:
            return {}
        
        times = np.array(self.query_encoding_times)
        return {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }


def load_bm25_baseline() -> dict:
    """Load BM25 baseline results."""
    baseline_path = ARTIFACTS_DIR / "baseline_bm25.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"BM25 baseline not found: {baseline_path}")
    
    with open(baseline_path) as f:
        return json.load(f)


def compare_results(
    splade_results: BenchmarkResults,
    bm25_results: dict,
    name: str,
) -> dict:
    """Compare SPLADE results to BM25 baseline.
    
    Args:
        splade_results: SPLADE benchmark results
        bm25_results: BM25 baseline dict from JSON
        name: Benchmark name
        
    Returns:
        Comparison dict
    """
    bm25_mrr = bm25_results["mrr"]
    splade_mrr = splade_results.mrr
    
    improvement = splade_mrr - bm25_mrr
    improvement_pct = (improvement / bm25_mrr * 100) if bm25_mrr > 0 else 0
    
    regression_threshold = 0.95
    is_regression = splade_mrr < (bm25_mrr * regression_threshold)
    
    return {
        "name": name,
        "bm25_mrr": bm25_mrr,
        "splade_mrr": splade_mrr,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "is_regression": is_regression,
        "bm25_found": bm25_results["found_count"],
        "splade_found": splade_results.found_count,
    }


def results_to_dict(results: BenchmarkResults) -> dict:
    """Convert BenchmarkResults to serializable dict."""
    return {
        "name": results.name,
        "total_queries": results.total_queries,
        "k": results.k,
        "mrr": results.mrr,
        "recall_at_k": results.recall_at_k,
        "hit_at_1": results.hit_at_1,
        "hit_at_5": results.hit_at_5,
        "hit_at_10": results.hit_at_10,
        "ndcg_at_10": results.ndcg_at_10,
        "avg_latency_ms": results.avg_latency_ms,
        "found_count": results.found_count,
        "not_found_count": results.not_found_count,
        "per_query": [asdict(q) for q in results.per_query],
    }


def main():
    """Run SPLADE benchmark evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPLADE Benchmark")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/splade_only_results.json",
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    if not SPLADE_INDEX_PATH.exists():
        print(f"ERROR: SPLADE index not found at {SPLADE_INDEX_PATH}")
        print("Run splade_index.py first to build the index.")
        return
    
    benchmark = SPLADEBenchmark()
    
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    print("\n" + "="*60)
    print("Running SPLADE-only benchmarks")
    print("="*60)
    
    informed_results = benchmark.run_benchmark("informed", informed, k=args.k)
    needle_results = benchmark.run_benchmark("needle", needle, k=args.k)
    realistic_results = benchmark.run_benchmark("realistic", realistic, k=args.k)
    
    bm25_baseline = load_bm25_baseline()
    
    informed_cmp = compare_results(informed_results, bm25_baseline["informed"], "informed")
    needle_cmp = compare_results(needle_results, bm25_baseline["needle"], "needle")
    realistic_cmp = compare_results(realistic_results, bm25_baseline["realistic"], "realistic")
    
    print("\n" + "="*60)
    print("SPLADE vs BM25 Comparison")
    print("="*60)
    
    for cmp in [informed_cmp, needle_cmp, realistic_cmp]:
        status = "REGRESSION" if cmp["is_regression"] else "OK"
        print(f"\n{cmp['name'].upper()}:")
        print(f"  BM25 MRR:   {cmp['bm25_mrr']:.4f}")
        print(f"  SPLADE MRR: {cmp['splade_mrr']:.4f}")
        print(f"  Change:     {cmp['improvement']:+.4f} ({cmp['improvement_pct']:+.1f}%) [{status}]")
    
    encoding_stats = benchmark.get_query_encoding_stats()
    print("\n" + "="*60)
    print("Query Encoding Latency")
    print("="*60)
    print(f"  Mean: {encoding_stats['mean_ms']:.2f}ms")
    print(f"  P50:  {encoding_stats['p50_ms']:.2f}ms")
    print(f"  P95:  {encoding_stats['p95_ms']:.2f}ms")
    print(f"  P99:  {encoding_stats['p99_ms']:.2f}ms")
    
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "k": args.k,
            "total_chunks": len(benchmark.index.doc_ids),
            "index_size_mb": benchmark.index.stats.index_size_mb if benchmark.index.stats else None,
        },
        "informed": results_to_dict(informed_results),
        "needle": results_to_dict(needle_results),
        "realistic": results_to_dict(realistic_results),
        "comparison": {
            "informed": informed_cmp,
            "needle": needle_cmp,
            "realistic": realistic_cmp,
        },
        "query_encoding_latency": encoding_stats,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    informed_pass = informed_cmp["improvement_pct"] > 10
    needle_ok = not needle_cmp["is_regression"]
    realistic_ok = not realistic_cmp["is_regression"]
    latency_ok = encoding_stats["mean_ms"] < 100
    
    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)
    print(f"  Informed MRR >10% improvement: {'PASS' if informed_pass else 'FAIL'} ({informed_cmp['improvement_pct']:+.1f}%)")
    print(f"  Needle MRR no regression:      {'PASS' if needle_ok else 'FAIL'}")
    print(f"  Realistic MRR no regression:   {'PASS' if realistic_ok else 'FAIL'}")
    print(f"  Query encoding <100ms:         {'PASS' if latency_ok else 'FAIL'} ({encoding_stats['mean_ms']:.1f}ms)")
    
    if informed_pass and needle_ok and realistic_ok and latency_ok:
        print("\nVERDICT: PASS - Recommend SPLADE as BM25 replacement")
    elif informed_cmp["improvement_pct"] > 5:
        print("\nVERDICT: PARTIAL - Some improvement, but not meeting all criteria")
    else:
        print("\nVERDICT: FAIL - SPLADE does not improve over BM25")


if __name__ == "__main__":
    main()
