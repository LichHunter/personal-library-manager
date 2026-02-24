#!/usr/bin/env python3
"""Configuration comparison for SPLADE encoding variations.

Tests different encoding strategies as specified in TODO 8.2 and 8.3:
- Raw content vs enriched content encoding
- With query expansion vs without query expansion

Usage:
    cd poc/splade_benchmark
    .venv/bin/python configuration_comparison.py
"""

from __future__ import annotations

import json
import re
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


class QueryExpander:
    """Simple query expander that adds related terms."""
    
    # Kubernetes abbreviation expansions
    EXPANSIONS = {
        "k8s": "kubernetes",
        "hpa": "horizontal pod autoscaler",
        "vpa": "vertical pod autoscaler",
        "pvc": "persistent volume claim",
        "pv": "persistent volume",
        "svc": "service",
        "ns": "namespace",
        "cm": "configmap",
        "sa": "service account",
        "rbac": "role based access control",
        "crd": "custom resource definition",
        "cni": "container network interface",
        "csi": "container storage interface",
        "cri": "container runtime interface",
        "etcd": "etcd distributed key value store",
    }
    
    def expand(self, query: str) -> str:
        """Expand abbreviations in query."""
        expanded = query.lower()
        for abbrev, full in self.EXPANSIONS.items():
            # Word boundary match
            pattern = rf'\b{abbrev}\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                expanded = re.sub(pattern, f"{abbrev} ({full})", expanded, flags=re.IGNORECASE)
        return expanded


def evaluate_query(
    encoder: SPLADEEncoder,
    index: SPLADEIndex,
    query_id: str,
    query: str,
    ground_truth_doc_id: str,
    k: int = 10,
) -> QueryResult:
    """Evaluate a single query."""
    start_time = time.time()
    
    query_vec = encoder.encode(query, return_tokens=False)
    assert isinstance(query_vec, dict)
    
    results = index.search_with_vec(query_vec, k=k)
    
    latency_ms = (time.time() - start_time) * 1000
    
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
        latency_ms=latency_ms,
    )


def run_benchmark(
    encoder: SPLADEEncoder,
    index: SPLADEIndex,
    name: str,
    queries: list[dict],
    k: int = 10,
    expand_queries: bool = False,
    expander: Optional[QueryExpander] = None,
) -> BenchmarkResults:
    """Run benchmark on a query set."""
    results = []
    
    for q in tqdm(queries, desc=f"[{name}]", leave=False):
        query_text = q["query"]
        if expand_queries and expander:
            query_text = expander.expand(query_text)
        
        result = evaluate_query(
            encoder,
            index,
            q["id"],
            query_text,
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


def results_to_dict(results: BenchmarkResults) -> dict:
    """Convert BenchmarkResults to serializable dict."""
    return {
        "name": results.name,
        "mrr": results.mrr,
        "recall_at_k": results.recall_at_k,
        "hit_at_1": results.hit_at_1,
        "hit_at_5": results.hit_at_5,
        "hit_at_10": results.hit_at_10,
        "ndcg_at_10": results.ndcg_at_10,
        "avg_latency_ms": results.avg_latency_ms,
        "found_count": results.found_count,
        "not_found_count": results.not_found_count,
    }


def main():
    """Run configuration comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Comparison")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--output", type=str, default="artifacts/configuration_comparison.json")
    
    args = parser.parse_args()
    
    if not SPLADE_INDEX_PATH.exists():
        print(f"ERROR: SPLADE index not found at {SPLADE_INDEX_PATH}")
        return
    
    # Load encoder and index
    print("[ConfigComparison] Loading SPLADE encoder...")
    encoder = SPLADEEncoder()
    
    print("[ConfigComparison] Loading SPLADE index...")
    index = SPLADEIndex.load(str(SPLADE_INDEX_PATH), encoder=encoder)
    
    # Load queries
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    # Initialize expander
    expander = QueryExpander()
    
    all_results = {
        "configurations": {},
        "summary": {},
    }
    
    # Configuration 1: Raw queries (no expansion) - baseline
    print("\n" + "="*60)
    print("Configuration 1: Raw queries (SPLADE expansion only)")
    print("="*60)
    
    raw_informed = run_benchmark(encoder, index, "informed", informed, k=args.k, expand_queries=False)
    raw_needle = run_benchmark(encoder, index, "needle", needle, k=args.k, expand_queries=False)
    raw_realistic = run_benchmark(encoder, index, "realistic", realistic, k=args.k, expand_queries=False)
    
    all_results["configurations"]["raw_queries"] = {
        "description": "No query expansion, SPLADE handles all expansion",
        "informed": results_to_dict(raw_informed),
        "needle": results_to_dict(raw_needle),
        "realistic": results_to_dict(raw_realistic),
    }
    
    print(f"\nResults (raw queries):")
    print(f"  Informed MRR:  {raw_informed.mrr:.4f}")
    print(f"  Needle MRR:    {raw_needle.mrr:.4f}")
    print(f"  Realistic MRR: {raw_realistic.mrr:.4f}")
    
    # Configuration 2: Expanded queries (abbreviation expansion + SPLADE)
    print("\n" + "="*60)
    print("Configuration 2: Pre-expanded queries (abbreviation + SPLADE)")
    print("="*60)
    
    # Show sample expansions
    print("\nSample expansions:")
    for q in informed[:3]:
        original = q["query"]
        expanded = expander.expand(original)
        if original.lower() != expanded:
            print(f"  '{original}' -> '{expanded}'")
    
    exp_informed = run_benchmark(encoder, index, "informed", informed, k=args.k, expand_queries=True, expander=expander)
    exp_needle = run_benchmark(encoder, index, "needle", needle, k=args.k, expand_queries=True, expander=expander)
    exp_realistic = run_benchmark(encoder, index, "realistic", realistic, k=args.k, expand_queries=True, expander=expander)
    
    all_results["configurations"]["expanded_queries"] = {
        "description": "Abbreviation expansion before SPLADE encoding",
        "informed": results_to_dict(exp_informed),
        "needle": results_to_dict(exp_needle),
        "realistic": results_to_dict(exp_realistic),
    }
    
    print(f"\nResults (expanded queries):")
    print(f"  Informed MRR:  {exp_informed.mrr:.4f}")
    print(f"  Needle MRR:    {exp_needle.mrr:.4f}")
    print(f"  Realistic MRR: {exp_realistic.mrr:.4f}")
    
    # Configuration 3: Test different query lengths (truncation effect)
    print("\n" + "="*60)
    print("Configuration 3: Analyzing query length effects")
    print("="*60)
    
    short_queries = [q for q in informed if len(q["query"].split()) <= 8]
    long_queries = [q for q in informed if len(q["query"].split()) > 8]
    
    if short_queries:
        short_results = run_benchmark(encoder, index, "short", short_queries, k=args.k)
        print(f"  Short queries (<=8 words): {len(short_queries)} queries, MRR = {short_results.mrr:.4f}")
        all_results["configurations"]["short_queries"] = {
            "description": "Informed queries with 8 or fewer words",
            "count": len(short_queries),
            "mrr": short_results.mrr,
        }
    
    if long_queries:
        long_results = run_benchmark(encoder, index, "long", long_queries, k=args.k)
        print(f"  Long queries (>8 words): {len(long_queries)} queries, MRR = {long_results.mrr:.4f}")
        all_results["configurations"]["long_queries"] = {
            "description": "Informed queries with more than 8 words",
            "count": len(long_queries),
            "mrr": long_results.mrr,
        }
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("="*60)
    
    raw_avg_mrr = (raw_informed.mrr + raw_needle.mrr + raw_realistic.mrr) / 3
    exp_avg_mrr = (exp_informed.mrr + exp_needle.mrr + exp_realistic.mrr) / 3
    
    all_results["summary"] = {
        "raw_queries_avg_mrr": raw_avg_mrr,
        "expanded_queries_avg_mrr": exp_avg_mrr,
        "best_configuration": "raw_queries" if raw_avg_mrr >= exp_avg_mrr else "expanded_queries",
        "recommendation": "SPLADE expansion only is sufficient" if raw_avg_mrr >= exp_avg_mrr else "Use abbreviation expansion before SPLADE",
    }
    
    print(f"\nRaw queries avg MRR:      {raw_avg_mrr:.4f}")
    print(f"Expanded queries avg MRR: {exp_avg_mrr:.4f}")
    print(f"\nBest configuration: {all_results['summary']['best_configuration']}")
    print(f"Recommendation: {all_results['summary']['recommendation']}")
    
    # Detailed comparison
    print("\n--- Detailed Comparison ---")
    print("| Benchmark | Raw | Expanded | Diff |")
    print("|-----------|-----|----------|------|")
    print(f"| Informed  | {raw_informed.mrr:.4f} | {exp_informed.mrr:.4f} | {exp_informed.mrr - raw_informed.mrr:+.4f} |")
    print(f"| Needle    | {raw_needle.mrr:.4f} | {exp_needle.mrr:.4f} | {exp_needle.mrr - raw_needle.mrr:+.4f} |")
    print(f"| Realistic | {raw_realistic.mrr:.4f} | {exp_realistic.mrr:.4f} | {exp_realistic.mrr - raw_realistic.mrr:+.4f} |")
    
    # Save results
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
