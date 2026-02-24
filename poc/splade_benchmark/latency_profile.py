#!/usr/bin/env python3
"""Latency profiling for SPLADE benchmark.

Profiles query encoding, retrieval, and memory usage with detailed statistics.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python latency_profile.py
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from splade_encoder import SPLADEEncoder
from splade_index import SPLADEIndex
from baseline_bm25 import load_informed_queries, load_needle_queries, load_realistic_queries


POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


def measure_memory_mb() -> float:
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def percentile_stats(values: list[float]) -> dict:
    """Calculate percentile statistics."""
    if not values:
        return {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0}
    
    arr = np.array(values)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
        "samples": len(values),
    }


def profile_cold_start() -> dict:
    """Profile cold start latency (first query after model load)."""
    print("\n[Cold Start Profiling]")
    
    # Force garbage collection
    gc.collect()
    
    # Start memory tracking
    tracemalloc.start()
    baseline_mem = measure_memory_mb()
    
    # Measure model load time
    load_start = time.time()
    encoder = SPLADEEncoder()
    load_time_ms = (time.time() - load_start) * 1000
    
    model_mem = measure_memory_mb()
    print(f"  Model load time: {load_time_ms:.2f}ms")
    print(f"  Memory after load: {model_mem:.2f}MB")
    
    # First query (cold)
    cold_start = time.time()
    _ = encoder.encode("What is Kubernetes?", return_tokens=False)
    cold_time_ms = (time.time() - cold_start) * 1000
    
    cold_mem = measure_memory_mb()
    print(f"  Cold start encoding: {cold_time_ms:.2f}ms")
    print(f"  Memory after cold query: {cold_mem:.2f}MB")
    
    # Warm queries
    warm_times = []
    for _ in range(5):
        start = time.time()
        _ = encoder.encode("What is a Pod?", return_tokens=False)
        warm_times.append((time.time() - start) * 1000)
    
    warm_mem = measure_memory_mb()
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    
    tracemalloc.stop()
    
    return {
        "model_load_ms": load_time_ms,
        "cold_start_ms": cold_time_ms,
        "warm_mean_ms": float(np.mean(warm_times)),
        "memory": {
            "baseline_mb": baseline_mem,
            "after_model_load_mb": model_mem,
            "after_cold_query_mb": cold_mem,
            "after_warm_queries_mb": warm_mem,
            "peak_mb": peak_mem,
        }
    }


def profile_query_encoding(encoder: SPLADEEncoder, queries: list[dict], n_runs: int = 3) -> dict:
    """Profile query encoding latency over multiple runs."""
    print(f"\n[Query Encoding Profiling] {len(queries)} queries x {n_runs} runs")
    
    all_times = []
    
    for run in range(n_runs):
        for q in tqdm(queries, desc=f"Run {run+1}/{n_runs}", leave=False):
            start = time.time()
            _ = encoder.encode(q["query"], return_tokens=False)
            elapsed_ms = (time.time() - start) * 1000
            all_times.append(elapsed_ms)
    
    stats = percentile_stats(all_times)
    print(f"  Mean: {stats['mean_ms']:.2f}ms, P50: {stats['p50_ms']:.2f}ms, P95: {stats['p95_ms']:.2f}ms, P99: {stats['p99_ms']:.2f}ms")
    
    return stats


def profile_retrieval(index: SPLADEIndex, encoder: SPLADEEncoder, queries: list[dict]) -> dict:
    """Profile retrieval latency (index lookup only, excluding encoding)."""
    print(f"\n[Retrieval Profiling] {len(queries)} queries")
    
    # Pre-encode all queries
    encoded_queries = []
    for q in tqdm(queries, desc="Pre-encoding queries"):
        vec = encoder.encode(q["query"], return_tokens=False)
        encoded_queries.append(vec)
    
    # Profile retrieval only
    retrieval_times = []
    for vec in tqdm(encoded_queries, desc="Profiling retrieval"):
        start = time.time()
        _ = index.search_with_vec(vec, k=10)
        elapsed_ms = (time.time() - start) * 1000
        retrieval_times.append(elapsed_ms)
    
    stats = percentile_stats(retrieval_times)
    print(f"  Mean: {stats['mean_ms']:.2f}ms, P50: {stats['p50_ms']:.2f}ms, P95: {stats['p95_ms']:.2f}ms")
    
    return stats


def profile_end_to_end(index: SPLADEIndex, encoder: SPLADEEncoder, queries: list[dict]) -> dict:
    """Profile end-to-end latency (encoding + retrieval)."""
    print(f"\n[End-to-End Profiling] {len(queries)} queries")
    
    e2e_times = []
    for q in tqdm(queries, desc="End-to-end queries"):
        start = time.time()
        vec = encoder.encode(q["query"], return_tokens=False)
        _ = index.search_with_vec(vec, k=10)
        elapsed_ms = (time.time() - start) * 1000
        e2e_times.append(elapsed_ms)
    
    stats = percentile_stats(e2e_times)
    print(f"  Mean: {stats['mean_ms']:.2f}ms, P50: {stats['p50_ms']:.2f}ms, P95: {stats['p95_ms']:.2f}ms")
    
    return stats


def profile_throughput(encoder: SPLADEEncoder, queries: list[dict], duration_sec: float = 10.0) -> dict:
    """Profile throughput (queries per second)."""
    print(f"\n[Throughput Profiling] ~{duration_sec}s test")
    
    query_texts = [q["query"] for q in queries]
    query_idx = 0
    query_count = 0
    
    start_time = time.time()
    while (time.time() - start_time) < duration_sec:
        _ = encoder.encode(query_texts[query_idx], return_tokens=False)
        query_count += 1
        query_idx = (query_idx + 1) % len(query_texts)
    
    elapsed = time.time() - start_time
    qps = query_count / elapsed
    
    print(f"  Queries: {query_count} in {elapsed:.2f}s = {qps:.2f} QPS")
    
    return {
        "queries_processed": query_count,
        "duration_seconds": elapsed,
        "queries_per_second": qps,
    }


def profile_memory_during_retrieval(index: SPLADEIndex, encoder: SPLADEEncoder, queries: list[dict]) -> dict:
    """Profile memory usage during retrieval operations."""
    print(f"\n[Memory Profiling During Retrieval]")
    
    gc.collect()
    tracemalloc.start()
    
    baseline = measure_memory_mb()
    
    # Encode queries
    for q in queries[:10]:
        _ = encoder.encode(q["query"], return_tokens=False)
    
    after_encoding = measure_memory_mb()
    
    # Retrieve
    for q in queries[:10]:
        vec = encoder.encode(q["query"], return_tokens=False)
        _ = index.search_with_vec(vec, k=10)
    
    after_retrieval = measure_memory_mb()
    peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    
    tracemalloc.stop()
    
    print(f"  Baseline: {baseline:.2f}MB")
    print(f"  After encoding: {after_encoding:.2f}MB")
    print(f"  After retrieval: {after_retrieval:.2f}MB")
    print(f"  Peak: {peak:.2f}MB")
    
    return {
        "baseline_mb": baseline,
        "after_encoding_mb": after_encoding,
        "after_retrieval_mb": after_retrieval,
        "peak_mb": peak,
    }


def main():
    """Run full latency profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Latency Profiler")
    parser.add_argument("--output", default="artifacts/latency_profile.json")
    parser.add_argument("--encoding-runs", type=int, default=2, help="Number of encoding profile runs")
    parser.add_argument("--throughput-duration", type=float, default=10.0, help="Throughput test duration")
    
    args = parser.parse_args()
    
    if not SPLADE_INDEX_PATH.exists():
        print(f"ERROR: SPLADE index not found at {SPLADE_INDEX_PATH}")
        return
    
    # Load queries
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    all_queries = informed + needle + realistic
    
    print(f"Loaded {len(all_queries)} total queries")
    
    # Profile cold start (separate encoder instance)
    cold_start_profile = profile_cold_start()
    
    # Load encoder and index for remaining profiles
    print("\n[Loading encoder and index for warm profiling]")
    encoder = SPLADEEncoder()
    index = SPLADEIndex.load(str(SPLADE_INDEX_PATH), encoder=encoder)
    
    # Profile query encoding
    query_encoding_profile = profile_query_encoding(encoder, all_queries, n_runs=args.encoding_runs)
    
    # Profile retrieval
    retrieval_profile = profile_retrieval(index, encoder, all_queries)
    
    # Profile end-to-end
    e2e_profile = profile_end_to_end(index, encoder, all_queries)
    
    # Profile throughput
    throughput_profile = profile_throughput(encoder, all_queries, duration_sec=args.throughput_duration)
    
    # Profile memory during retrieval
    memory_profile = profile_memory_during_retrieval(index, encoder, all_queries)
    
    # Compile results
    results = {
        "cold_start": cold_start_profile,
        "query_encoding": query_encoding_profile,
        "retrieval": retrieval_profile,
        "end_to_end": e2e_profile,
        "throughput": throughput_profile,
        "memory": {
            "model_load_mb": cold_start_profile["memory"]["after_model_load_mb"],
            "encoding_peak_mb": memory_profile["peak_mb"],
            "retrieval_peak_mb": memory_profile["peak_mb"],
        },
        "summary": {
            "cold_start_ms": cold_start_profile["cold_start_ms"],
            "warm_encoding_mean_ms": query_encoding_profile["mean_ms"],
            "warm_encoding_p95_ms": query_encoding_profile["p95_ms"],
            "warm_encoding_p99_ms": query_encoding_profile["p99_ms"],
            "retrieval_mean_ms": retrieval_profile["mean_ms"],
            "e2e_mean_ms": e2e_profile["mean_ms"],
            "e2e_p95_ms": e2e_profile["p95_ms"],
            "queries_per_second": throughput_profile["queries_per_second"],
            "peak_memory_mb": memory_profile["peak_mb"],
        }
    }
    
    # Save results
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("LATENCY PROFILE SUMMARY")
    print("="*60)
    print(f"  Cold start:         {results['summary']['cold_start_ms']:.2f}ms")
    print(f"  Encoding (mean):    {results['summary']['warm_encoding_mean_ms']:.2f}ms")
    print(f"  Encoding (P95):     {results['summary']['warm_encoding_p95_ms']:.2f}ms")
    print(f"  Encoding (P99):     {results['summary']['warm_encoding_p99_ms']:.2f}ms")
    print(f"  Retrieval (mean):   {results['summary']['retrieval_mean_ms']:.2f}ms")
    print(f"  E2E (mean):         {results['summary']['e2e_mean_ms']:.2f}ms")
    print(f"  E2E (P95):          {results['summary']['e2e_p95_ms']:.2f}ms")
    print(f"  Throughput:         {results['summary']['queries_per_second']:.2f} QPS")
    print(f"  Peak Memory:        {results['summary']['peak_memory_mb']:.2f}MB")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
