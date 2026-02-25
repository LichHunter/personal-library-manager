#!/usr/bin/env python3
"""Visualization for SPLADE benchmark results.

Generates comparison charts and saves to artifacts/plots/.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python visualize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_results() -> dict:
    """Load all benchmark results."""
    results = {}
    
    bm25_path = ARTIFACTS_DIR / "baseline_bm25.json"
    if bm25_path.exists():
        with open(bm25_path) as f:
            results["bm25"] = json.load(f)
    
    splade_path = ARTIFACTS_DIR / "splade_only_results.json"
    if splade_path.exists():
        with open(splade_path) as f:
            results["splade"] = json.load(f)
    
    hybrid_path = ARTIFACTS_DIR / "hybrid_splade_results.json"
    if hybrid_path.exists():
        with open(hybrid_path) as f:
            results["hybrid"] = json.load(f)
    
    return results


def plot_mrr_comparison(results: dict):
    """Create MRR comparison bar chart."""
    benchmarks = ["informed", "needle", "realistic"]
    
    data = {}
    for system in ["bm25", "splade", "hybrid"]:
        if system in results:
            data[system] = [results[system][b]["mrr"] for b in benchmarks]
    
    if not data:
        print("No results to plot")
        return
    
    x = np.arange(len(benchmarks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"bm25": "#1f77b4", "splade": "#ff7f0e", "hybrid": "#2ca02c"}
    labels = {"bm25": "BM25", "splade": "SPLADE", "hybrid": "SPLADE+Semantic"}
    
    for i, (system, mrrs) in enumerate(data.items()):
        offset = (i - len(data) / 2 + 0.5) * width
        bars = ax.bar(x + offset, mrrs, width, label=labels.get(system, system), color=colors.get(system))
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("MRR@10")
    ax.set_title("Mean Reciprocal Rank Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in benchmarks])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mrr_comparison.png", dpi=150)
    plt.close()
    print(f"Saved mrr_comparison.png")


def plot_rank_scatter(results: dict):
    """Create BM25 rank vs SPLADE rank scatter plot."""
    if "bm25" not in results or "splade" not in results:
        print("Need both BM25 and SPLADE results for scatter plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    benchmarks = ["informed", "needle", "realistic"]
    
    for ax, benchmark in zip(axes, benchmarks):
        bm25_queries = {q["query_id"]: q for q in results["bm25"][benchmark]["per_query"]}
        splade_queries = {q["query_id"]: q for q in results["splade"][benchmark]["per_query"]}
        
        bm25_ranks = []
        splade_ranks = []
        
        for qid in bm25_queries:
            bm25_rank = bm25_queries[qid]["rank"]
            splade_rank = splade_queries.get(qid, {}).get("rank")
            
            if bm25_rank is not None and splade_rank is not None:
                bm25_ranks.append(bm25_rank)
                splade_ranks.append(splade_rank)
        
        if bm25_ranks:
            ax.scatter(bm25_ranks, splade_ranks, alpha=0.6)
            
            max_rank = max(max(bm25_ranks), max(splade_ranks)) + 1
            ax.plot([0, max_rank], [0, max_rank], "r--", alpha=0.5, label="No change")
            
            ax.set_xlabel("BM25 Rank")
            ax.set_ylabel("SPLADE Rank")
            ax.set_title(f"{benchmark.capitalize()}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{benchmark.capitalize()}")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rank_scatter.png", dpi=150)
    plt.close()
    print(f"Saved rank_scatter.png")


def plot_latency_comparison(results: dict):
    """Create latency comparison bar chart."""
    systems = []
    latencies = []
    
    if "bm25" in results:
        avg_latency = np.mean([
            results["bm25"][b]["avg_latency_ms"]
            for b in ["informed", "needle", "realistic"]
        ])
        systems.append("BM25")
        latencies.append(avg_latency)
    
    if "splade" in results:
        avg_latency = np.mean([
            results["splade"][b]["avg_latency_ms"]
            for b in ["informed", "needle", "realistic"]
        ])
        systems.append("SPLADE")
        latencies.append(avg_latency)
        
        if "query_encoding_latency" in results["splade"]:
            systems.append("SPLADE (encode only)")
            latencies.append(results["splade"]["query_encoding_latency"]["mean_ms"])
    
    if "hybrid" in results:
        avg_latency = np.mean([
            results["hybrid"][b]["avg_latency_ms"]
            for b in ["informed", "needle", "realistic"]
        ])
        systems.append("SPLADE+Semantic")
        latencies.append(avg_latency)
    
    if not systems:
        print("No latency data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(systems, latencies, color=["#1f77b4", "#ff7f0e", "#ffbb78", "#2ca02c"][:len(systems)])
    ax.bar_label(bars, fmt="%.1f ms", fontsize=10)
    
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Average Query Latency Comparison")
    ax.grid(axis="y", alpha=0.3)
    
    ax.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="100ms threshold")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=150)
    plt.close()
    print(f"Saved latency_comparison.png")


def plot_index_size(results: dict):
    """Create index size comparison chart."""
    sizes = {}
    
    if "bm25" in results and "metadata" in results["bm25"]:
        sizes["BM25"] = results["bm25"]["metadata"].get("index_size_mb", 0)
    
    if "splade" in results and "metadata" in results["splade"]:
        sizes["SPLADE"] = results["splade"]["metadata"].get("index_size_mb", 0)
    
    if not sizes or all(v == 0 for v in sizes.values()):
        print("No index size data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    bars = ax.bar(sizes.keys(), sizes.values(), color=["#1f77b4", "#ff7f0e"])
    ax.bar_label(bars, fmt="%.1f MB", fontsize=10)
    
    ax.set_ylabel("Size (MB)")
    ax.set_title("Index Size Comparison")
    ax.grid(axis="y", alpha=0.3)
    
    if sizes.get("BM25", 0) > 0:
        ratio = sizes.get("SPLADE", 0) / sizes["BM25"]
        ax.text(0.95, 0.95, f"SPLADE/BM25 = {ratio:.1f}x",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "index_size.png", dpi=150)
    plt.close()
    print(f"Saved index_size.png")


def plot_improvement_delta(results: dict):
    """Create improvement delta chart (SPLADE - BM25)."""
    if "bm25" not in results or "splade" not in results:
        print("Need both BM25 and SPLADE results for improvement chart")
        return
    
    benchmarks = ["informed", "needle", "realistic"]
    
    deltas = []
    for b in benchmarks:
        bm25_mrr = results["bm25"][b]["mrr"]
        splade_mrr = results["splade"][b]["mrr"]
        delta_pct = (splade_mrr - bm25_mrr) / bm25_mrr * 100 if bm25_mrr > 0 else 0
        deltas.append(delta_pct)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["green" if d > 0 else "red" for d in deltas]
    bars = ax.bar(benchmarks, deltas, color=colors, alpha=0.7)
    ax.bar_label(bars, fmt="%+.1f%%", fontsize=10)
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=10, color="green", linestyle="--", alpha=0.5, label="+10% target (informed)")
    ax.axhline(y=-5, color="red", linestyle="--", alpha=0.5, label="-5% regression threshold")
    
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("MRR Change (%)")
    ax.set_title("SPLADE vs BM25: MRR Improvement")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "improvement_delta.png", dpi=150)
    plt.close()
    print(f"Saved improvement_delta.png")


def main():
    """Generate all visualizations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("No results found. Run benchmarks first.")
        return
    
    print(f"Found results: {list(results.keys())}")
    print()
    
    print("Generating plots...")
    plot_mrr_comparison(results)
    plot_rank_scatter(results)
    plot_latency_comparison(results)
    plot_index_size(results)
    plot_improvement_delta(results)
    
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
