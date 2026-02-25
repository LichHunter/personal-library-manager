#!/usr/bin/env python3
"""Visualization for convex fusion benchmark.

Creates plots comparing convex fusion vs RRF performance.

Usage:
    cd poc/convex_fusion_benchmark
    python visualize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_results() -> dict:
    """Load all results files."""
    results = {}
    
    for name in ["raw_scores", "alpha_sweep_informed", "final_results", "regression_check"]:
        path = ARTIFACTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    
    return results


def plot_alpha_vs_mrr(results: dict) -> None:
    """Plot 1: Alpha vs MRR curve for all normalizations."""
    sweep_data = results.get("alpha_sweep_informed", {})
    sweeps = sweep_data.get("sweeps", [])
    rrf_baseline = sweep_data.get("rrf_baseline", {}).get("mrr", 0)
    
    if not sweeps:
        print("No sweep data found")
        return
    
    # Group by normalization
    by_norm = {}
    for s in sweeps:
        norm = s["normalization"]
        if norm not in by_norm:
            by_norm[norm] = {"alpha": [], "mrr": []}
        by_norm[norm]["alpha"].append(s["alpha"])
        by_norm[norm]["mrr"].append(s["mrr"])
    
    plt.figure(figsize=(10, 6))
    
    colors = {"min_max": "#2196F3", "z_score": "#4CAF50", "rank_percentile": "#FF9800"}
    labels = {"min_max": "Min-Max", "z_score": "Z-Score", "rank_percentile": "Rank-Percentile"}
    
    for norm, data in by_norm.items():
        plt.plot(
            data["alpha"], data["mrr"],
            marker="o", linewidth=2, markersize=6,
            color=colors.get(norm, "gray"),
            label=labels.get(norm, norm)
        )
    
    # Add RRF baseline
    plt.axhline(y=rrf_baseline, color="red", linestyle="--", linewidth=2, label=f"RRF Baseline ({rrf_baseline:.4f})")
    
    # Mark best point
    best = sweep_data.get("best")
    if best:
        plt.scatter(
            [best["alpha"]], [best["mrr"]],
            s=200, c="gold", marker="*", zorder=5,
            edgecolors="black", linewidths=1,
            label=f"Best: α={best['alpha']}, MRR={best['mrr']:.4f}"
        )
    
    plt.xlabel("Alpha (BM25 weight)", fontsize=12)
    plt.ylabel("MRR@10", fontsize=12)
    plt.title("Convex Fusion: Alpha vs MRR (Informed Queries)", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "alpha_vs_mrr.png", dpi=150)
    plt.close()
    print("Saved: plots/alpha_vs_mrr.png")


def plot_normalization_comparison(results: dict) -> None:
    """Plot 2: Bar chart comparing normalizations at optimal alpha."""
    sweep_data = results.get("alpha_sweep_informed", {})
    sweeps = sweep_data.get("sweeps", [])
    rrf_baseline = sweep_data.get("rrf_baseline", {}).get("mrr", 0)
    
    if not sweeps:
        return
    
    # Find best alpha for each normalization
    by_norm = {}
    for s in sweeps:
        norm = s["normalization"]
        if norm not in by_norm or s["mrr"] > by_norm[norm]["mrr"]:
            by_norm[norm] = s
    
    labels = ["RRF", "Min-Max", "Z-Score", "Rank-Pct"]
    mrrs = [rrf_baseline]
    alphas = ["N/A"]
    
    for norm in ["min_max", "z_score", "rank_percentile"]:
        if norm in by_norm:
            mrrs.append(by_norm[norm]["mrr"])
            alphas.append(f"α={by_norm[norm]['alpha']}")
    
    colors = ["#F44336", "#2196F3", "#4CAF50", "#FF9800"]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, mrrs, color=colors, edgecolor="black", linewidth=1)
    
    # Add value labels
    for bar, mrr, alpha in zip(bars, mrrs, alphas):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, height + 0.01,
            f"{mrr:.4f}\n{alpha}",
            ha="center", va="bottom", fontsize=10
        )
    
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Best MRR@10", fontsize=12)
    plt.title("Normalization Strategy Comparison (Informed Queries)", fontsize=14)
    plt.ylim(0, max(mrrs) * 1.15)
    plt.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "normalization_comparison.png", dpi=150)
    plt.close()
    print("Saved: plots/normalization_comparison.png")


def plot_per_query_scatter(results: dict) -> None:
    """Plot 3: Scatter plot of RRF rank vs Convex rank."""
    final = results.get("final_results", {})
    per_query = final.get("per_query", {})
    
    if not per_query:
        print("No per-query data found")
        return
    
    winners = per_query.get("winners", [])
    losers = per_query.get("losers", [])
    
    # Collect all data points
    rrf_ranks = []
    convex_ranks = []
    colors = []
    
    for w in winners:
        if w["rrf_rank"] and w["convex_rank"]:
            rrf_ranks.append(w["rrf_rank"])
            convex_ranks.append(w["convex_rank"])
            colors.append("#4CAF50")  # Green for winners
    
    for l in losers:
        if l["rrf_rank"] and l["convex_rank"]:
            rrf_ranks.append(l["rrf_rank"])
            convex_ranks.append(l["convex_rank"])
            colors.append("#F44336")  # Red for losers
    
    if not rrf_ranks:
        print("No rank data to plot")
        return
    
    plt.figure(figsize=(8, 8))
    
    # Diagonal line (equal performance)
    max_rank = max(max(rrf_ranks), max(convex_ranks)) + 5
    plt.plot([0, max_rank], [0, max_rank], "k--", alpha=0.5, label="Equal performance")
    
    # Scatter points
    plt.scatter(rrf_ranks, convex_ranks, c=colors, s=100, alpha=0.7, edgecolors="black", linewidths=1)
    
    # Shade regions
    plt.fill_between([0, max_rank], [0, max_rank], [0, 0], alpha=0.1, color="green", label="Convex better")
    plt.fill_between([0, max_rank], [0, max_rank], [max_rank, max_rank], alpha=0.1, color="red", label="RRF better")
    
    plt.xlabel("RRF Rank", fontsize=12)
    plt.ylabel("Convex Rank", fontsize=12)
    plt.title("Per-Query Rank Comparison", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_rank)
    plt.ylim(0, max_rank)
    plt.gca().set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_query_scatter.png", dpi=150)
    plt.close()
    print("Saved: plots/per_query_scatter.png")


def plot_score_distributions(results: dict) -> None:
    """Plot 4: Score distribution histograms."""
    raw_scores = results.get("raw_scores", {})
    informed = raw_scores.get("informed", [])
    
    if not informed:
        print("No raw score data found")
        return
    
    # Collect all BM25 and semantic scores
    bm25_all = []
    sem_all = []
    
    for q in informed:
        bm25_all.extend(q["bm25_scores"].values())
        sem_all.extend(q["semantic_scores"].values())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # BM25 histogram
    axes[0].hist(bm25_all, bins=50, color="#2196F3", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("BM25 Score", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(f"BM25 Score Distribution\n(n={len(bm25_all)}, range=[{min(bm25_all):.2f}, {max(bm25_all):.2f}])", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Semantic histogram
    axes[1].hist(sem_all, bins=50, color="#4CAF50", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Semantic Score (Cosine Similarity)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(f"Semantic Score Distribution\n(n={len(sem_all)}, range=[{min(sem_all):.4f}, {max(sem_all):.4f}])", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "score_distributions.png", dpi=150)
    plt.close()
    print("Saved: plots/score_distributions.png")


def plot_cross_benchmark(results: dict) -> None:
    """Plot 5: Cross-benchmark comparison."""
    regression = results.get("regression_check", {})
    final = results.get("final_results", {})
    baselines = final.get("rrf_baselines", {})
    
    if not regression or not baselines:
        print("No regression data found")
        return
    
    benchmarks = ["Informed", "Needle", "Realistic"]
    rrf_mrrs = [
        baselines.get("informed", 0),
        baselines.get("needle", 0),
        baselines.get("realistic", 0),
    ]
    
    informed_sweep = final.get("informed_sweep", {})
    convex_mrrs = [
        informed_sweep.get("best", {}).get("mrr", 0) if informed_sweep.get("best") else baselines.get("informed", 0),
        regression.get("needle", {}).get("convex_mrr", baselines.get("needle", 0)),
        regression.get("realistic", {}).get("convex_mrr", baselines.get("realistic", 0)),
    ]
    
    x = np.arange(len(benchmarks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, rrf_mrrs, width, label="RRF", color="#F44336", edgecolor="black")
    bars2 = ax.bar(x + width/2, convex_mrrs, width, label="Convex (Optimal)", color="#4CAF50", edgecolor="black")
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=10
            )
    
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("MRR@10", fontsize=12)
    ax.set_title("Cross-Benchmark Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(rrf_mrrs), max(convex_mrrs)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_benchmark.png", dpi=150)
    plt.close()
    print("Saved: plots/cross_benchmark.png")


def generate_markdown_tables(results: dict) -> str:
    """Generate markdown tables for RESULTS.md."""
    md = []
    
    # Alpha sweep table
    sweep_data = results.get("alpha_sweep_informed", {})
    sweeps = sweep_data.get("sweeps", [])
    
    if sweeps:
        md.append("## Alpha Sweep Results (Informed Queries)\n")
        md.append("| Normalization | Alpha | MRR@10 | Recall@10 | Hit@1 |\n")
        md.append("|---------------|-------|--------|-----------|-------|\n")
        
        # Group and sort
        sorted_sweeps = sorted(sweeps, key=lambda x: (x["normalization"], x["alpha"]))
        for s in sorted_sweeps:
            md.append(f"| {s['normalization']} | {s['alpha']:.1f} | {s['mrr']:.4f} | {s['recall_at_k']:.4f} | {s['hit_at_1']:.1f}% |\n")
        md.append("\n")
    
    # Best configuration
    best = sweep_data.get("best")
    if best:
        md.append("## Best Configuration\n\n")
        md.append(f"- **Normalization**: {best['normalization']}\n")
        md.append(f"- **Alpha**: {best['alpha']}\n")
        md.append(f"- **MRR**: {best['mrr']:.4f}\n")
        md.append(f"- **Improvement over RRF**: {best['improvement_pct']:+.2f}%\n\n")
    
    # Regression results
    regression = results.get("regression_check", {})
    if regression:
        md.append("## Regression Testing\n\n")
        md.append("| Benchmark | RRF MRR | Convex MRR | Change |\n")
        md.append("|-----------|---------|------------|--------|\n")
        
        for bench in ["needle", "realistic"]:
            if bench in regression:
                r = regression[bench]
                check = r.get("regression_check", {})
                diff_pct = check.get("diff_pct", 0)
                md.append(f"| {bench.capitalize()} | {r['rrf_mrr']:.4f} | {r['convex_mrr']:.4f} | {diff_pct:+.2f}% |\n")
        md.append("\n")
    
    return "".join(md)


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found! Run alpha_sweep.py first.")
        return
    
    # Generate plots
    print("\nGenerating plots...")
    plot_alpha_vs_mrr(results)
    plot_normalization_comparison(results)
    plot_per_query_scatter(results)
    plot_score_distributions(results)
    plot_cross_benchmark(results)
    
    # Generate markdown tables
    print("\nGenerating markdown tables...")
    tables = generate_markdown_tables(results)
    
    tables_path = ARTIFACTS_DIR / "tables.md"
    with open(tables_path, "w") as f:
        f.write(tables)
    print(f"Saved: artifacts/tables.md")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
