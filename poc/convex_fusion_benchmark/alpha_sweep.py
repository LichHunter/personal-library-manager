#!/usr/bin/env python3
"""Alpha sweep for convex fusion benchmark.

Runs comprehensive comparison of convex fusion vs RRF across:
- 11 alpha values (0.0 to 1.0)
- 3 normalization strategies (min-max, z-score, rank-percentile)
- 3 benchmark query sets (informed, needle, realistic)

Usage:
    cd poc/convex_fusion_benchmark
    python alpha_sweep.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from fusion import (
    rrf_fusion,
    convex_fusion,
    get_ranking_from_scores,
    analyze_score_distribution,
    compute_score_correlation,
    NormalizationMethod,
)
from metrics import (
    calculate_mrr,
    calculate_recall_at_k,
    calculate_hit_at_k,
    get_rank_from_scores,
    bootstrap_mrr_ci,
    paired_bootstrap_test,
)


# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
RAW_SCORES_PATH = ARTIFACTS_DIR / "raw_scores.json"

# Alpha values for coarse sweep
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Normalization strategies
NORMALIZATION_METHODS: list[NormalizationMethod] = ["min_max", "z_score", "rank_percentile"]

# Default RRF parameters (used when per-query params not available)
DEFAULT_RRF_K = 60
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_SEM_WEIGHT = 1.0

# Metrics settings
METRICS_K = 10

# Known baseline MRR values (from EVALUATION_CRITERIA.md)
EXPECTED_BASELINES = {
    "informed": 0.621,
    "needle": 0.842,
    "realistic": 0.196,
}


@dataclass
class QueryResult:
    """Results for a single query."""
    query_id: str
    query: str
    ground_truth_doc_id: str
    rrf_rank: Optional[int]
    convex_rank: Optional[int]
    rank_delta: Optional[int]  # positive = convex better


@dataclass
class SweepResult:
    """Results for a single (normalization, alpha) combination."""
    normalization: str
    alpha: float
    mrr: float
    recall_at_k: float
    hit_at_1: float
    hit_at_k: float
    per_query: list[QueryResult]


def load_raw_scores() -> dict:
    """Load raw scores from JSON file."""
    with open(RAW_SCORES_PATH) as f:
        return json.load(f)


def find_ground_truth_indices(
    chunk_doc_ids: dict[str, str],  # Note: JSON keys are strings
    ground_truth_doc_id: str,
) -> list[int]:
    """Find chunk indices belonging to ground truth document."""
    return [
        int(idx) for idx, doc_id in chunk_doc_ids.items()
        if ground_truth_doc_id in doc_id
    ]


def compute_rrf_baseline(queries: list[dict], k: int = METRICS_K) -> dict:
    """Compute RRF baseline metrics using per-query RRF parameters."""
    ranks = []
    per_query = []
    
    for q in queries:
        bm25_scores = {int(idx): v for idx, v in q["bm25_scores"].items()}
        semantic_scores = {int(idx): v for idx, v in q["semantic_scores"].items()}
        chunk_doc_ids = q["chunk_doc_ids"]
        ground_truth = q["ground_truth_doc_id"]
        
        rrf_k = q.get("rrf_k", DEFAULT_RRF_K)
        bm25_weight = q.get("bm25_weight", DEFAULT_BM25_WEIGHT)
        sem_weight = q.get("sem_weight", DEFAULT_SEM_WEIGHT)
        
        rrf_scores = rrf_fusion(
            bm25_scores,
            semantic_scores,
            k=rrf_k,
            bm25_weight=bm25_weight,
            semantic_weight=sem_weight,
        )
        
        gt_indices = find_ground_truth_indices(chunk_doc_ids, ground_truth)
        rank = get_rank_from_scores(rrf_scores, gt_indices)
        ranks.append(rank)
        
        per_query.append({
            "query_id": q["query_id"],
            "rank": rank,
        })
    
    return {
        "mrr": calculate_mrr(ranks, at_k=k),
        "recall_at_k": calculate_recall_at_k(ranks, k=k),
        "hit_at_1": calculate_hit_at_k(ranks, k=1),
        "hit_at_k": calculate_hit_at_k(ranks, k=k),
        "ranks": ranks,
        "per_query": per_query,
    }


def run_convex_sweep(
    queries: list[dict],
    normalization: NormalizationMethod,
    alpha: float,
    rrf_ranks: list[Optional[int]],
    k: int = METRICS_K,
) -> SweepResult:
    """Run convex fusion for a specific (normalization, alpha) combination.
    
    Args:
        queries: List of query dicts from raw_scores.json
        normalization: Normalization method
        alpha: BM25 weight (0 = semantic only, 1 = BM25 only)
        rrf_ranks: Pre-computed RRF ranks for comparison
        k: Cutoff for @k metrics
        
    Returns:
        SweepResult with metrics and per-query results
    """
    ranks = []
    per_query_results = []
    
    for i, q in enumerate(queries):
        bm25_scores = {int(k): v for k, v in q["bm25_scores"].items()}
        semantic_scores = {int(k): v for k, v in q["semantic_scores"].items()}
        chunk_doc_ids = q["chunk_doc_ids"]
        ground_truth = q["ground_truth_doc_id"]
        
        # Compute convex fusion scores
        convex_scores = convex_fusion(
            bm25_scores,
            semantic_scores,
            alpha=alpha,
            normalization=normalization,
        )
        
        # Find ground truth rank
        gt_indices = find_ground_truth_indices(chunk_doc_ids, ground_truth)
        rank = get_rank_from_scores(convex_scores, gt_indices)
        ranks.append(rank)
        
        # Compute rank delta (positive = convex better)
        rrf_rank = rrf_ranks[i]
        if rank is not None and rrf_rank is not None:
            rank_delta = rrf_rank - rank
        else:
            rank_delta = None
        
        per_query_results.append(QueryResult(
            query_id=q["query_id"],
            query=q["query"],
            ground_truth_doc_id=ground_truth,
            rrf_rank=rrf_rank,
            convex_rank=rank,
            rank_delta=rank_delta,
        ))
    
    return SweepResult(
        normalization=normalization,
        alpha=alpha,
        mrr=calculate_mrr(ranks, at_k=k),
        recall_at_k=calculate_recall_at_k(ranks, k=k),
        hit_at_1=calculate_hit_at_k(ranks, k=1),
        hit_at_k=calculate_hit_at_k(ranks, k=k),
        per_query=[asdict(q) for q in per_query_results],
    )


def run_full_sweep(
    queries: list[dict],
    label: str,
    rrf_baseline: dict,
) -> dict:
    """Run full alpha sweep across all normalizations.
    
    Args:
        queries: List of query dicts
        label: Query set label (informed, needle, realistic)
        rrf_baseline: Pre-computed RRF baseline
        
    Returns:
        Dict with all sweep results
    """
    rrf_ranks = rrf_baseline["ranks"]
    results = {
        "label": label,
        "rrf_baseline": {
            "mrr": rrf_baseline["mrr"],
            "recall_at_k": rrf_baseline["recall_at_k"],
            "hit_at_1": rrf_baseline["hit_at_1"],
            "hit_at_k": rrf_baseline["hit_at_k"],
        },
        "sweeps": [],
        "best": None,
    }
    
    best_mrr = rrf_baseline["mrr"]
    best_config = None
    
    total_combinations = len(NORMALIZATION_METHODS) * len(ALPHA_VALUES)
    pbar = tqdm(total=total_combinations, desc=f"Sweep ({label})")
    
    for norm in NORMALIZATION_METHODS:
        for alpha in ALPHA_VALUES:
            sweep_result = run_convex_sweep(
                queries, norm, alpha, rrf_ranks
            )
            
            results["sweeps"].append({
                "normalization": sweep_result.normalization,
                "alpha": sweep_result.alpha,
                "mrr": sweep_result.mrr,
                "recall_at_k": sweep_result.recall_at_k,
                "hit_at_1": sweep_result.hit_at_1,
                "hit_at_k": sweep_result.hit_at_k,
            })
            
            if sweep_result.mrr > best_mrr:
                best_mrr = sweep_result.mrr
                best_config = {
                    "normalization": norm,
                    "alpha": alpha,
                    "mrr": sweep_result.mrr,
                    "improvement": sweep_result.mrr - rrf_baseline["mrr"],
                    "improvement_pct": (sweep_result.mrr - rrf_baseline["mrr"]) / rrf_baseline["mrr"] * 100,
                }
            
            pbar.update(1)
    
    pbar.close()
    results["best"] = best_config
    
    return results


def run_fine_grained_sweep(
    queries: list[dict],
    normalization: NormalizationMethod,
    alpha_center: float,
    rrf_ranks: list[Optional[int]],
    step: float = 0.02,
    radius: float = 0.1,
) -> list[dict]:
    """Run fine-grained alpha sweep around optimal value.
    
    Args:
        queries: List of query dicts
        normalization: Normalization method
        alpha_center: Center alpha value
        rrf_ranks: Pre-computed RRF ranks
        step: Alpha step size
        radius: Search radius around center
        
    Returns:
        List of (alpha, mrr) results
    """
    results = []
    
    alpha_min = max(0.0, alpha_center - radius)
    alpha_max = min(1.0, alpha_center + radius)
    
    alpha = alpha_min
    while alpha <= alpha_max:
        sweep_result = run_convex_sweep(queries, normalization, alpha, rrf_ranks)
        results.append({
            "alpha": round(alpha, 3),
            "mrr": sweep_result.mrr,
        })
        alpha += step
    
    return results


def run_statistical_tests(
    queries: list[dict],
    normalization: NormalizationMethod,
    alpha: float,
    rrf_baseline: dict,
) -> dict:
    """Run statistical significance tests for convex vs RRF.
    
    Args:
        queries: List of query dicts
        normalization: Normalization method
        alpha: Alpha value
        rrf_baseline: Pre-computed RRF baseline
        
    Returns:
        Dict with CI and p-value
    """
    rrf_ranks = rrf_baseline["ranks"]
    
    # Compute convex ranks
    convex_ranks = []
    for q in queries:
        bm25_scores = {int(k): v for k, v in q["bm25_scores"].items()}
        semantic_scores = {int(k): v for k, v in q["semantic_scores"].items()}
        chunk_doc_ids = q["chunk_doc_ids"]
        ground_truth = q["ground_truth_doc_id"]
        
        convex_scores = convex_fusion(bm25_scores, semantic_scores, alpha, normalization)
        gt_indices = find_ground_truth_indices(chunk_doc_ids, ground_truth)
        rank = get_rank_from_scores(convex_scores, gt_indices)
        convex_ranks.append(rank)
    
    # Compute reciprocal ranks for bootstrap
    rrf_rr = [1.0/r if r and r <= METRICS_K else 0.0 for r in rrf_ranks]
    convex_rr = [1.0/r if r and r <= METRICS_K else 0.0 for r in convex_ranks]
    
    # Bootstrap CIs
    rrf_ci = bootstrap_mrr_ci(rrf_ranks, at_k=METRICS_K)
    convex_ci = bootstrap_mrr_ci(convex_ranks, at_k=METRICS_K)
    
    # Paired bootstrap test
    mean_diff, p_value, significant = paired_bootstrap_test(convex_rr, rrf_rr)
    
    return {
        "rrf_ci_95": rrf_ci,
        "convex_ci_95": convex_ci,
        "mean_diff": mean_diff,
        "p_value": p_value,
        "significant": significant,
        "ci_overlap": rrf_ci[1] > convex_ci[0] and convex_ci[1] > rrf_ci[0],
    }


def analyze_per_query(
    queries: list[dict],
    normalization: NormalizationMethod,
    alpha: float,
    rrf_baseline: dict,
) -> dict:
    """Analyze per-query improvements and regressions.
    
    Args:
        queries: List of query dicts
        normalization: Normalization method
        alpha: Alpha value
        rrf_baseline: Pre-computed RRF baseline
        
    Returns:
        Dict with winners, losers, and patterns
    """
    rrf_ranks = rrf_baseline["ranks"]
    
    winners = []  # Queries where convex improves rank by >3
    losers = []   # Queries where convex worsens rank (any amount)
    unchanged = []
    
    for i, q in enumerate(queries):
        bm25_scores = {int(k): v for k, v in q["bm25_scores"].items()}
        semantic_scores = {int(k): v for k, v in q["semantic_scores"].items()}
        chunk_doc_ids = q["chunk_doc_ids"]
        ground_truth = q["ground_truth_doc_id"]
        
        convex_scores = convex_fusion(bm25_scores, semantic_scores, alpha, normalization)
        gt_indices = find_ground_truth_indices(chunk_doc_ids, ground_truth)
        convex_rank = get_rank_from_scores(convex_scores, gt_indices)
        rrf_rank = rrf_ranks[i]
        
        if rrf_rank is None and convex_rank is None:
            continue
        
        # Handle None ranks (treat as very bad rank for comparison)
        rrf_r = rrf_rank if rrf_rank else 100
        convex_r = convex_rank if convex_rank else 100
        
        delta = rrf_r - convex_r  # positive = convex better
        
        query_info = {
            "query_id": q["query_id"],
            "query": q["query"][:80],  # Truncate for readability
            "rrf_rank": rrf_rank,
            "convex_rank": convex_rank,
            "delta": delta,
        }
        
        if delta > 3:
            winners.append(query_info)
        elif delta < 0:
            losers.append(query_info)
        else:
            unchanged.append(query_info)
    
    return {
        "winners": sorted(winners, key=lambda x: x["delta"], reverse=True),
        "losers": sorted(losers, key=lambda x: x["delta"]),
        "unchanged_count": len(unchanged),
        "summary": {
            "improved": len(winners),
            "worsened": len(losers),
            "unchanged": len(unchanged),
        }
    }


def check_regression(
    baseline_mrr: float,
    actual_mrr: float,
    threshold_pct: float = 5.0,
) -> dict:
    """Check if there's a regression beyond threshold.
    
    Args:
        baseline_mrr: Expected baseline MRR
        actual_mrr: Actual MRR
        threshold_pct: Acceptable regression percentage
        
    Returns:
        Dict with regression info
    """
    diff = actual_mrr - baseline_mrr
    diff_pct = (diff / baseline_mrr) * 100 if baseline_mrr > 0 else 0
    
    return {
        "baseline_mrr": baseline_mrr,
        "actual_mrr": actual_mrr,
        "diff": diff,
        "diff_pct": diff_pct,
        "regression": diff_pct < -threshold_pct,
        "threshold_pct": threshold_pct,
    }


def main():
    """Run full convex fusion benchmark."""
    print("=" * 60)
    print("CONVEX FUSION BENCHMARK")
    print("=" * 60)
    
    # Load raw scores
    print("\nLoading raw scores...")
    raw_scores = load_raw_scores()
    
    informed = raw_scores["informed"]
    needle = raw_scores["needle"]
    realistic = raw_scores["realistic"]
    
    print(f"Loaded: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    # Phase 3: Compute RRF baselines
    print("\n" + "=" * 60)
    print("PHASE 3: RRF BASELINE")
    print("=" * 60)
    
    rrf_informed = compute_rrf_baseline(informed)
    rrf_needle = compute_rrf_baseline(needle)
    rrf_realistic = compute_rrf_baseline(realistic)
    
    print(f"\nRRF Baselines (MRR@{METRICS_K}):")
    print(f"  Informed:  {rrf_informed['mrr']:.4f} (expected: {EXPECTED_BASELINES['informed']:.3f})")
    print(f"  Needle:    {rrf_needle['mrr']:.4f} (expected: {EXPECTED_BASELINES['needle']:.3f})")
    print(f"  Realistic: {rrf_realistic['mrr']:.4f} (expected: {EXPECTED_BASELINES['realistic']:.3f})")
    
    # Check baseline deviation
    for label, rrf, expected in [
        ("informed", rrf_informed, EXPECTED_BASELINES["informed"]),
        ("needle", rrf_needle, EXPECTED_BASELINES["needle"]),
        ("realistic", rrf_realistic, EXPECTED_BASELINES["realistic"]),
    ]:
        deviation = abs(rrf["mrr"] - expected) / expected * 100
        if deviation > 5:
            print(f"  WARNING: {label} deviates {deviation:.1f}% from expected!")
    
    # Phase 6: Alpha sweep on informed queries
    print("\n" + "=" * 60)
    print("PHASE 6: ALPHA SWEEP (INFORMED)")
    print("=" * 60)
    
    informed_sweep = run_full_sweep(informed, "informed", rrf_informed)
    
    print(f"\nBest configuration for informed queries:")
    if informed_sweep["best"]:
        best = informed_sweep["best"]
        print(f"  Normalization: {best['normalization']}")
        print(f"  Alpha: {best['alpha']}")
        print(f"  MRR: {best['mrr']:.4f}")
        print(f"  Improvement: {best['improvement']:+.4f} ({best['improvement_pct']:+.2f}%)")
    else:
        print("  No improvement over RRF baseline")
    
    # Save informed sweep results
    with open(ARTIFACTS_DIR / "alpha_sweep_informed.json", "w") as f:
        json.dump(informed_sweep, f, indent=2)
    print(f"\nSaved: artifacts/alpha_sweep_informed.json")
    
    # Phase 7: Fine-grained tuning if improvement found
    print("\n" + "=" * 60)
    print("PHASE 7: FINE-GRAINED TUNING")
    print("=" * 60)
    
    fine_grained_results = None
    statistical_results = None
    
    if informed_sweep["best"] and informed_sweep["best"]["improvement_pct"] > 0:
        best_norm = informed_sweep["best"]["normalization"]
        best_alpha = informed_sweep["best"]["alpha"]
        
        print(f"\nFine-grained sweep around alpha={best_alpha} with {best_norm}...")
        fine_grained = run_fine_grained_sweep(
            informed, best_norm, best_alpha, rrf_informed["ranks"],
            step=0.02, radius=0.1
        )
        
        optimal = max(fine_grained, key=lambda x: x["mrr"])
        print(f"Optimal alpha: {optimal['alpha']:.3f} (MRR: {optimal['mrr']:.4f})")
        
        fine_grained_results = {
            "normalization": best_norm,
            "center_alpha": best_alpha,
            "results": fine_grained,
            "optimal_alpha": optimal["alpha"],
            "optimal_mrr": optimal["mrr"],
        }
        
        # Statistical significance
        print("\nStatistical significance testing...")
        statistical_results = run_statistical_tests(
            informed, best_norm, optimal["alpha"], rrf_informed
        )
        
        print(f"  RRF 95% CI: [{statistical_results['rrf_ci_95'][0]:.4f}, {statistical_results['rrf_ci_95'][1]:.4f}]")
        print(f"  Convex 95% CI: [{statistical_results['convex_ci_95'][0]:.4f}, {statistical_results['convex_ci_95'][1]:.4f}]")
        print(f"  Mean difference: {statistical_results['mean_diff']:+.4f}")
        print(f"  p-value: {statistical_results['p_value']:.4f}")
        print(f"  Significant (p<0.05): {statistical_results['significant']}")
    else:
        print("No improvement found - skipping fine-grained tuning")
    
    # Phase 8: Per-query analysis
    print("\n" + "=" * 60)
    print("PHASE 8: PER-QUERY ANALYSIS")
    print("=" * 60)
    
    per_query_analysis = None
    
    if informed_sweep["best"]:
        best_norm = informed_sweep["best"]["normalization"]
        best_alpha = fine_grained_results["optimal_alpha"] if fine_grained_results else informed_sweep["best"]["alpha"]
        
        print(f"\nAnalyzing per-query results for alpha={best_alpha}, norm={best_norm}...")
        per_query_analysis = analyze_per_query(informed, best_norm, best_alpha, rrf_informed)
        
        summary = per_query_analysis["summary"]
        print(f"  Improved (>3 ranks): {summary['improved']}")
        print(f"  Worsened: {summary['worsened']}")
        print(f"  Unchanged: {summary['unchanged']}")
        
        if per_query_analysis["winners"]:
            print("\n  Top 3 Winners:")
            for w in per_query_analysis["winners"][:3]:
                print(f"    {w['query_id']}: rank {w['rrf_rank']} → {w['convex_rank']} (delta={w['delta']:+d})")
        
        if per_query_analysis["losers"]:
            print("\n  Top 3 Losers:")
            for l in per_query_analysis["losers"][:3]:
                print(f"    {l['query_id']}: rank {l['rrf_rank']} → {l['convex_rank']} (delta={l['delta']:+d})")
    
    # Phase 9: Regression testing
    print("\n" + "=" * 60)
    print("PHASE 9: REGRESSION TESTING")
    print("=" * 60)
    
    regression_results = {}
    
    if informed_sweep["best"]:
        best_norm = informed_sweep["best"]["normalization"]
        best_alpha = fine_grained_results["optimal_alpha"] if fine_grained_results else informed_sweep["best"]["alpha"]
        
        print(f"\nTesting alpha={best_alpha}, norm={best_norm} on other benchmarks...")
        
        # Needle
        needle_result = run_convex_sweep(needle, best_norm, best_alpha, rrf_needle["ranks"])
        needle_regression = check_regression(
            EXPECTED_BASELINES["needle"], needle_result.mrr
        )
        regression_results["needle"] = {
            "convex_mrr": needle_result.mrr,
            "rrf_mrr": rrf_needle["mrr"],
            "regression_check": needle_regression,
        }
        print(f"\n  Needle:")
        print(f"    RRF MRR: {rrf_needle['mrr']:.4f}")
        print(f"    Convex MRR: {needle_result.mrr:.4f}")
        print(f"    Regression: {'YES' if needle_regression['regression'] else 'NO'} ({needle_regression['diff_pct']:+.2f}%)")
        
        # Realistic
        realistic_result = run_convex_sweep(realistic, best_norm, best_alpha, rrf_realistic["ranks"])
        realistic_regression = check_regression(
            EXPECTED_BASELINES["realistic"], realistic_result.mrr
        )
        regression_results["realistic"] = {
            "convex_mrr": realistic_result.mrr,
            "rrf_mrr": rrf_realistic["mrr"],
            "regression_check": realistic_regression,
        }
        print(f"\n  Realistic:")
        print(f"    RRF MRR: {rrf_realistic['mrr']:.4f}")
        print(f"    Convex MRR: {realistic_result.mrr:.4f}")
        print(f"    Regression: {'YES' if realistic_regression['regression'] else 'NO'} ({realistic_regression['diff_pct']:+.2f}%)")
    
    # Save regression results
    with open(ARTIFACTS_DIR / "regression_check.json", "w") as f:
        json.dump(regression_results, f, indent=2)
    print(f"\nSaved: artifacts/regression_check.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    final_results = {
        "rrf_baselines": {
            "informed": rrf_informed["mrr"],
            "needle": rrf_needle["mrr"],
            "realistic": rrf_realistic["mrr"],
        },
        "informed_sweep": informed_sweep,
        "fine_grained": fine_grained_results,
        "statistical": statistical_results,
        "per_query": per_query_analysis,
        "regression": regression_results,
    }
    
    # Determine verdict
    verdict = "FAIL"
    if informed_sweep["best"]:
        improvement_pct = informed_sweep["best"]["improvement_pct"]
        needle_ok = not regression_results.get("needle", {}).get("regression_check", {}).get("regression", True)
        realistic_ok = not regression_results.get("realistic", {}).get("regression_check", {}).get("regression", True)
        significant = statistical_results.get("significant", False) if statistical_results else False
        
        if improvement_pct > 5 and needle_ok and realistic_ok and significant:
            verdict = "PASS"
        elif improvement_pct > 2 or (improvement_pct > 0 and needle_ok and realistic_ok):
            verdict = "PARTIAL"
    
    final_results["verdict"] = verdict
    
    print(f"\nVerdict: {verdict}")
    print(f"\nCriteria:")
    if informed_sweep["best"]:
        print(f"  MRR improvement >5%: {informed_sweep['best']['improvement_pct']:.2f}% {'✓' if informed_sweep['best']['improvement_pct'] > 5 else '✗'}")
    else:
        print(f"  MRR improvement >5%: No improvement ✗")
    
    if regression_results.get("needle"):
        needle_ok = not regression_results["needle"]["regression_check"]["regression"]
        print(f"  No needle regression >5%: {'✓' if needle_ok else '✗'}")
    
    if regression_results.get("realistic"):
        realistic_ok = not regression_results["realistic"]["regression_check"]["regression"]
        print(f"  No realistic regression >5%: {'✓' if realistic_ok else '✗'}")
    
    if statistical_results:
        print(f"  Statistical significance: {'✓' if statistical_results['significant'] else '✗'}")
    
    # Save final results
    with open(ARTIFACTS_DIR / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nSaved: artifacts/final_results.json")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    return verdict


if __name__ == "__main__":
    verdict = main()
    exit(0 if verdict in ["PASS", "PARTIAL"] else 1)
