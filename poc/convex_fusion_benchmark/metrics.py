#!/usr/bin/env python3
"""Metrics for convex fusion benchmark.

Implements MRR, Recall@k, Hit@k, and statistical utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def calculate_mrr(ranks: list[Optional[int]], at_k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank at k.
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        at_k: Only consider ranks <= at_k
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    reciprocals = []
    for rank in ranks:
        if rank is not None and rank <= at_k:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0


def calculate_recall_at_k(
    ranks: list[Optional[int]],
    k: int = 10,
) -> float:
    """Calculate Recall@k (fraction of queries where ground truth found in top k).
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        k: Number of top results to consider
        
    Returns:
        Recall@k as fraction (0.0 to 1.0)
    """
    found = sum(1 for rank in ranks if rank is not None and rank <= k)
    return found / len(ranks) if ranks else 0.0


def calculate_hit_at_k(ranks: list[Optional[int]], k: int = 1) -> float:
    """Calculate Hit@k (percentage of queries where ground truth found in top k).
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        k: Number of top results to consider
        
    Returns:
        Hit@k as percentage (0 to 100)
    """
    return calculate_recall_at_k(ranks, k) * 100


def get_rank_from_scores(
    scores: dict[int, float],
    ground_truth_indices: list[int],
) -> Optional[int]:
    """Get rank of ground truth document from score dict.
    
    Args:
        scores: Dict of chunk_index -> score
        ground_truth_indices: List of chunk indices belonging to ground truth doc
        
    Returns:
        1-indexed rank of first ground truth chunk found, or None if not in scores
    """
    if not scores or not ground_truth_indices:
        return None
    
    # Sort by score descending
    sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Find rank of first ground truth chunk
    for rank, idx in enumerate(sorted_indices, 1):
        if idx in ground_truth_indices:
            return rank
    
    return None


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval.
    
    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)
        statistic: Statistic to compute ("mean" or "median")
        
    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    if not values:
        return (0.0, 0.0)
    
    values_arr = np.array(values)
    bootstrap_stats = []
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    for _ in range(n_bootstrap):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    lower = np.percentile(bootstrap_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + ci) / 2 * 100)
    
    return (float(lower), float(upper))


def bootstrap_mrr_ci(
    ranks: list[Optional[int]],
    at_k: int = 10,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for MRR.
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        at_k: Only consider ranks <= at_k
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        
    Returns:
        (lower_bound, upper_bound) of 95% CI for MRR
    """
    # Convert to reciprocal ranks
    rr = []
    for rank in ranks:
        if rank is not None and rank <= at_k:
            rr.append(1.0 / rank)
        else:
            rr.append(0.0)
    
    return bootstrap_ci(rr, n_bootstrap=n_bootstrap, ci=ci, statistic="mean")


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
) -> tuple[float, float, bool]:
    """Paired bootstrap test for significance.
    
    Tests if values_a is significantly different from values_b.
    
    Args:
        values_a: First list of values (e.g., reciprocal ranks for method A)
        values_b: Second list of values (e.g., reciprocal ranks for method B)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        (mean_diff, p_value, significant) where:
            mean_diff: mean(values_a) - mean(values_b)
            p_value: Two-sided p-value
            significant: True if p < 0.05
    """
    if len(values_a) != len(values_b):
        raise ValueError("Lists must have same length")
    
    n = len(values_a)
    arr_a = np.array(values_a)
    arr_b = np.array(values_b)
    
    # Observed difference
    observed_diff = float(np.mean(arr_a) - np.mean(arr_b))
    
    # Bootstrap test
    rng = np.random.default_rng(42)
    count_extreme = 0
    
    # Under null hypothesis, the two methods are equivalent
    # Pool the differences and randomly assign signs
    diffs = arr_a - arr_b
    
    for _ in range(n_bootstrap):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        bootstrap_diff = np.mean(diffs * signs)
        
        if abs(bootstrap_diff) >= abs(observed_diff):
            count_extreme += 1
    
    p_value = count_extreme / n_bootstrap
    
    return (float(observed_diff), p_value, p_value < 0.05)


def compute_rank_delta(rank_a: Optional[int], rank_b: Optional[int]) -> Optional[int]:
    """Compute rank improvement (positive = better for A).
    
    Args:
        rank_a: Rank in method A
        rank_b: Rank in method B
        
    Returns:
        rank_b - rank_a (positive means A is better), or None if either is None
    """
    if rank_a is None or rank_b is None:
        return None
    return rank_b - rank_a


class MetricsCalculator:
    """Calculator for benchmark metrics from score data."""
    
    def __init__(self, chunk_doc_ids: dict[int, str]):
        """Initialize calculator.
        
        Args:
            chunk_doc_ids: Mapping from chunk index to doc_id
        """
        self.chunk_doc_ids = chunk_doc_ids
    
    def find_ground_truth_indices(self, doc_id: str, valid_indices: set[int]) -> list[int]:
        """Find chunk indices belonging to a document.
        
        Args:
            doc_id: Document ID to find
            valid_indices: Set of indices to consider
            
        Returns:
            List of chunk indices belonging to this document
        """
        return [
            idx for idx in valid_indices
            if doc_id in self.chunk_doc_ids.get(idx, "")
        ]
    
    def calculate_all_metrics(
        self,
        scores: dict[int, float],
        ground_truth_doc_id: str,
        k: int = 10,
    ) -> dict:
        """Calculate all metrics for a single query.
        
        Args:
            scores: Dict of chunk_index -> score
            ground_truth_doc_id: Expected document ID
            k: Cutoff for @k metrics
            
        Returns:
            Dict with rank, mrr, hit@1, hit@k, recall@k
        """
        gt_indices = self.find_ground_truth_indices(
            ground_truth_doc_id, set(scores.keys())
        )
        
        rank = get_rank_from_scores(scores, gt_indices)
        
        return {
            "rank": rank,
            "mrr": 1.0 / rank if rank and rank <= k else 0.0,
            "hit_at_1": 1 if rank == 1 else 0,
            "hit_at_k": 1 if rank and rank <= k else 0,
            "recall_at_k": 1.0 if rank and rank <= k else 0.0,
        }
