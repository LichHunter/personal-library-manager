"""Metrics for benchmark evaluation.

Implements Hit@k, Reciprocal Rank, and NDCG@k for information retrieval evaluation.
"""

from __future__ import annotations

from math import log2


def hit_at_k(expected: list[str], retrieved: list[str], k: int) -> bool:
    """Check if any expected chunk is in retrieved[:k].
    
    Args:
        expected: List of expected (ground truth) chunk IDs.
        retrieved: List of retrieved chunk IDs (ranked order).
        k: Number of top results to consider.
        
    Returns:
        True if any expected chunk is in the top k results.
    """
    return any(cid in retrieved[:k] for cid in expected)


def first_relevant_rank(expected: list[str], retrieved: list[str]) -> int | None:
    """Find the 1-indexed rank of the first relevant result.
    
    Args:
        expected: List of expected (ground truth) chunk IDs.
        retrieved: List of retrieved chunk IDs (ranked order).
        
    Returns:
        1-indexed rank of first relevant result, or None if not found.
    """
    expected_set = set(expected)
    for i, cid in enumerate(retrieved, 1):
        if cid in expected_set:
            return i
    return None


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    """Calculate reciprocal rank of first relevant result.
    
    Args:
        expected: List of expected (ground truth) chunk IDs.
        retrieved: List of retrieved chunk IDs (ranked order).
        
    Returns:
        1/rank of first relevant result, or 0.0 if not found.
    """
    rank = first_relevant_rank(expected, retrieved)
    if rank is not None:
        return 1.0 / rank
    return 0.0


def ndcg_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.
    
    Uses binary relevance: 1 if chunk is in expected, 0 otherwise.
    
    Args:
        expected: List of expected (ground truth) chunk IDs.
        retrieved: List of retrieved chunk IDs (ranked order).
        k: Cutoff for NDCG calculation.
        
    Returns:
        NDCG@k score (0.0 to 1.0).
    """
    if not expected or k <= 0:
        return 0.0
    
    expected_set = set(expected)
    
    dcg = sum(
        1.0 / log2(i + 1)
        for i, cid in enumerate(retrieved[:k], 1)
        if cid in expected_set
    )
    
    ideal_relevant_count = min(len(expected), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_relevant_count + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        reciprocal_ranks: List of reciprocal rank scores.
        
    Returns:
        Mean of reciprocal ranks.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hit_rate(hit_results: list[bool]) -> float:
    """Calculate hit rate (proportion of hits).
    
    Args:
        hit_results: List of hit@k boolean results.
        
    Returns:
        Proportion of True values (0.0 to 1.0).
    """
    if not hit_results:
        return 0.0
    return sum(hit_results) / len(hit_results)


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a list of values.
    
    Args:
        values: List of numeric values.
        p: Percentile to calculate (0-100).
        
    Returns:
        Value at the given percentile.
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    interpolation_index = (n - 1) * p / 100.0
    lower_idx = int(interpolation_index)
    upper_idx = lower_idx + 1 if lower_idx + 1 < n else lower_idx
    
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    
    fraction = interpolation_index - lower_idx
    return sorted_values[lower_idx] + (sorted_values[upper_idx] - sorted_values[lower_idx]) * fraction
