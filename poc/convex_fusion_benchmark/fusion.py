#!/usr/bin/env python3
"""Fusion methods for convex fusion benchmark.

Implements:
- RRF (Reciprocal Rank Fusion) - baseline
- Score normalization strategies (min-max, z-score, rank-percentile)
- Convex combination fusion (alpha * BM25 + (1-alpha) * semantic)
"""

from __future__ import annotations

from typing import Literal

import numpy as np


# =============================================================================
# RRF (Reciprocal Rank Fusion) - Baseline
# =============================================================================

def rrf_fusion(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
    k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> dict[int, float]:
    """Reciprocal Rank Fusion matching HybridRetriever behavior.
    
    CRITICAL: Semantic FIRST, BM25 SECOND (for POC parity).
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        k: RRF parameter (default 60)
        bm25_weight: Weight for BM25 component (default 1.0)
        semantic_weight: Weight for semantic component (default 1.0)
        
    Returns:
        Dict of chunk_index -> RRF fused score
    """
    rrf_scores: dict[int, float] = {}
    
    # Sort semantic scores descending
    sem_sorted = sorted(
        semantic_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # CRITICAL: Process semantic results FIRST
    for rank, (idx, _) in enumerate(sem_sorted):
        sem_component = semantic_weight / (k + rank)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_component
    
    # Sort BM25 scores descending
    bm25_sorted = sorted(
        bm25_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # CRITICAL: Process BM25 results SECOND
    for rank, (idx, _) in enumerate(bm25_sorted):
        bm25_component = bm25_weight / (k + rank)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_component
    
    return rrf_scores


# =============================================================================
# Normalization Strategies
# =============================================================================

NormalizationMethod = Literal["min_max", "z_score", "rank_percentile"]


def normalize_min_max(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalization to [0, 1] range.
    
    Args:
        scores: Dict of index -> score
        
    Returns:
        Dict of index -> normalized score in [0, 1]
    """
    if not scores:
        return {}
    
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    
    # Handle edge case: all scores identical
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    
    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }


def normalize_z_score(scores: dict[int, float], shift_positive: bool = True) -> dict[int, float]:
    """Z-score normalization (center around mean, unit variance).
    
    Args:
        scores: Dict of index -> score
        shift_positive: If True, shift all values to be non-negative
        
    Returns:
        Dict of index -> normalized score
    """
    if not scores:
        return {}
    
    values = np.array(list(scores.values()))
    mean = np.mean(values)
    std = np.std(values)
    
    # Handle edge case: zero variance
    if std == 0:
        return {k: 0.0 for k in scores}
    
    normalized = {
        k: float((v - mean) / std)
        for k, v in scores.items()
    }
    
    # Optionally shift to positive range
    if shift_positive:
        min_val = min(normalized.values())
        if min_val < 0:
            normalized = {k: float(v - min_val) for k, v in normalized.items()}
    
    return normalized


def normalize_rank_percentile(scores: dict[int, float]) -> dict[int, float]:
    """Rank-percentile normalization to [0, 1] range.
    
    Converts scores to percentile ranks. Handles ties by averaging.
    
    Args:
        scores: Dict of index -> score
        
    Returns:
        Dict of index -> percentile rank in [0, 1]
    """
    if not scores:
        return {}
    
    n = len(scores)
    if n == 1:
        return {list(scores.keys())[0]: 1.0}
    
    # Sort by score ascending (lower score = lower percentile)
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    
    # Assign percentile ranks (handle ties by averaging)
    result = {}
    i = 0
    while i < n:
        # Find all items with same score (ties)
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        
        # Average rank for ties (using 0-indexed positions)
        avg_rank = (i + j - 1) / 2
        percentile = avg_rank / (n - 1)  # Normalize to [0, 1]
        
        for k in range(i, j):
            result[sorted_items[k][0]] = percentile
        
        i = j
    
    return result


def normalize_scores(
    scores: dict[int, float],
    method: NormalizationMethod,
) -> dict[int, float]:
    """Normalize scores using specified method.
    
    Args:
        scores: Dict of index -> score
        method: Normalization method
        
    Returns:
        Dict of index -> normalized score
    """
    if method == "min_max":
        return normalize_min_max(scores)
    elif method == "z_score":
        return normalize_z_score(scores, shift_positive=True)
    elif method == "rank_percentile":
        return normalize_rank_percentile(scores)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Convex Combination Fusion
# =============================================================================

def convex_fusion(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
    alpha: float,
    normalization: NormalizationMethod = "min_max",
) -> dict[int, float]:
    """Convex combination fusion of BM25 and semantic scores.
    
    combined = alpha * bm25_normalized + (1 - alpha) * semantic_normalized
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        alpha: Weight for BM25 (0.0 = pure semantic, 1.0 = pure BM25)
        normalization: Score normalization method
        
    Returns:
        Dict of chunk_index -> fused score
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    
    # Normalize scores
    bm25_norm = normalize_scores(bm25_scores, normalization)
    sem_norm = normalize_scores(semantic_scores, normalization)
    
    # Get union of all indices
    all_indices = set(bm25_norm.keys()) | set(sem_norm.keys())
    
    # Compute convex combination
    # Documents appearing in only one retriever get 0 for the missing component
    fused = {}
    for idx in all_indices:
        bm25_val = bm25_norm.get(idx, 0.0)
        sem_val = sem_norm.get(idx, 0.0)
        fused[idx] = alpha * bm25_val + (1 - alpha) * sem_val
    
    return fused


def get_ranking_from_scores(scores: dict[int, float]) -> list[int]:
    """Get ranked list of indices from scores.
    
    Args:
        scores: Dict of index -> score
        
    Returns:
        List of indices sorted by score descending
    """
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_rrf_parity(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
    expected_top_k: list[int],
    k: int = 60,
) -> bool:
    """Validate that our RRF produces same ranking as expected.
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        expected_top_k: Expected top-k indices from retriever
        k: RRF parameter
        
    Returns:
        True if rankings match
    """
    rrf_scores = rrf_fusion(bm25_scores, semantic_scores, k=k)
    our_ranking = get_ranking_from_scores(rrf_scores)[:len(expected_top_k)]
    
    return our_ranking == expected_top_k


def validate_convex_extremes(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
    normalization: NormalizationMethod = "min_max",
) -> tuple[bool, bool]:
    """Validate convex fusion at extreme alpha values.
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        normalization: Normalization method to use
        
    Returns:
        (alpha_0_matches_semantic, alpha_1_matches_bm25)
    """
    # Alpha = 0.0 should produce semantic-only ranking
    fused_0 = convex_fusion(bm25_scores, semantic_scores, alpha=0.0, normalization=normalization)
    sem_norm = normalize_scores(semantic_scores, normalization)
    
    # Compare rankings (not exact scores, since missing docs get 0)
    ranking_0 = get_ranking_from_scores(fused_0)
    ranking_sem = get_ranking_from_scores(sem_norm)
    
    # Only compare indices that appear in both
    common_indices = set(fused_0.keys()) & set(sem_norm.keys())
    ranking_0_common = [i for i in ranking_0 if i in common_indices]
    ranking_sem_common = [i for i in ranking_sem if i in common_indices]
    
    alpha_0_ok = ranking_0_common == ranking_sem_common
    
    # Alpha = 1.0 should produce BM25-only ranking
    fused_1 = convex_fusion(bm25_scores, semantic_scores, alpha=1.0, normalization=normalization)
    bm25_norm = normalize_scores(bm25_scores, normalization)
    
    ranking_1 = get_ranking_from_scores(fused_1)
    ranking_bm25 = get_ranking_from_scores(bm25_norm)
    
    common_indices = set(fused_1.keys()) & set(bm25_norm.keys())
    ranking_1_common = [i for i in ranking_1 if i in common_indices]
    ranking_bm25_common = [i for i in ranking_bm25 if i in common_indices]
    
    alpha_1_ok = ranking_1_common == ranking_bm25_common
    
    return (alpha_0_ok, alpha_1_ok)


# =============================================================================
# Score Analysis
# =============================================================================

def analyze_score_distribution(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
) -> dict:
    """Analyze score distributions for a query.
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        
    Returns:
        Dict with distribution statistics
    """
    bm25_vals = np.array(list(bm25_scores.values())) if bm25_scores else np.array([0])
    sem_vals = np.array(list(semantic_scores.values())) if semantic_scores else np.array([0])
    
    # Overlap analysis
    bm25_indices = set(bm25_scores.keys())
    sem_indices = set(semantic_scores.keys())
    overlap = bm25_indices & sem_indices
    
    return {
        "bm25": {
            "min": float(bm25_vals.min()),
            "max": float(bm25_vals.max()),
            "mean": float(bm25_vals.mean()),
            "std": float(bm25_vals.std()),
            "count": len(bm25_scores),
        },
        "semantic": {
            "min": float(sem_vals.min()),
            "max": float(sem_vals.max()),
            "mean": float(sem_vals.mean()),
            "std": float(sem_vals.std()),
            "count": len(semantic_scores),
        },
        "overlap": {
            "count": len(overlap),
            "bm25_only": len(bm25_indices - sem_indices),
            "semantic_only": len(sem_indices - bm25_indices),
        },
    }


def compute_score_correlation(
    bm25_scores: dict[int, float],
    semantic_scores: dict[int, float],
) -> float:
    """Compute correlation between BM25 and semantic scores.
    
    Only considers documents appearing in both result sets.
    
    Args:
        bm25_scores: Dict of chunk_index -> BM25 score
        semantic_scores: Dict of chunk_index -> semantic score
        
    Returns:
        Pearson correlation coefficient, or 0.0 if no overlap
    """
    overlap = set(bm25_scores.keys()) & set(semantic_scores.keys())
    if len(overlap) < 2:
        return 0.0
    
    bm25_vals = [bm25_scores[i] for i in overlap]
    sem_vals = [semantic_scores[i] for i in overlap]
    
    corr = np.corrcoef(bm25_vals, sem_vals)[0, 1]
    
    return float(corr) if not np.isnan(corr) else 0.0
