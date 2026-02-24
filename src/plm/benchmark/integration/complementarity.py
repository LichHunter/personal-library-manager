"""Complementarity analysis for BM25 and semantic retrieval.

Measures overlap and unique contributions of each retriever to understand
whether they provide complementary signal or redundant results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ComplementarityResult:
    """Result of complementarity analysis between BM25 and semantic retrievers.

    Attributes:
        overlap_at_10: Fraction of results appearing in both retrievers' top 10.
        overlap_at_50: Fraction of results appearing in both retrievers' top 50.
        bm25_unique_hits: Count of relevant docs found ONLY by BM25.
        semantic_unique_hits: Count of relevant docs found ONLY by semantic.
        error_correlation: Correlation between BM25 and semantic miss patterns.
            High correlation (>0.4) indicates retrievers fail on same queries.
        fusion_potential: Score (0-1) indicating potential benefit of fusion.
            Higher = retrievers are more complementary.
    """

    overlap_at_10: float
    overlap_at_50: float
    bm25_unique_hits: int
    semantic_unique_hits: int
    error_correlation: float
    fusion_potential: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_overlap_at_k(
    bm25_ranks: list[int | None],
    semantic_ranks: list[int | None],
    k: int,
) -> float:
    """Compute fraction of results in top-k of BOTH retrievers.

    Args:
        bm25_ranks: List of 0-indexed BM25 ranks per result (None if not retrieved).
        semantic_ranks: List of 0-indexed semantic ranks per result (None if not retrieved).
        k: Cutoff for top-k calculation.

    Returns:
        Fraction of results in top-k of both retrievers.
    """
    if not bm25_ranks or not semantic_ranks:
        return 0.0

    both_in_top_k = 0
    either_in_top_k = 0

    for bm25_rank, sem_rank in zip(bm25_ranks, semantic_ranks):
        bm25_in_top = bm25_rank is not None and bm25_rank < k
        sem_in_top = sem_rank is not None and sem_rank < k

        if bm25_in_top and sem_in_top:
            both_in_top_k += 1
        if bm25_in_top or sem_in_top:
            either_in_top_k += 1

    if either_in_top_k == 0:
        return 0.0

    return both_in_top_k / either_in_top_k


def _compute_unique_hits(
    debug_info_list: list[list[dict[str, Any]]],
    expected_ids_list: list[list[str]],
    retrieved_ids_list: list[list[str]],
) -> tuple[int, int]:
    """Count relevant docs found ONLY by one retriever.

    Returns:
        Tuple of (bm25_unique_hits, semantic_unique_hits).
    """
    bm25_unique = 0
    semantic_unique = 0

    for debug_info, expected_ids, retrieved_ids in zip(
        debug_info_list, expected_ids_list, retrieved_ids_list
    ):
        expected_set = set(expected_ids)

        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id not in expected_set:
                continue

            if i >= len(debug_info):
                continue

            info = debug_info[i]
            bm25_rank = info.get("bm25_rank")
            sem_rank = info.get("semantic_rank")

            if bm25_rank is not None and sem_rank is None:
                bm25_unique += 1
            elif sem_rank is not None and bm25_rank is None:
                semantic_unique += 1

    return bm25_unique, semantic_unique


def _compute_error_correlation(
    debug_info_list: list[list[dict[str, Any]]],
    expected_ids_list: list[list[str]],
    retrieved_ids_list: list[list[str]],
    top_k: int = 50,
) -> float:
    """Compute correlation between BM25 and semantic miss patterns.

    A "miss" is when the retriever doesn't return the expected result in top-k.
    High correlation indicates both retrievers fail on the same queries.

    Returns:
        Pearson correlation coefficient (-1 to 1).
    """
    bm25_misses: list[int] = []
    semantic_misses: list[int] = []

    for debug_info, expected_ids, retrieved_ids in zip(
        debug_info_list, expected_ids_list, retrieved_ids_list
    ):
        expected_set = set(expected_ids)
        bm25_hit = False
        semantic_hit = False

        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id not in expected_set:
                continue
            if i >= len(debug_info):
                continue

            info = debug_info[i]
            bm25_rank = info.get("bm25_rank")
            sem_rank = info.get("semantic_rank")

            if bm25_rank is not None and bm25_rank < top_k:
                bm25_hit = True
            if sem_rank is not None and sem_rank < top_k:
                semantic_hit = True

        bm25_misses.append(0 if bm25_hit else 1)
        semantic_misses.append(0 if semantic_hit else 1)

    if not bm25_misses:
        return 0.0

    n = len(bm25_misses)
    mean_bm25 = sum(bm25_misses) / n
    mean_sem = sum(semantic_misses) / n

    numerator = sum(
        (b - mean_bm25) * (s - mean_sem)
        for b, s in zip(bm25_misses, semantic_misses)
    )

    var_bm25 = sum((b - mean_bm25) ** 2 for b in bm25_misses)
    var_sem = sum((s - mean_sem) ** 2 for s in semantic_misses)

    if var_bm25 == 0 or var_sem == 0:
        return 0.0

    denominator = (var_bm25 * var_sem) ** 0.5
    return numerator / denominator


def analyze_complementarity(
    per_query_results: list[dict[str, Any]],
) -> ComplementarityResult:
    """Analyze complementarity between BM25 and semantic retrievers.

    Args:
        per_query_results: List of PerQueryResult dicts from benchmark runner.
            Each must have 'debug_info', 'expected_chunk_ids', 'retrieved_chunk_ids'.

    Returns:
        ComplementarityResult with overlap and unique hit statistics.
    """
    if not per_query_results:
        return ComplementarityResult(
            overlap_at_10=0.0,
            overlap_at_50=0.0,
            bm25_unique_hits=0,
            semantic_unique_hits=0,
            error_correlation=0.0,
            fusion_potential=0.0,
        )

    all_bm25_ranks: list[int | None] = []
    all_semantic_ranks: list[int | None] = []
    debug_info_list: list[list[dict[str, Any]]] = []
    expected_ids_list: list[list[str]] = []
    retrieved_ids_list: list[list[str]] = []

    for result in per_query_results:
        debug_info = result.get("debug_info", [])
        expected_ids = result.get("expected_chunk_ids", [])
        retrieved_ids = result.get("retrieved_chunk_ids", [])

        debug_info_list.append(debug_info)
        expected_ids_list.append(expected_ids)
        retrieved_ids_list.append(retrieved_ids)

        for info in debug_info:
            all_bm25_ranks.append(info.get("bm25_rank"))
            all_semantic_ranks.append(info.get("semantic_rank"))

    overlap_10 = _compute_overlap_at_k(all_bm25_ranks, all_semantic_ranks, 10)
    overlap_50 = _compute_overlap_at_k(all_bm25_ranks, all_semantic_ranks, 50)

    bm25_unique, semantic_unique = _compute_unique_hits(
        debug_info_list, expected_ids_list, retrieved_ids_list
    )

    error_correlation = _compute_error_correlation(
        debug_info_list, expected_ids_list, retrieved_ids_list
    )

    total_unique = bm25_unique + semantic_unique
    fusion_potential = (1 - overlap_10) * 0.5 + (1 - max(0, error_correlation)) * 0.5
    if total_unique > 0:
        fusion_potential = min(1.0, fusion_potential + 0.1)

    return ComplementarityResult(
        overlap_at_10=overlap_10,
        overlap_at_50=overlap_50,
        bm25_unique_hits=bm25_unique,
        semantic_unique_hits=semantic_unique,
        error_correlation=error_correlation,
        fusion_potential=fusion_potential,
    )
