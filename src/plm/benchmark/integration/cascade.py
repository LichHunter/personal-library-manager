"""Cascade analysis for retrieval pipeline stages.

Measures contribution of each stage (BM25, semantic, RRF fusion, reranker)
to overall retrieval quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plm.benchmark.runner.metrics import hit_at_k, mean_reciprocal_rank, reciprocal_rank


@dataclass
class CascadeResult:
    """Result of cascade analysis measuring stage contributions.

    Attributes:
        bm25_recall_at_100: Fraction of queries with relevant doc in BM25 top-100.
        semantic_recall_at_100: Fraction of queries with relevant doc in semantic top-100.
        rrf_recall_at_50: Fraction of queries with relevant doc in RRF top-50.
        rrf_mrr: Mean Reciprocal Rank after RRF fusion.
        rerank_mrr: Mean Reciprocal Rank after reranking (None if no reranking).
        stage_contributions: Dict mapping stage name to recall delta vs previous stage.
    """

    bm25_recall_at_100: float
    semantic_recall_at_100: float
    rrf_recall_at_50: float
    rrf_mrr: float
    rerank_mrr: float | None
    stage_contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_recall_at_k_from_ranks(
    ranks: list[int | None],
    k: int,
) -> float:
    """Compute recall@k from list of ranks (None = not retrieved)."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r < k)
    return hits / len(ranks)


def _extract_first_relevant_rank(
    debug_info: list[dict[str, Any]],
    expected_ids: list[str],
    retrieved_ids: list[str],
    rank_key: str,
) -> int | None:
    """Extract the rank of first relevant result from debug_info."""
    expected_set = set(expected_ids)

    best_rank: int | None = None
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id not in expected_set:
            continue
        if i >= len(debug_info):
            continue

        rank = debug_info[i].get(rank_key)
        if rank is not None:
            if best_rank is None or rank < best_rank:
                best_rank = rank

    return best_rank


def analyze_cascade(
    per_query_results: list[dict[str, Any]],
) -> CascadeResult:
    """Analyze contribution of each retrieval stage.

    Args:
        per_query_results: List of PerQueryResult dicts from benchmark runner.
            Each must have 'debug_info', 'expected_chunk_ids', 'retrieved_chunk_ids'.

    Returns:
        CascadeResult with stage-by-stage metrics.
    """
    if not per_query_results:
        return CascadeResult(
            bm25_recall_at_100=0.0,
            semantic_recall_at_100=0.0,
            rrf_recall_at_50=0.0,
            rrf_mrr=0.0,
            rerank_mrr=None,
            stage_contributions={},
        )

    bm25_first_ranks: list[int | None] = []
    semantic_first_ranks: list[int | None] = []
    rrf_reciprocal_ranks: list[float] = []
    rerank_reciprocal_ranks: list[float] = []
    has_rerank_scores = False

    for result in per_query_results:
        debug_info = result.get("debug_info", [])
        expected_ids = result.get("expected_chunk_ids", [])
        retrieved_ids = result.get("retrieved_chunk_ids", [])

        bm25_rank = _extract_first_relevant_rank(
            debug_info, expected_ids, retrieved_ids, "bm25_rank"
        )
        semantic_rank = _extract_first_relevant_rank(
            debug_info, expected_ids, retrieved_ids, "semantic_rank"
        )

        bm25_first_ranks.append(bm25_rank)
        semantic_first_ranks.append(semantic_rank)

        rrf_rr = reciprocal_rank(expected_ids, retrieved_ids)
        rrf_reciprocal_ranks.append(rrf_rr)

        if debug_info and debug_info[0].get("rerank_score") is not None:
            has_rerank_scores = True
            reranked_order = _get_reranked_order(debug_info, retrieved_ids)
            rerank_rr = reciprocal_rank(expected_ids, reranked_order)
            rerank_reciprocal_ranks.append(rerank_rr)

    bm25_recall_100 = _compute_recall_at_k_from_ranks(bm25_first_ranks, 100)
    semantic_recall_100 = _compute_recall_at_k_from_ranks(semantic_first_ranks, 100)

    rrf_hits_50 = sum(
        1 for result in per_query_results
        if hit_at_k(result["expected_chunk_ids"], result["retrieved_chunk_ids"], 50)
    )
    rrf_recall_50 = rrf_hits_50 / len(per_query_results)

    rrf_mrr = mean_reciprocal_rank(rrf_reciprocal_ranks)
    rerank_mrr = mean_reciprocal_rank(rerank_reciprocal_ranks) if has_rerank_scores else None

    union_recall_100 = _compute_union_recall(
        bm25_first_ranks, semantic_first_ranks, 100
    )

    stage_contributions = {
        "bm25_to_union": union_recall_100 - bm25_recall_100,
        "semantic_to_union": union_recall_100 - semantic_recall_100,
        "union_to_rrf": rrf_recall_50 - union_recall_100,
    }
    if rerank_mrr is not None:
        stage_contributions["rrf_to_rerank"] = rerank_mrr - rrf_mrr

    return CascadeResult(
        bm25_recall_at_100=bm25_recall_100,
        semantic_recall_at_100=semantic_recall_100,
        rrf_recall_at_50=rrf_recall_50,
        rrf_mrr=rrf_mrr,
        rerank_mrr=rerank_mrr,
        stage_contributions=stage_contributions,
    )


def _get_reranked_order(
    debug_info: list[dict[str, Any]],
    retrieved_ids: list[str],
) -> list[str]:
    """Re-sort retrieved_ids by rerank_score descending."""
    scored = []
    for i, chunk_id in enumerate(retrieved_ids):
        if i < len(debug_info):
            score = debug_info[i].get("rerank_score", 0.0) or 0.0
            scored.append((chunk_id, score))
        else:
            scored.append((chunk_id, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in scored]


def _compute_union_recall(
    bm25_ranks: list[int | None],
    semantic_ranks: list[int | None],
    k: int,
) -> float:
    """Compute recall@k for union of both retrievers."""
    if not bm25_ranks:
        return 0.0

    hits = 0
    for bm25_r, sem_r in zip(bm25_ranks, semantic_ranks):
        bm25_hit = bm25_r is not None and bm25_r < k
        sem_hit = sem_r is not None and sem_r < k
        if bm25_hit or sem_hit:
            hits += 1

    return hits / len(bm25_ranks)
