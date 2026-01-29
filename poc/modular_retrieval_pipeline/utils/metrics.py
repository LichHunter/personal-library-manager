"""Retrieval evaluation metrics.

Provides standard IR metrics for evaluating retrieval quality:
- Rank calculation
- Hit@k metrics
- Mean Reciprocal Rank (MRR)
- Total score (combines LLM grade with position)
"""

from typing import Optional


class RetrievalMetrics:
    """Calculator for retrieval evaluation metrics.

    All methods are stateless and can be used as static methods.
    Instance creation allows for future configuration if needed.

    Position weights for total_score:
    - Rank 1:     1.0  (full credit - best possible)
    - Rank 2-3:   0.95 (small penalty)
    - Rank 4-5:   0.85 (moderate penalty)
    - Not found:  0.6  (significant penalty)
    """

    POSITION_WEIGHTS = {
        1: 1.0,
        2: 0.95,
        3: 0.95,
        4: 0.85,
        5: 0.85,
        None: 0.6,  # Not found
    }

    def calculate_rank(
        self, retrieved_chunks: list[dict], needle_doc_id: str
    ) -> Optional[int]:
        """Find rank of needle document in retrieved chunks (1-indexed).

        Args:
            retrieved_chunks: List of chunk dicts with 'doc_id' field
            needle_doc_id: Document ID to find

        Returns:
            1-indexed rank if found, None if not in top-k
        """
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.get("doc_id") == needle_doc_id:
                return i + 1  # 1-indexed rank
        return None

    def calculate_total_score(
        self, llm_grade: Optional[int], rank: Optional[int]
    ) -> Optional[float]:
        """Calculate total score = llm_grade × position_weight.

        Args:
            llm_grade: LLM grade 1-10 (or None)
            rank: Rank 1-5 (or None if not found)

        Returns:
            Total score (grade weighted by position), or None if grade is None
        """
        if llm_grade is None:
            return None
        weight = self.POSITION_WEIGHTS.get(rank, self.POSITION_WEIGHTS[None])
        return llm_grade * weight

    def calculate_mrr(self, results: list[dict]) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            results: List of result dicts with 'rank' field

        Returns:
            MRR score (0.0 to 1.0)
        """
        reciprocal_ranks = []
        for r in results:
            rank = r.get("rank")
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        return (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        )

    def calculate_pass_rates(
        self, results: list[dict], thresholds: list[float]
    ) -> dict[float, float]:
        """Calculate pass rates at multiple thresholds.

        Args:
            results: List of result dicts with 'total_score' field
            thresholds: List of score thresholds (e.g., [8.0, 7.0, 6.5])

        Returns:
            Dict mapping threshold to pass rate percentage
        """
        valid_scores = [r for r in results if r.get("total_score") is not None]
        total = len(results)
        rates = {}
        for threshold in thresholds:
            passed = len([r for r in valid_scores if r["total_score"] >= threshold])
            rates[threshold] = (passed / total * 100) if total > 0 else 0.0
        return rates
