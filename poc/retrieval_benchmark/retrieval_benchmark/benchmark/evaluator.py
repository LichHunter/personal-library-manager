"""Metric computation for benchmark evaluation."""

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.types import (
    ContentMetrics,
    GroundTruth,
    QueryResult,
    RetrievedChunk,
    SearchResponse,
    StrategySummary,
)
from .content_metrics import compute_content_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context for a single evaluation."""
    strategy: str
    backend: str
    embedding_model: str
    llm_model: Optional[str]


class Evaluator:
    """Computes metrics from search results and ground truth."""
    
    def evaluate_query(
        self,
        query_id: str,
        run_number: int,
        ground_truth: GroundTruth,
        response: SearchResponse,
        context: EvaluationContext,
        top_k: int,
    ) -> QueryResult:
        """
        Evaluate a single query result against ground truth.
        
        Args:
            query_id: Unique query identifier
            run_number: Which run (1-indexed)
            ground_truth: Expected results
            response: Actual search response
            context: Strategy configuration context
            top_k: Number of results to consider
        
        Returns:
            QueryResult with metrics
        """
        hits = response.hits[:top_k]
        
        expected_doc = ground_truth.document_id
        expected_sections = set(ground_truth.section_ids)
        
        retrieved_chunks: list[RetrievedChunk] = []
        retrieved_doc_ids: list[str] = []
        retrieved_section_ids: list[str] = []
        
        doc_found = False
        doc_rank = -1
        section_found = False
        section_rank = -1
        
        for i, hit in enumerate(hits):
            rank = i + 1
            is_correct_doc = hit.document_id == expected_doc
            is_correct_section = hit.section_id in expected_sections
            
            retrieved_chunks.append(RetrievedChunk(
                rank=rank,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                section_id=hit.section_id,
                content=hit.content,
                score=hit.score,
                is_correct_doc=is_correct_doc,
                is_correct_section=is_correct_section,
            ))
            
            retrieved_doc_ids.append(hit.document_id)
            if hit.section_id:
                retrieved_section_ids.append(hit.section_id)
            
            if is_correct_doc and not doc_found:
                doc_found = True
                doc_rank = rank
            
            if is_correct_section and not section_found:
                section_found = True
                section_rank = rank
        
        chunk_contents = [c.content for c in retrieved_chunks]
        content_metrics = compute_content_metrics(
            retrieved_chunks=chunk_contents,
            expected_answer=ground_truth.answer,
            evidence_quotes=ground_truth.evidence,
            embedder=None,
        )
        
        return QueryResult(
            query_id=query_id,
            run_number=run_number,
            strategy=context.strategy,
            backend=context.backend,
            embedding_model=context.embedding_model,
            llm_model=context.llm_model,
            question=ground_truth.question,
            expected_doc_id=expected_doc,
            expected_section_ids=list(expected_sections),
            expected_answer=ground_truth.answer,
            expected_evidence=ground_truth.evidence,
            retrieved_chunks=retrieved_chunks,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_section_ids=retrieved_section_ids,
            top_k=top_k,
            doc_found=doc_found,
            doc_rank=doc_rank,
            section_found=section_found,
            section_rank=section_rank,
            content_metrics=content_metrics,
            search_time_ms=response.stats.duration_ms,
            embed_calls=response.stats.embed_calls,
            llm_calls=response.stats.llm_calls,
        )
    
    def compute_summary(
        self,
        results: list[QueryResult],
        index_time_sec: float,
        num_vectors: int,
        runs_per_query: int,
    ) -> StrategySummary:
        """
        Compute aggregated metrics from query results.
        
        Args:
            results: All query results for this configuration
            index_time_sec: Time spent indexing
            num_vectors: Number of vectors in index
            runs_per_query: Number of runs per query
        
        Returns:
            StrategySummary with aggregated metrics
        """
        if not results:
            raise ValueError("No results to summarize")
        
        # Get context from first result
        context = results[0]
        
        # Group by top_k for recall calculations
        results_by_k: dict[int, list[QueryResult]] = {}
        for r in results:
            if r.top_k not in results_by_k:
                results_by_k[r.top_k] = []
            results_by_k[r.top_k].append(r)
        
        # Compute recall at each k
        def recall_at_k(k: int, check_doc: bool) -> float:
            if k not in results_by_k:
                return 0.0
            k_results = results_by_k[k]
            if not k_results:
                return 0.0
            if check_doc:
                found = sum(1 for r in k_results if r.doc_found)
            else:
                found = sum(1 for r in k_results if r.section_found)
            return found / len(k_results)
        
        # Compute MRR (Mean Reciprocal Rank)
        def compute_mrr(check_doc: bool) -> float:
            # Use the largest k for MRR calculation
            max_k = max(results_by_k.keys())
            k_results = results_by_k[max_k]
            if not k_results:
                return 0.0
            
            reciprocal_ranks = []
            for r in k_results:
                rank = r.doc_rank if check_doc else r.section_rank
                if rank > 0:
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
            
            return sum(reciprocal_ranks) / len(reciprocal_ranks)
        
        # Compute timing statistics (use largest k)
        max_k = max(results_by_k.keys())
        search_times = [r.search_time_ms for r in results_by_k[max_k]]
        search_times_sorted = sorted(search_times)
        
        avg_search_time = sum(search_times) / len(search_times)
        p50_idx = int(len(search_times_sorted) * 0.5)
        p95_idx = min(int(len(search_times_sorted) * 0.95), len(search_times_sorted) - 1)
        p50_search_time = search_times_sorted[p50_idx]
        p95_search_time = search_times_sorted[p95_idx]
        
        # Compute consistency score
        # Group results by query_id, check if all runs got same top results
        consistency = self._compute_consistency(results)
        
        # Total calls
        total_llm_calls = sum(r.llm_calls for r in results)
        total_embed_calls = sum(r.embed_calls for r in results)
        
        # Unique queries evaluated
        unique_queries = len(set(r.query_id for r in results))
        
        return StrategySummary(
            strategy=context.strategy,
            backend=context.backend,
            embedding_model=context.embedding_model,
            llm_model=context.llm_model,
            index_time_sec=index_time_sec,
            num_vectors=num_vectors,
            doc_recall_at_1=recall_at_k(1, check_doc=True),
            doc_recall_at_3=recall_at_k(3, check_doc=True),
            doc_recall_at_5=recall_at_k(5, check_doc=True),
            doc_recall_at_10=recall_at_k(10, check_doc=True),
            section_recall_at_1=recall_at_k(1, check_doc=False),
            section_recall_at_3=recall_at_k(3, check_doc=False),
            section_recall_at_5=recall_at_k(5, check_doc=False),
            section_recall_at_10=recall_at_k(10, check_doc=False),
            doc_mrr=compute_mrr(check_doc=True),
            section_mrr=compute_mrr(check_doc=False),
            avg_search_time_ms=avg_search_time,
            p50_search_time_ms=p50_search_time,
            p95_search_time_ms=p95_search_time,
            consistency_score=consistency,
            total_llm_calls=total_llm_calls,
            total_embed_calls=total_embed_calls,
            queries_evaluated=unique_queries,
            runs_per_query=runs_per_query,
        )
    
    def _compute_consistency(self, results: list[QueryResult]) -> float:
        """
        Compute consistency score across runs.
        
        Consistency = % of queries where all runs returned the same
        top-k documents (for the largest k tested).
        """
        # Group by (query_id, top_k)
        by_query: dict[tuple[str, int], list[QueryResult]] = {}
        for r in results:
            key = (r.query_id, r.top_k)
            if key not in by_query:
                by_query[key] = []
            by_query[key].append(r)
        
        # Find largest k
        max_k = max(k for (_, k) in by_query.keys())
        
        # Check consistency for max_k queries
        consistent_count = 0
        total_queries = 0
        
        for (query_id, k), runs in by_query.items():
            if k != max_k:
                continue
            
            total_queries += 1
            
            if len(runs) <= 1:
                consistent_count += 1
                continue
            
            # Compare doc_ids across runs
            first_docs = tuple(runs[0].retrieved_doc_ids)
            all_same = all(
                tuple(r.retrieved_doc_ids) == first_docs 
                for r in runs[1:]
            )
            
            if all_same:
                consistent_count += 1
        
        if total_queries == 0:
            return 1.0
        
        return consistent_count / total_queries
