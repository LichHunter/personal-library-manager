"""Metrics calculation for retrieval evaluation."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .retriever import RetrievalResult


@dataclass
class RetrievalMetrics:
    """Metrics for a single query's retrieval results."""
    query_id: str
    query: str
    
    # Document-level metrics
    doc_recall_at_1: float = 0.0
    doc_recall_at_3: float = 0.0
    doc_recall_at_5: float = 0.0
    doc_recall_at_10: float = 0.0
    
    # Chunk-level metrics
    chunk_hit_at_1: bool = False  # Did we hit a relevant chunk at rank 1?
    chunk_hit_at_5: bool = False
    chunk_hit_at_10: bool = False
    
    # MRR (Mean Reciprocal Rank)
    mrr: float = 0.0
    
    # First relevant document rank (0 if not found)
    first_relevant_rank: int = 0
    
    # Details
    expected_docs: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries."""
    num_queries: int = 0
    
    # Document recall averages
    avg_doc_recall_at_1: float = 0.0
    avg_doc_recall_at_3: float = 0.0
    avg_doc_recall_at_5: float = 0.0
    avg_doc_recall_at_10: float = 0.0
    
    # Chunk hit rates
    chunk_hit_rate_at_1: float = 0.0
    chunk_hit_rate_at_5: float = 0.0
    chunk_hit_rate_at_10: float = 0.0
    
    # MRR
    mean_mrr: float = 0.0
    
    # Per-category breakdown
    category_metrics: dict = field(default_factory=dict)


def calculate_metrics(
    query_id: str,
    query: str,
    results: list[RetrievalResult],
    expected_docs: list[str],
    expected_chunks: Optional[list[str]] = None,
) -> RetrievalMetrics:
    """Calculate retrieval metrics for a single query.
    
    Args:
        query_id: Unique query identifier.
        query: The query string.
        results: List of retrieval results.
        expected_docs: List of expected document IDs.
        expected_chunks: Optional list of expected chunk IDs (for chunk-level metrics).
        
    Returns:
        RetrievalMetrics for this query.
    """
    metrics = RetrievalMetrics(
        query_id=query_id,
        query=query,
        expected_docs=expected_docs,
    )
    
    # Get retrieved doc IDs at different cutoffs
    retrieved_docs = [r.doc_id for r in results]
    metrics.retrieved_docs = retrieved_docs[:10]
    
    # Document recall at K
    expected_set = set(expected_docs)
    
    def recall_at_k(k: int) -> float:
        if not expected_docs:
            return 0.0
        retrieved_at_k = set(retrieved_docs[:k])
        hits = len(expected_set & retrieved_at_k)
        return hits / len(expected_docs)
    
    metrics.doc_recall_at_1 = recall_at_k(1)
    metrics.doc_recall_at_3 = recall_at_k(3)
    metrics.doc_recall_at_5 = recall_at_k(5)
    metrics.doc_recall_at_10 = recall_at_k(10)
    
    # Find first relevant document rank
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in expected_set:
            metrics.first_relevant_rank = i + 1
            metrics.mrr = 1.0 / (i + 1)
            break
    
    # Chunk-level hit metrics (if expected chunks provided)
    if expected_chunks:
        expected_chunk_set = set(expected_chunks)
        retrieved_chunks = [r.chunk_id for r in results]
        
        metrics.chunk_hit_at_1 = any(c in expected_chunk_set for c in retrieved_chunks[:1])
        metrics.chunk_hit_at_5 = any(c in expected_chunk_set for c in retrieved_chunks[:5])
        metrics.chunk_hit_at_10 = any(c in expected_chunk_set for c in retrieved_chunks[:10])
    else:
        # Fall back to doc-level for chunk hit metrics
        metrics.chunk_hit_at_1 = any(r.doc_id in expected_set for r in results[:1])
        metrics.chunk_hit_at_5 = any(r.doc_id in expected_set for r in results[:5])
        metrics.chunk_hit_at_10 = any(r.doc_id in expected_set for r in results[:10])
    
    return metrics


def aggregate_metrics(query_metrics: list[RetrievalMetrics]) -> AggregateMetrics:
    """Aggregate metrics across multiple queries.
    
    Args:
        query_metrics: List of per-query metrics.
        
    Returns:
        AggregateMetrics with averages and per-category breakdown.
    """
    if not query_metrics:
        return AggregateMetrics()
    
    n = len(query_metrics)
    
    agg = AggregateMetrics(
        num_queries=n,
        avg_doc_recall_at_1=sum(m.doc_recall_at_1 for m in query_metrics) / n,
        avg_doc_recall_at_3=sum(m.doc_recall_at_3 for m in query_metrics) / n,
        avg_doc_recall_at_5=sum(m.doc_recall_at_5 for m in query_metrics) / n,
        avg_doc_recall_at_10=sum(m.doc_recall_at_10 for m in query_metrics) / n,
        chunk_hit_rate_at_1=sum(1 for m in query_metrics if m.chunk_hit_at_1) / n,
        chunk_hit_rate_at_5=sum(1 for m in query_metrics if m.chunk_hit_at_5) / n,
        chunk_hit_rate_at_10=sum(1 for m in query_metrics if m.chunk_hit_at_10) / n,
        mean_mrr=sum(m.mrr for m in query_metrics) / n,
    )
    
    # Group by category (extracted from query_id)
    categories = {}
    for m in query_metrics:
        # Assume query_id format: "category_XX"
        parts = m.query_id.rsplit('_', 1)
        if len(parts) == 2:
            category = parts[0]
        else:
            category = "unknown"
        
        if category not in categories:
            categories[category] = []
        categories[category].append(m)
    
    # Calculate per-category metrics
    for category, cat_metrics in categories.items():
        cat_n = len(cat_metrics)
        agg.category_metrics[category] = {
            "num_queries": cat_n,
            "avg_doc_recall_at_5": sum(m.doc_recall_at_5 for m in cat_metrics) / cat_n,
            "avg_mrr": sum(m.mrr for m in cat_metrics) / cat_n,
            "hit_rate_at_5": sum(1 for m in cat_metrics if m.chunk_hit_at_5) / cat_n,
        }
    
    return agg


@dataclass
class ChunkStats:
    """Statistics about chunk sizes."""
    num_chunks: int = 0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    std_tokens: float = 0.0
    
    # Size distribution
    pct_under_100: float = 0.0
    pct_100_300: float = 0.0
    pct_300_500: float = 0.0
    pct_500_800: float = 0.0
    pct_over_800: float = 0.0


def calculate_chunk_stats(chunks: list) -> ChunkStats:
    """Calculate statistics about chunk sizes.
    
    Args:
        chunks: List of Chunk objects.
        
    Returns:
        ChunkStats with size distribution info.
    """
    if not chunks:
        return ChunkStats()
    
    # Get token counts (using the property on Chunk)
    token_counts = [c.token_count for c in chunks]
    
    n = len(token_counts)
    total = sum(token_counts)
    
    stats = ChunkStats(
        num_chunks=n,
        total_tokens=total,
        avg_tokens=total / n,
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        std_tokens=float(np.std(token_counts)) if n > 1 else 0.0,
    )
    
    # Size distribution
    stats.pct_under_100 = sum(1 for t in token_counts if t < 100) / n
    stats.pct_100_300 = sum(1 for t in token_counts if 100 <= t < 300) / n
    stats.pct_300_500 = sum(1 for t in token_counts if 300 <= t < 500) / n
    stats.pct_500_800 = sum(1 for t in token_counts if 500 <= t < 800) / n
    stats.pct_over_800 = sum(1 for t in token_counts if t >= 800) / n
    
    return stats
